"""
Core routines for the Streamlit demo: video -> observation matrix, truncated-SVD
baseline and Robust PCA via the Inexact Augmented Lagrangian Method (IALM).

The algorithms mirror video_decomposition_pipeline.ipynb (soft-thresholding, SVT, IALM).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ video I/O
def read_video_matrix(path: str | Path, width: int, height: int, max_frames: int,
                      start_frame: int = 0, step: int = 1, binarise: bool = False):
    """Return (M, fps, n_total): M is (height*width, n_frames) float32 in [0, 1]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cols, i = [], 0
    while len(cols) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            v = small.astype(np.float32).ravel() / 255.0
            cols.append((v > 0.5).astype(np.float32) if binarise else v)
        i += 1
    cap.release()
    if not cols:
        raise RuntimeError("No frames could be read from the video.")
    return np.column_stack(cols), fps, n_total


# ------------------------------------------------------------------ operators
def soft_threshold(X: np.ndarray, lam: float) -> np.ndarray:
    if lam < 0:
        raise ValueError("lam must be >= 0")
    return X.copy() if lam == 0 else np.sign(X) * np.maximum(np.abs(X) - lam, 0)


def svt(A: np.ndarray, tau: float):
    """Singular value thresholding: returns (low-rank approx, surviving rank)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_t = soft_threshold(s, tau)
    rank = int(np.sum(s_t > 0))
    if rank == 0:
        return np.zeros_like(A), 0
    return (U[:, :rank] * s_t[:rank]) @ Vt[:rank], rank


# ------------------------------------------------------------------ solvers
@dataclass
class Diagnostics:
    residuals: list = field(default_factory=list)
    rank_L: list = field(default_factory=list)
    sparsity_S: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    runtime_sec: float = 0.0
    lam: float = 0.0


def svd_baseline(M: np.ndarray, k: int = 1):
    """Rank-k truncated SVD: L = best rank-k approximation, S = M - L."""
    t0 = time.perf_counter()
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    L = (U[:, :k] * s[:k]) @ Vt[:k]
    return np.clip(L, 0, 1).astype(np.float32), (M - L).astype(np.float32), time.perf_counter() - t0


def rpca_ialm(M: np.ndarray, lam: float | None = None, rho: float = 1.2, mu_max: float = 1e4,
              max_iter: int = 100, tol: float = 1e-7, progress=None):
    """Solve min ||L||_* + lam ||S||_1  s.t.  L + S = M with IALM (Lin et al., 2010)."""
    m, n = M.shape
    M64 = M.astype(np.float64)
    M_norm = np.linalg.norm(M64, "fro")
    lam = lam if lam is not None else 1.0 / np.sqrt(max(m, n))
    mu = 1.25 / np.linalg.norm(M64, 2)
    L = np.zeros_like(M64); S = np.zeros_like(M64); Y = np.zeros_like(M64)
    d = Diagnostics(lam=lam)
    t0 = time.perf_counter()
    for k in range(max_iter):
        S = soft_threshold(M64 - L + Y / mu, lam / mu)
        L, rank = svt(M64 - S + Y / mu, 1.0 / mu)
        R = M64 - L - S
        Y = Y + mu * R
        mu = min(rho * mu, mu_max)
        res = np.linalg.norm(R, "fro") / M_norm
        d.residuals.append(res); d.rank_L.append(rank); d.sparsity_S.append(float(np.mean(S == 0)))
        if progress:
            progress((k + 1) / max_iter, res, rank)
        if res < tol:
            d.converged = True
            break
    d.n_iterations = len(d.residuals)
    d.runtime_sec = time.perf_counter() - t0
    return np.clip(L, 0, 1).astype(np.float32), S.astype(np.float32), d


# ------------------------------------------------------------------ metrics
def foreground_scores(S: np.ndarray, GT: np.ndarray, threshold: float = 0.10):
    """Pixel-level precision / recall / F1 of |S| > threshold against a binary GT mask."""
    pred, truth = np.abs(S) > threshold, GT > 0.5
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def psnr(ref: np.ndarray, est: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - est) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
