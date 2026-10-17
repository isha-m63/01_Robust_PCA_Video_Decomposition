"""
Streamlit demo: Robust PCA (IALM) for video background / foreground separation.

M = L + S   (L: low-rank static background, S: sparse moving foreground)
Compares a rank-k truncated SVD baseline with RPCA solved by IALM.
"""
import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from rpca_demo import read_video_matrix, svd_baseline, rpca_ialm, foreground_scores


ASSETS = Path(__file__).parent / "demo_assets"
SAMPLES = {
    "Synthetic street scene (BMC 2012 #111, with ground truth)": dict(video="synthetic_scene.mp4", gt="synthetic_scene_gt.mp4", start=125, frames=125, tol=1e-7),
    "Real surveillance clip (BMC 2012 Video_008)": dict(video="real_scene.mp4", gt=None, start=50, frames=100, tol=1e-2),
}
UPLOAD = "Upload your own video"
MAX_PIXELS = 160 * 120
MAX_FRAMES = 200

st.set_page_config(page_title="Robust PCA Video Decomposition", page_icon="🎬", layout="wide")
st.title("🎬 Robust PCA — Video Background / Foreground Separation")
st.caption("Observation matrix **M = L + S**. Each video frame is a column of M; RPCA (IALM) recovers a "
           "low-rank background **L** and a sparse moving foreground **S**.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Input")
    source = st.radio("Video", list(SAMPLES) + [UPLOAD])
    if source == UPLOAD:
        up = st.file_uploader("mp4 / avi / mov (short clips work best)", type=["mp4", "avi", "mov", "mkv"])
        sample, start, frames_default = None, 0, 60
    else:
        sample = SAMPLES[source]
        up, start, frames_default = None, sample["start"], sample["frames"]

    st.header("Settings")
    width = st.select_slider("Frame width (px)", [60, 80, 100, 120, 160], value=120)
    height = int(round(width * 0.75))
    n_frames = st.slider("Frames", 20, MAX_FRAMES, min(frames_default, MAX_FRAMES), step=5)
    step = st.select_slider("Frame step", [1, 2, 3, 5], value=1)
    svd_k = st.slider("SVD baseline rank k", 1, 10, 1)
    lam_scale = st.slider("RPCA λ multiplier", 0.5, 3.0, 1.0, 0.1,
                          help="λ = multiplier / √max(m, n). Larger → sparser foreground.")
    max_iter = st.slider("RPCA max iterations", 20, 250, 100, 10)
    tol = st.select_slider("RPCA tolerance", [1e-7, 1e-5, 1e-4, 1e-3, 1e-2], value=(sample or {}).get("tol", 1e-5),
                           format_func=lambda v: f"{v:.0e}", key=f"tol_{source}",
                           help="Relative residual at which IALM stops. Noisy real footage needs a looser value "
                                "(early stopping) so noise is not absorbed into a high-rank background.")
    run = st.button("Run decomposition", type="primary", use_container_width=True)


def get_video_path():
    if sample:
        return ASSETS / sample["video"]
    if up is None:
        return None
    tmp = Path(tempfile.gettempdir()) / f"upload_{abs(hash(up.name))}{Path(up.name).suffix}"
    tmp.write_bytes(up.getvalue())
    return tmp


def to_u8(vec, h, w):
    return (np.clip(vec, 0, 1).reshape(h, w) * 255).astype(np.uint8)


if run:
    path = get_video_path()
    if path is None:
        st.warning("Upload a video first.")
        st.stop()
    try:
        M, fps, n_total = read_video_matrix(path, width, height, n_frames, start_frame=start if sample else 0, step=step)
        GT = None
        if sample and sample["gt"]:
            GT, _, _ = read_video_matrix(ASSETS / sample["gt"], width, height, n_frames, start_frame=start, step=step, binarise=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read video: {e}")
        st.stop()

    bar = st.progress(0.0, text="Running RPCA (IALM)…")
    L_svd, S_svd, t_svd = svd_baseline(M, svd_k)
    lam = lam_scale / np.sqrt(max(M.shape))
    L, S, diag = rpca_ialm(M, lam=lam, max_iter=max_iter, tol=tol,
                           progress=lambda f, r, k: bar.progress(f, text=f"IALM — residual {r:.1e}, rank {k}"))
    bar.empty()
    st.session_state["res"] = dict(M=M, GT=GT, L_svd=L_svd, S_svd=S_svd, t_svd=t_svd, L=L, S=S, diag=diag,
                                   h=height, w=width, fps=fps, k=svd_k)

res = st.session_state.get("res")
if not res:
    st.info("Pick a video in the sidebar and press **Run decomposition**. "
            "The sample clips take roughly 5–20 seconds on CPU.")
    st.stop()

M, GT, L, S, diag = res["M"], res["GT"], res["L"], res["S"], res["diag"]
h, w = res["h"], res["w"]
n = M.shape[1]

# ------------------------------------------------------------------ headline numbers
c = st.columns(5)
c[0].metric("Matrix M", f"{M.shape[0]} × {M.shape[1]}")
c[1].metric("RPCA iterations", diag.n_iterations, "converged" if diag.converged else "max-iter reached", delta_color="off")
c[2].metric("Background rank (RPCA)", diag.rank_L[-1])
c[3].metric("Foreground sparsity", f"{diag.sparsity_S[-1]:.1%} zeros")
c[4].metric("Runtime RPCA / SVD", f"{diag.runtime_sec:.1f}s / {res['t_svd']:.2f}s")

tab_view, tab_diag, tab_metrics = st.tabs(["Decomposition viewer", "Convergence & spectrum", "Metrics vs. SVD"])

# ------------------------------------------------------------------ viewer
with tab_view:
    method = st.radio("Method", ["RPCA (IALM)", f"Truncated SVD (k={res['k']})"], horizontal=True)
    Lm, Sm = (L, S) if method.startswith("RPCA") else (res["L_svd"], res["S_svd"])
    idx = st.slider("Frame", 0, n - 1, n // 2)
    cols = st.columns(4 if GT is not None else 3)
    cols[0].image(to_u8(M[:, idx], h, w), caption="Original frame (M)", use_container_width=True, clamp=True)
    cols[1].image(to_u8(Lm[:, idx], h, w), caption="Background (L, low-rank)", use_container_width=True, clamp=True)
    cols[2].image(to_u8(np.abs(Sm[:, idx]) * 4, h, w), caption="Foreground (|S|, ×4 brightness)", use_container_width=True, clamp=True)
    if GT is not None:
        cols[3].image(to_u8(GT[:, idx], h, w), caption="Ground-truth mask", use_container_width=True, clamp=True)

    if st.button("Render animation (GIF)"):
        frames = []
        for i in range(0, n, max(1, n // 60)):
            tiles = [to_u8(M[:, i], h, w), to_u8(Lm[:, i], h, w), to_u8(np.abs(Sm[:, i]) * 4, h, w)]
            frames.append(Image.fromarray(np.hstack(tiles)).convert("P"))
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=int(1000 / min(res["fps"], 15)), loop=0)
        st.image(buf.getvalue(), caption="original | background | foreground")
        st.download_button("Download GIF", buf.getvalue(), "rpca_decomposition.gif", "image/gif")

# ------------------------------------------------------------------ diagnostics
with tab_diag:
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(diag.residuals); ax.set_xlabel("Iteration"); ax.set_ylabel("‖M−L−S‖F / ‖M‖F")
    ax.set_title("IALM convergence"); ax.grid(alpha=0.3)
    a.pyplot(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(diag.rank_L, color="tab:red"); ax.set_xlabel("Iteration"); ax.set_ylabel("rank(L)")
    ax.set_title("Rank of the background estimate"); ax.grid(alpha=0.3)
    b.pyplot(fig)
    sv = np.linalg.svd(M, compute_uv=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(sv, "o-", ms=3); ax.set_xlabel("Index"); ax.set_ylabel("Singular value of M")
    ax.set_title("Spectrum of M — a few dominant values indicate a low-rank background"); ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(f"λ used = {diag.lam:.5f}")

# ------------------------------------------------------------------ metrics
with tab_metrics:
    rows = {"Method": ["Truncated SVD", "RPCA (IALM)"],
            "Runtime (s)": [round(res["t_svd"], 3), round(diag.runtime_sec, 2)]}
    rows["Reconstruction ‖M−L−S‖F/‖M‖F"] = [
        f"{np.linalg.norm(M - res['L_svd'] - res['S_svd']) / np.linalg.norm(M):.2e}",
        f"{np.linalg.norm(M - L - S) / np.linalg.norm(M):.2e}"]
    if GT is not None:
        sc_svd = foreground_scores(res["S_svd"], GT)
        sc_rpca = foreground_scores(S, GT)
        rows["Precision"] = [round(sc_svd[0], 3), round(sc_rpca[0], 3)]
        rows["Recall"] = [round(sc_svd[1], 3), round(sc_rpca[1], 3)]
        rows["F1"] = [round(sc_svd[2], 3), round(sc_rpca[2], 3)]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if GT is None:
        st.caption("Ground-truth masks are only available for the synthetic sample (pixel-level F1 at |S| > 0.1).")
    else:
        st.caption("Foreground = |S| > 0.1, compared pixel-wise with the BMC ground-truth masks.")
