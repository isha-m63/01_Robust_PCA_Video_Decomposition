import numpy as np
import pytest

from rpca_demo import (center_location_error, foreground_scores, randomized_svd, read_video_matrix, rpca_ialm,
                       soft_threshold, svd_baseline, svt)


@pytest.fixture(scope="module")
def synthetic():
    """Rank-1 'background' + a small bright moving square."""
    rng = np.random.default_rng(0)
    h, w, n = 20, 30, 40
    bg = rng.uniform(0.2, 0.6, size=(h * w, 1))
    M = np.tile(bg, (1, n))
    GT = np.zeros((h * w, n))
    for j in range(n):
        frame = np.zeros((h, w)); frame[5:9, j % 20:j % 20 + 4] = 1
        GT[:, j] = frame.ravel()
    return np.clip(M + 0.4 * GT, 0, 1).astype(np.float32), GT, (h, w)


def test_soft_threshold():
    assert soft_threshold(np.array([-2.0, -0.5, 0.5, 2.0]), 1.0).tolist() == [-1.0, 0.0, 0.0, 1.0]


def test_svt_drops_small_singular_values():
    A = np.diag([5.0, 1.0, 0.1])
    L, rank = svt(A, 0.5)
    assert rank == 2 and np.allclose(np.diag(L), [4.5, 0.5, 0.0])


def test_randomized_svd_matches_exact_on_low_rank():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 60))
    _, s, _ = randomized_svd(A, 5)
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False)[:5], rtol=1e-6)


@pytest.mark.parametrize("engine", ["full", "randomized"])
def test_rpca_recovers_low_rank_background_and_foreground(synthetic, engine):
    M, GT, (h, w) = synthetic
    L, S, d = rpca_ialm(M, max_iter=200, svd=engine)
    assert d.converged and d.rank_L[-1] <= 2
    p, r, f1 = foreground_scores(S, GT, 0.1)
    assert f1 > 0.8
    cle, used = center_location_error(S, GT, (h, w), 0.1)
    assert used > 0 and cle < 2.0


def test_rpca_beats_or_matches_svd_reconstruction(synthetic):
    M, GT, _ = synthetic
    L_svd, S_svd, _ = svd_baseline(M, 1)
    L, S, _ = rpca_ialm(M, max_iter=200)
    assert np.abs(L - L_svd).max() < 0.5      # both close to the same background
    assert foreground_scores(S, GT)[2] >= foreground_scores(S_svd, GT)[2] - 0.05


def test_cle_skips_empty_masks():
    cle, used = center_location_error(np.zeros((12, 3)), np.zeros((12, 3)), (3, 4))
    assert np.isnan(cle) and used == 0


def test_read_video_matrix_sample_clip():
    M, fps, n = read_video_matrix("demo_assets/synthetic_scene.mp4", 60, 45, 10)
    assert M.shape == (60 * 45, 10) and M.dtype == np.float32 and 0 <= M.min() and M.max() <= 1
