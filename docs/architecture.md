# Architecture

## Idea

Stack each grayscale frame as a column of `M` (pixels × frames). A static background is nearly identical in every
frame, so it forms a low-rank matrix `L`. Moving objects touch few pixels, so they form a sparse matrix `S`:

```
M = L + S        minimize ||L||_* + λ ||S||_1      (λ = 1/√max(m, n))
```

## Pipeline

```
video (.mp4/.avi) ──► grayscale ──► resize (default 120×90) ──► scale to [0,1] ──► flatten ──► M (10,800 × n_frames)
                                                                                      │
                          ┌───────────────────────────────────────────────────────────┤
                          ▼                            ▼                               ▼
                 truncated SVD (rank k)          PCA (sklearn)                  RPCA via IALM
                 L = best rank-k fit             low-dim reconstruction         L low-rank, S sparse
                          └────────────────────────────┴───────────────────────────────┘
                                                       ▼
                       reconstruction metrics (PSNR, SSIM, Frobenius, MSE)  +  foreground metrics vs. GT mask
```

## IALM as implemented (`rpca_demo.py: rpca_ialm`)

Initial `L = S = Y = 0`, `μ = 1.25 / ‖M‖₂`, ratio `ρ = 1.2`, `μ_max = 1e4`. Each iteration:

```
S ← soft_threshold( M − L + Y/μ,  λ/μ )          sparse foreground
L ← SVT( M − S + Y/μ,  1/μ )                     low-rank background (singular value thresholding)
Y ← Y + μ (M − L − S)                            dual update
μ ← min(ρ μ, μ_max)
stop when ‖M − L − S‖_F / ‖M‖_F < tol
```

Soft-thresholding is `sign(x) · max(|x| − λ, 0)`; SVT applies it to the singular values. Per-iteration residual, rank of
`L` and the fraction of zeros in `S` are recorded. All notebooks and the demo use Lin et al.'s initial `μ`; the
`μ = mn / (4‖M‖₁)` rule appears only in a raw diagnostic cell of the pipeline notebook. `svd="randomized"` replaces the
exact SVD in the SVT step by a randomized SVD of the leading singular triplets.

## Files

| File | Role |
|---|---|
| `rpca_demo.py` | `read_video_matrix`, `soft_threshold`, `svt`, `svd_baseline`, `rpca_ialm`, `randomized_svd`, `foreground_scores` (precision/recall/F1 of `|S| > threshold`), `center_location_error`, `psnr`, `Diagnostics` dataclass |
| `streamlit_app.py` | UI: sample or uploaded video, settings, side-by-side SVD vs RPCA, diagnostics, GIF export |
| `video_decomposition_pipeline.ipynb` | Original end-to-end notebook; downloads data, config, matrix cache, SVD baseline, IALM, metrics, real-video test |
| `v1_no_gt/`, `v2_with_gt/` | One notebook per method (SVD, PCA, RPCA) run on simulated and real data. `v2` adds ground-truth metrics (Otsu-thresholded `|S|` vs. mask: precision, recall, F1, specificity, accuracy) |
| `test/` | Earlier per-method notebooks and `eval.py`, a standalone evaluation script |

## Design decisions

- **Same matrix for every method**, so metrics are directly comparable.
- **Looser tolerance for noisy video.** Pushing IALM to a tight tolerance on a real clip makes it absorb sensor noise
  into a high-rank `L`; stopping early acts as regularisation. The demo exposes `tol` for this reason.
- **Cached observation matrices** in `outputs/` avoid re-reading video on every notebook run.
