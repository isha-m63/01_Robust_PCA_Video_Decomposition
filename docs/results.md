# Results

Source: `v1_no_gt/` and `v2_with_gt/` notebooks and `video_decomposition_pipeline.ipynb`. `M` is 10,800 × 125 for
the synthetic sequence (BMC 2012 #111, 25–30 s at 25 fps) and 10,800 × 50 for the real clip (Video_008, 25–30 s at
10 fps). Both have estimated rank 1 at 90% and 99% explained variance.

## Reconstruction quality

Measured on `L` against frame 0 of `M` as a proxy for a clean background.

| Data | Method | Frobenius error | MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| Synthetic | Truncated SVD (k=3) | 0.0428 | 0.000099 | 42.26 | 0.9870 |
| Synthetic | PCA | 0.0414 | 0.000117 | 42.09 | 0.9860 |
| Synthetic | **RPCA (IALM)** | 0.0481 | **0.000027** | **45.65** | **0.9972** |
| Real | Truncated SVD (k=3) | 0.0770 | 0.004426 | 23.76 | 0.8362 |
| Real | PCA | 0.0736 | 0.004556 | 23.70 | 0.8319 |
| Real | RPCA (IALM) | 0.0803 | 0.004419 | 23.71 | 0.8395 |

SVD and PCA minimise Frobenius error by construction (Eckart–Young), so RPCA having a higher Frobenius error but lower
MSE/higher PSNR against the proxy background is consistent with theory: RPCA does not optimise Frobenius error, and the proxy
reference (frame 0) is itself only an approximation of the true background. On the real clip the three methods are indistinguishable.

SVD rank sweep on the synthetic data (pipeline notebook): Frobenius error 0.0470 / 0.0428 / 0.0400 for k = 1 / 3 / 5.
More rank gives lower error but also lets moving objects leak into `L` ("ghosting"), which is why error alone is not a
good selection criterion.

## Foreground segmentation (ground truth available for the synthetic sequence)

`|S|` thresholded with Otsu, compared with the provided mask:

| Method | Precision | Recall | F1 | Specificity | Accuracy |
|---|---|---|---|---|---|
| Truncated SVD | 0.626 | 0.057 | 0.104 | 1.000 | 0.992 |
| PCA | 0.628 | 0.060 | 0.108 | 0.9999 | – |
| RPCA (IALM) | 0.613 | 0.050 | 0.091 | 1.000 | 0.992 |

Precision is decent but recall is under 6% for every method, so the recovered foreground covers only a small part of
the true moving objects. Foreground pixels are a tiny fraction of the image, which is why accuracy and specificity are
near 1 regardless. F1 is the figure to look at, and by it RPCA does not beat the baselines on this sequence.

## IALM behaviour

| | Synthetic | Real |
|---|---|---|
| Iterations to tol 1e-7 | 46 | 70 |
| Final residual | 9.2e-8 | 9.0e-8 |
| Runtime | 5–7 s | 2–3 s |
| Rank of L | **1** | **25** (of 50 frames) |
| Zeros in S | **97.2%** | 17.4% |

On the synthetic scene RPCA finds exactly the structure the model assumes. On the real clip the residual is just as
small, but a rank-25 `L` and a mostly non-zero `S` mean the solver has fit the data (including noise and lighting
change) rather than separated a background. A small residual says the optimisation worked; it does not say the
decomposition is meaningful.

## Caveats

- Single sequence for each of synthetic and real; no variance estimates.
- The ground-truth scores printed for the real clip (F1 ≈ 0.01 for all methods) are invalid: they compare against the
  synthetic sequence's mask.
- Center Location Error and randomized-SVD acceleration appear in plans/descriptions but have no results here.
