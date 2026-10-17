# Known issues and unfinished work

Found while documenting the code. Nothing below has been changed.

## Claims without code or results
- **Randomized SVD** is mentioned (`In Step 7 we replace this with randomized SVD for speed`), but all SVDs use
  `np.linalg.svd`.
- **Center Location Error (CLE)** is described in a markdown cell of the pipeline notebook; no function or output
  computes it.
- **"Reduced ghosting compared with standard SVD"** (original README) is supported visually by the saved figures
  (`outputs/v2/SVD_MLS_sim_data_ghosting.png`), but not by the quantitative foreground metrics, where RPCA's F1 (0.091)
  is slightly below SVD's (0.104).

## Evaluation
- PSNR/SSIM use frame 0 of `M` as a "clean background" reference, which may itself contain foreground objects.
- Ground-truth metrics for the real clip reuse the synthetic sequence's mask (the notebook says so in a note) and are
  not meaningful.
- Recall is below 6% on the synthetic ground truth for all methods; the Otsu threshold on `|S|` may be too strict, but
  other thresholds were not explored.

## Reproducibility
- Notebook paths are absolute paths from the original machine
  (`/home/isha/Desktop/AI_ML_2026/Projects/Robust_PCA_Video_Decomposition/...`); `test/eval.py` has the same.
- The first git-tracked notebook run shows a warning that `111_gt.mp4` was not found at the time.
- `outputs/` (cached `.npy` matrices), `data/`, `test/` and `v1_no_gt`/`v2_with_gt` are untracked in git.
- `video_decomposition_pipeline.ipynb` has uncommitted modifications.

## Code
- `rpca_demo.py` and the pipeline notebook use different initial `μ`; results are close but not bit-identical.
- There is no automated test suite for this project.
