# Known issues and remaining limitations

## Fixed in the demo code (`rpca_demo.py`, `streamlit_app.py`)
- **Randomized SVD** is implemented (`randomized_svd`, `svt(..., randomized=True)`, `rpca_ialm(..., svd="randomized")`)
  and selectable in the demo. On the synthetic clip it gives the same result as the exact SVD about 20% faster.
- **Center Location Error** is implemented (`center_location_error`) and shown next to precision/recall/F1.
- **Threshold exploration:** the demo's metrics tab has a slider for the `|S|` threshold, so the low-recall behaviour of
  a fixed Otsu threshold can be explored interactively.
- **Different initial μ:** the notebooks' solver and `rpca_demo.py` both use `μ₀ = 1.25 / ‖M‖₂`. The other rule
  (`mn / (4‖M‖₁)`) appears only in a raw "#EXTRA" diagnostic cell of the pipeline notebook. It does not work with
  this solver on its own (L becomes dense after two iterations), so it is not offered.
- A pytest suite (`tests/test_rpca.py`) covers the operators, both SVD engines, the metrics and video reading.
- Requirements are version ranges in `requirements.txt` / `requirements-dev.txt`; duplicate/Hugging-Face-only files
  were removed.
- **Notebooks run.** All ten notebooks (`video_decomposition_pipeline`, the three root `01-03` notebooks, `v1_no_gt/`,
  `v2_with_gt/`) execute end to end from a clean checkout of the data. Fixes: absolute `/home/isha/...` paths became a
  project-root lookup (`rpca_demo.py` marks the root), the `path/...` placeholders in the root notebooks point at
  `data/raw_data/` and `outputs/`, and a duplicated `@dataclass(frozen=True)` decorator that made the Config cell fail on
  Python 3.10 was removed. `test/eval.py` uses a path relative to itself.
- The notebook text that claimed randomized SVD and CLE results now points to `rpca_demo.py`, where both are implemented
  (the notebooks themselves do not compute CLE or use randomized SVD).

## Still open
- The demo's foreground metrics are not comparable with the notebooks' table: the demo uses 320×240 clips, a fixed
  threshold on `|S|`, and its own ground-truth binarisation, while the notebooks use Otsu thresholds on the full-size
  masks.
- PSNR/SSIM in the notebooks use frame 0 of `M` as a "clean background" reference, which may itself contain foreground.
- Ground-truth metrics for the real clip reuse the synthetic sequence's mask (the notebook says so) and are not
  meaningful.
- "Reduced ghosting compared with SVD" is supported visually (`outputs/v2/SVD_MLS_sim_data_ghosting.png`) but not by
  the notebooks' quantitative foreground metrics.
- `outputs/`, `data/`, `test/` and `v1_no_gt`/`v2_with_gt` are untracked in git (large files / your choice).
