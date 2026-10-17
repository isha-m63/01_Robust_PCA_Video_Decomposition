# Usage

## Install

```bash
pip install -r requirements-dev.txt     # demo + notebooks + pytest
```

The demo alone needs only `requirements.txt` (Streamlit, NumPy, OpenCV headless, Matplotlib, Pillow). Run the tests with `pytest`.

## Data

`data/raw_data/` holds `111.mp4` and `111_gt.mp4` (BMC 2012 synthetic sequence 111 and its foreground mask) and
`Video_008.avi` (real surveillance clip). The pipeline notebook downloads the BMC files if they are missing.
`demo_assets/` contains short clips for the demo.

## Run the demo

```bash
streamlit run streamlit_app.py            # http://localhost:8501
```

Pick a sample or upload a video (mp4 / avi / mov / mkv, short clips work best), then press **Run decomposition**.

| Setting | Meaning |
|---|---|
| Frame width | 60 – 160 px; height is 3/4 of width |
| Frames / Frame step | How many frames to stack into `M`, and sub-sampling |
| SVD baseline rank k | Rank of the truncated-SVD comparison |
| RPCA λ multiplier | `λ = multiplier / √max(m, n)`; larger gives a sparser foreground |
| RPCA max iterations / tolerance | IALM stopping rule; loosen the tolerance for noisy real footage |

Tabs: **Decomposition viewer** (original, background, foreground, ground truth, GIF export), **Convergence &
spectrum**, **Metrics vs. SVD**. Caps: 160×120 pixels, 200 frames.

## Use the functions directly

```python
from rpca_demo import read_video_matrix, svd_baseline, rpca_ialm, foreground_scores

M, fps, n_total = read_video_matrix("demo_assets/synthetic_scene.mp4", width=120, height=90, max_frames=125)
L, S, diag = rpca_ialm(M, lam=1/ (M.shape[0] ** 0.5), max_iter=100, tol=1e-7)
print(diag.converged, diag.n_iterations, diag.rank_L[-1])
```

## Notebooks

Start with `v2_with_gt/01…03` for a clean per-method comparison, or
`video_decomposition_pipeline.ipynb` for the full story. Paths in the notebooks point to the original machine
(`/home/isha/Desktop/AI_ML_2026/...`); update them to your checkout.

## Docker and Streamlit Community Cloud

```bash
docker build -t rpca-demo .
docker run -p 7860:7860 rpca-demo         # http://localhost:7860
```

For Streamlit Community Cloud create an app from the repository: branch `main`, file `streamlit_app.py`, Python 3.11.

The image is `python:3.11-slim` with `libglib2.0-0` (OpenCV), runs as uid 1000, and allows uploads up to 50 MB.
