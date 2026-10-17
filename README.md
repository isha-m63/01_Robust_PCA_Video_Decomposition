# Robust PCA for Video Background–Foreground Decomposition

A computer vision and optimization project implementing **Robust Principal Component Analysis (RPCA)** from scratch for background subtraction in surveillance video.

This project decomposes a video sequence into:

- a **low-rank background matrix** \(L\)
- a **sparse foreground matrix** \(S\)

using the **Inexact Augmented Lagrangian Method (IALM)**.

The implementation includes:
- Singular Value Decomposition (SVD)
- Singular Value Thresholding (SVT)
- soft-thresholding proximal operators
- iterative convex optimization
- (planned, not yet implemented: randomized SVD acceleration)
- evaluation on both synthetic and real-world video sequences


---

# project overview

Given an observation matrix:

\[
M = L + S
\]

where:
- \(M\): observed video matrix
- \(L\): low-rank background
- \(S\): sparse moving foreground

the goal is to separate stationary scene structure from moving objects.

Each video frame is:
1. converted to grayscale
2. resized
3. flattened into a vector
4. stacked column-wise into the matrix \(M\)

RPCA exploits the assumption that:
- backgrounds are highly correlated across frames → low-rank
- moving objects occupy relatively few pixels → sparse


---

# mathematical formulation

Robust PCA solves:

\[
\min_{L,S}
\;
\|L\|_*
+
\lambda \|S\|_1
\]

subject to:

\[
M = L + S
\]

where:
- \(\|L\|_*\) is the nuclear norm (sum of singular values)
- \(\|S\|_1\) promotes sparsity

The optimization is solved using the **Inexact Augmented Lagrangian Method (IALM)**.


---

# implemented components

## 1. data acquisition and preprocessing

- BMC synthetic benchmark
- real-world `.avi` testing

Preprocessing pipeline:
- grayscale conversion
- frame resizing
- normalization to \([0,1]\)
- matrix construction


## 2. baseline truncated SVD

Implemented a low-rank baseline using truncated SVD before RPCA.

Purpose:
- establish comparison baseline
- demonstrate ghosting failure modes
- motivate robust decomposition


## 3. soft-thresholding operator

Implemented element-wise proximal shrinkage:

\[
\text{soft}(x,\lambda)
=
\text{sign}(x)\max(|x|-\lambda,0)
\]

Used to enforce sparsity in \(S\).


## 4. singular value thresholding (SVT)

Implemented spectral shrinkage on singular values:

\[
M = U\Sigma V^T
\]

\[
SVT_\tau(M)
=
U \cdot \text{soft}(\Sigma,\tau)\cdot V^T
\]

Used to enforce low-rank structure in \(L\).


## 5. IALM optimization loop

Implemented iterative updates for:
- sparse matrix \(S\)
- low-rank matrix \(L\)
- dual variable \(Y\)

with convergence checking using:

\[
\frac{
\|M-L-S\|_F
}{
\|M\|_F
}
< \epsilon
\]



---

# datasets

## BMC 2012 synthetic sequences

Used for:
- clean static background
- moving vehicle foreground separation
- baseline decomposition testing
- robustness analysis


## real-world `.avi` sequences

Used to evaluate:
- generalization
- failure modes
- limitations of low-rank assumptions


---

# evaluation metrics

Implemented and analyzed:

## reconstruction metrics
- PSNR
- SSIM
- RMSE
- Frobenius reconstruction error
- Relative Frobenius norm error

## foreground segmentation metrics
- Precision
- Recall
- F1 score

## geometric metric
- Center Location Error (CLE): defined in the pipeline notebook but not yet computed


---

# results

BMC 2012 sequence 111 (synthetic street scene, 25–30 s = 125 frames, 90×120 px, `M` is 10,800 × 125) and a real
surveillance clip, Video_008 (50 frames). Values are from `v2_with_gt/` notebooks.

| | Truncated SVD (k=3) | PCA | RPCA (IALM) |
|---|---|---|---|
| Synthetic: PSNR / SSIM of L | 42.26 dB / 0.9870 | 42.09 dB / 0.9860 | **45.65 dB / 0.9972** |
| Synthetic: foreground F1 (Otsu) | 0.104 | 0.108 | 0.091 |
| Real: PSNR / SSIM of L | 23.76 dB / 0.8362 | 23.70 dB / 0.8319 | 23.71 dB / 0.8395 |

RPCA on the synthetic sequence converges in 46 iterations (6.7 s), recovers a **rank-1** background, and leaves 97.2%
of `S` exactly zero. On the real clip it converges in 70 iterations but the background rank grows to **25 of 50** and
`S` is only 17% zeros, i.e. the low-rank/sparse assumption breaks down.

Reading these honestly:
- RPCA gives the best background reconstruction on the synthetic video. PSNR and SSIM are measured against frame 0 of
  `M`, used as a stand-in for a clean background, so they are a proxy.
- On the ground-truth foreground masks all three methods have **high precision (≈ 0.6) but very low recall (≈ 0.05)**,
  so F1 is ≈ 0.1 and RPCA is not better than SVD there. The moving vehicles are small, and at the Otsu threshold most
  foreground pixels are missed.
- The ground-truth numbers for the real clip (F1 ≈ 0.01) are not meaningful: the notebook applies the synthetic
  sequence's mask to a different video.
- Real-world limitations observed or expected: illumination changes, dynamic backgrounds, increased effective rank,
  shadows, and camera motion. The Streamlit demo uses a looser stopping tolerance for the real clip so that noise is not
  absorbed into a high-rank background.

Details in [docs/results.md](docs/results.md).

---

# documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline, IALM algorithm as implemented, module and notebook map |
| [docs/usage.md](docs/usage.md) | Install, data, running the notebooks and demo, parameters |
| [docs/results.md](docs/results.md) | Full metrics tables, convergence behaviour, interpretation |
| [docs/known-issues.md](docs/known-issues.md) | Claims not backed by code, hard-coded paths, other gaps |

# repository structure

```text
├── video_decomposition_pipeline.ipynb   Original end-to-end notebook (SVD baseline, IALM, metrics, real video)
├── v1_no_gt/                            Per-method notebooks without ground truth (01_svd, 02_pca, 03_rpca)
├── v2_with_gt/                          Same notebooks with ground-truth foreground metrics
├── test/                                Earlier notebooks and eval.py (standalone metrics script)
├── rpca_demo.py                         Core routines for the demo: video -> matrix, SVD baseline, IALM, metrics
├── streamlit_app.py                     Interactive demo
├── Dockerfile, requirements-demo.txt    Demo image
├── demo_assets/                         Short sample clips (synthetic, synthetic GT mask, real)
├── deploy/, scripts/deploy_space.py     Hugging Face Space deployment
├── data/raw_data/                       111.mp4, 111_gt.mp4 (BMC 2012), Video_008.avi
└── outputs/{v1,v2}/                     Cached observation matrices (.npy) and figures
```

---

# live demo (Streamlit, Docker)

`streamlit_app.py` provides an interactive demo (RPCA-IALM vs. truncated SVD, convergence plots,
F1 against ground truth, animated GIF export). Short sample clips are bundled in `demo_assets/`;
you can also upload your own video. The algorithms live in `rpca_demo.py`.

```bash
docker build -t rpca-demo .
docker run -p 7860:7860 rpca-demo      # http://localhost:7860
```

Deploy to a Hugging Face Docker Space:

```bash
export HF_TOKEN=hf_xxx
python scripts/deploy_space.py <username>/robust-pca-video
```

---

# references

- Candès, Li, Ma & Wright (2011), *Robust Principal Component Analysis?*, J. ACM 58(3).
- Lin, Chen & Ma (2010), *The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices*, arXiv:1009.5055.
- Vacavant et al. (2012), *A Benchmark Dataset for Outdoor Foreground/Background Extraction* (BMC 2012).
