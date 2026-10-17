# Robust PCA for Video Background–Foreground Decomposition

A comparative implementation of **Truncated SVD**, **Principal Component Analysis (PCA)**, and **Robust PCA (RPCA)** for video background modelling and foreground extraction.

This project began as an effort to better understand the mathematics behind **Singular Value Decomposition (SVD)**, **Principal Component Analysis (PCA)**, and **convex optimization**. Rather than treating these algorithms as black boxes, the goal was to study how they formulate low-rank approximation, why classical methods struggle with outliers, and how Robust PCA addresses these limitations through optimization.

The project evaluates all three methods on the **Background Models Challenge (BMC)** dataset using a common preprocessing pipeline and compares both reconstruction quality and foreground detection performance.

---

## Motivation

SVD and PCA are fundamental tools in machine learning, computer vision, and scientific computing. Both recover the best low-rank approximation of a matrix in the least-squares sense, but they do not explicitly distinguish moving foreground objects from the background.

Robust PCA reformulates the problem by assuming that an observation matrix can be decomposed into

* a **low-rank** background, and
* a **sparse** foreground,

which is solved through **Principal Component Pursuit (PCP)** using the **Inexact Augmented Lagrangian Method (IALM)**.

This repository explores these ideas from both a mathematical and implementation perspective.

---

## Repository Structure

```text
.
├── 01_svd_decomposition.ipynb     
├── 02_pca_decomposition.ipynb      
├── 03_rpca_decomposition.ipynb     
├── 01_RPCA_Report.pdf             
└── README.md
```

---

## Methods

### Truncated SVD

* Computes the best rank-(k) approximation of the observation matrix
* Minimizes Frobenius reconstruction error
* Optimal under the Eckart–Young theorem

### Principal Component Analysis (PCA)

* Mean-centres the observation matrix
* Performs low-rank approximation through SVD
* Models variation around the temporal mean frame

### Robust PCA (RPCA)

Uses the Principal Component Pursuit formulation

[
\min_{L,S} |L|_* + \lambda|S|_1
\quad\text{s.t.}\quad
M=L+S
]

where

* (L) is the low-rank background
* (S) is the sparse foreground

The optimization is solved using the **Inexact Augmented Lagrangian Method (IALM)** with alternating

* Singular Value Thresholding (SVT)
* Soft Thresholding
* Dual variable updates

---

## Evaluation

Background quality was evaluated using

* Mean Squared Error (MSE)
* Peak Signal-to-Noise Ratio (PSNR)
* Structural Similarity Index (SSIM)
* Relative Frobenius Error

Foreground segmentation was evaluated using

* Precision
* Recall
* F1-score
* Specificity
* Accuracy

---

## Results

The experiments highlight the strengths and limitations of each approach.

**SVD and PCA**

* Produce optimal low-rank approximations
* Frequently suffer from ghosting artifacts
* Do not explicitly separate moving objects from background

**Robust PCA**

* Successfully separates low-rank static background from sparse moving foreground when its assumptions hold
* Reduced ghosting artifacts compared to SVD and PCA
* Recovers an almost rank-1 background on the synthetic BMC sequence
* Performance degrades on complex real-world scenes where the background itself is no longer well modelled as low rank

Observed real-world limitations can included:
* illumination changes
* dynamic backgrounds
* increased effective rank
- shadow artifacts
* camera motion sensitivity


These observations demonstrate why RPCA remains a powerful baseline while motivating more advanced methods such as dynamic and online RPCA.

---

## Report

The accompanying report includes

* optimization formulation
* implementation details
* convergence analysis
* quantitative evaluation
* qualitative visual results
* discussion of limitations 

---

## References

* Candès, E. J., Li, X., Ma, Y., & Wright, J. *Robust Principal Component Analysis?* Journal of the ACM, 2011.
* Lin, Z., Chen, M., & Ma, Y. *The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices*, 2010.
* Bouwmans, T., et al. *On the Applications of Robust PCA in Image and Video Processing*, Proceedings of the IEEE, 2018.

---


## Author

**Isha Mahuvakar**

This project was developed as part of a self-study into linear algebra, optimization, and computer vision, with an emphasis on understanding the mathematical foundations rather than simply applying existing libraries.


---



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
├── Dockerfile, requirements.txt         Demo image and runtime dependencies (requirements-dev.txt: notebooks + tests)
├── demo_assets/                         Short sample clips (synthetic, synthetic GT mask, real)
├── tests/                               pytest suite for rpca_demo.py
├── data/raw_data/                       111.mp4, 111_gt.mp4 (BMC 2012), Video_008.avi
└── outputs/{v1,v2}/                     Cached observation matrices (.npy) and figures
```

---

# live demo (Streamlit, Docker)

`streamlit_app.py` provides an interactive demo (RPCA-IALM vs. truncated SVD, convergence plots,
F1 and center location error against ground truth, optional randomized SVD, animated GIF export). Short sample clips are bundled in `demo_assets/`;
you can also upload your own video. The algorithms live in `rpca_demo.py`.

```bash
docker build -t rpca-demo .
docker run -p 7860:7860 rpca-demo      # http://localhost:7860
```

**Streamlit Community Cloud:** new app from this repo, branch `main`, file `streamlit_app.py`, Python 3.11.

Run the tests with `pip install -r requirements-dev.txt && pytest`.

---
