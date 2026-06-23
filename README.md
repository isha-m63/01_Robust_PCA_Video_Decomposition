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
- randomized SVD acceleration
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
- Center Location Error (CLE)


---

# results

The RPCA decomposition successfully:
- separated static backgrounds
- isolated moving foreground objects
- reduced ghosting artifacts compared to standard SVD

Observed real-world limitations can included:
- illumination changes
- dynamic backgrounds
- increased effective rank
- shadow artifacts
- camera motion sensitivity

---


