---
title: Robust PCA Video Decomposition
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Separate video background and moving foreground with RPCA
---

# Robust PCA — Video Background / Foreground Separation

Decompose a video matrix **M = L + S** into a low-rank background and a sparse moving foreground using
Robust PCA (Inexact Augmented Lagrangian Method, implemented from scratch in NumPy) and compare it with a
truncated-SVD baseline. Try the bundled BMC 2012 clips or upload your own short video.

Source code and notebooks: https://github.com/isha-m63/01_Robust_PCA_Video_Decomposition
