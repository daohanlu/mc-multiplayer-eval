# Consistency Metric Comparison Tables

Metrics compare visual similarity of two players generated frames.
Look-Same (turnToLookEval): both players should see similar scenery.
Look-Opposite (turnToLookOppositeEval): players face opposite directions.
Thresholding accuracy: best 1-D threshold separability (fixed direction, chance = 50%).

### Table 1: Training Objective Variants

**Mean Similarity (Look-Same / Look-Opposite)**

| Model | 1-LPIPS | CLIP | DINOv2 | DINOv3 |
| :--- | :---: | :---: | :---: | :---: |
| Solaris | 0.671 / 0.636 | 0.957 / 0.934 | 0.888 / 0.839 | 0.962 / 0.947 |
| Solaris w/o KV-BP | 0.676 / 0.644 | 0.957 / 0.951 | 0.882 / 0.831 | 0.963 / 0.950 |
| Solaris w Pre-DMD | 0.574 / 0.564 | 0.898 / 0.894 | 0.827 / 0.842 | 0.931 / 0.946 |
| ODE Reg | 0.630 / 0.602 | 0.942 / 0.931 | 0.841 / 0.811 | 0.934 / 0.932 |

**Thresholding Accuracy (%)**

| Model | 1-LPIPS | CLIP | DINOv2 | DINOv3 | **Avg** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Solaris | 65.6 | 75.0 | 71.9 | 71.9 | **71.1** |
| Solaris w/o KV-BP | 67.2 | 68.8 | 68.8 | 70.3 | **68.8** |
| Solaris w Pre-DMD | 59.4 | 56.2 | 53.1 | 51.6 | **55.1** |
| ODE Reg | 57.8 | 59.4 | 62.5 | 56.2 | **59.0** |


### Table 2: Architecture Variants

**Mean Similarity (Look-Same / Look-Opposite)**

| Model | 1-LPIPS | CLIP | DINOv2 | DINOv3 |
| :--- | :---: | :---: | :---: | :---: |
| Solaris | 0.671 / 0.636 | 0.957 / 0.934 | 0.888 / 0.839 | 0.962 / 0.947 |
| Solaris w/o pretrain | 0.641 / 0.637 | 0.941 / 0.940 | 0.879 / 0.873 | 0.955 / 0.955 |
| Frame concat | 0.608 / 0.627 | 0.941 / 0.945 | 0.831 / 0.821 | 0.919 / 0.916 |

**Thresholding Accuracy (%)**

| Model | 1-LPIPS | CLIP | DINOv2 | DINOv3 | **Avg** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Solaris | 65.6 | 75.0 | 71.9 | 71.9 | **71.1** |
| Solaris w/o pretrain | 62.5 | 54.7 | 62.5 | 57.8 | **59.4** |
| Frame concat | 57.8 | 51.6 | 57.8 | 57.8 | **56.2** |
