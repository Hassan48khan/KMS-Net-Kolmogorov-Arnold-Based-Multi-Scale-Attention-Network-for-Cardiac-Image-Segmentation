# KMS-Net: Kolmogorov–Arnold-Based Multi-Scale Attention Network for Cardiac Image Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/IEEE%20Access-Published%202026-blue" alt="IEEE Access"/>
</p>

> **KMS-Net: Kolmogorov–Arnold-Based Multi-Scale Attention Network for Cardiac Segmentation**  
> Hassan Ali, Abid Mehmood, David Noule Tolno, Sery Gahouidi Junior Thierry S, Muhammad Saeed, Naeem Ahmed  
> *IEEE Access*, Volume 14, 2026 · DOI: [10.1109/ACCESS.2026.3674209](https://doi.org/10.1109/ACCESS.2026.3674209)

---

## Overview

KMS-Net is a hybrid deep learning architecture for accurate cardiac image segmentation in both 2D echocardiography and cardiac MRI. It addresses the fundamental tension between local feature precision (where CNNs excel) and long-range dependency modeling (where Transformers excel) by unifying five complementary mechanisms into a single, efficient encoder–decoder framework.

**Key idea:** Replace fixed activation functions with learnable, spline-based Kolmogorov–Arnold Network (KAN) operators throughout the network, and augment them with multi-scale attention and linear-complexity state-space modeling for robust cardiac boundary delineation.

---

## Architecture

```
Input (1×H×W)
    │
    ▼
┌─────────────────────────────────────────┐
│  Stem Block                             │
│  ResConv → SS2D → EMA → Downsample×2   │
└────────────────────┬────────────────────┘
                     │
        ┌────────────▼────────────┐
        │     Encoder (4 stages)  │
        │  ResConv+EMA → SS2D     │
        │  PatchEmbed → KAN Block │
        │  (×2 downsample/stage)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Bottleneck: KASPPS    │
        │  KAN Conv (rates 6,12,18│
        │  grid 3,6,9) + SE + Pool│
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │     Decoder (4 stages)  │
        │  Upsample×2 → MSAG      │
        │  (skip fusion) → SS2D   │
        │  → EMA → ResConv        │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Prediction Head        │
        │  Conv1×1 → EMA → Output │
        └─────────────────────────┘
```

### Core Components

| Component | Role |
|---|---|
| **ResConv + EMA** | Residual conv blocks with Efficient Multi-Scale Attention for local boundary discrimination |
| **SS2D** | Selective State-Space 2D for linear-complexity long-range spatial modeling |
| **KASPPS** | KAN-based Atrous Spatial Pyramid Pooling with Squeeze-and-Excitation for multi-scale bottleneck context |
| **MSAG** | Multi-Scale Attention Gates on skip connections for refined boundary delineation |
| **KAN Block** | Tokenized KAN layers (spline-based, grid size=7) for expressive nonlinear feature modeling |

---

## Results

### ACDC Cardiac MRI Dataset

| Model | Mean Dice (%) | HD95 (mm) | Mean IoU (%) |
|---|:---:|:---:|:---:|
| UNet | 87.55 | — | — |
| TransUNet | 89.71 | 2.54 | — |
| Swin-UNet | 90.00 | 4.52 | — |
| nnUNet | 91.61 | — | — |
| U-KAN | 91.54 | — | — |
| LKCA-Net | 92.52 | 1.09 | 85.71 |
| **KMS-Net (Ours)** | **92.65** | **1.08** | **85.95** |

### CAMUS 2D Echocardiography Dataset (Mask 1 — LV Endocardium)

| Model | 2CH-ED Dice | 4CH-ED Dice | HD (mm) |
|---|:---:|:---:|:---:|
| UNet | 0.9156 | 0.9101 | 8.63 |
| Swin-UNet | 0.9363 | 0.9483 | 7.51 |
| U-KAN | 0.9299 | 0.9394 | 4.55 |
| **KMS-Net (Ours)** | **0.9417** | **0.9518** | **3.45** |

### EchoNet-Dynamic (Zero-Shot Transfer from CAMUS)

| Setting | Mean Dice | LVEDV MAE (ml) | LVESV MAE (ml) | LVEF MAE (%) |
|---|:---:|:---:|:---:|:---:|
| Trained on EchoNet (3000 frames) | 0.89 | 12.72 | 8.61 | 4.87 |
| **Zero-shot from CAMUS** | 0.79 | 12.59 | 8.60 | 9.02 |

---
## Citation

If you use KMS-Net in your research, please cite:

```bibtex
@article{ali2026kmsnet,
  author  = {Hassan Ali and Abid Mehmood and David Noule Tolno and
             Sery Gahouidi Junior Thierry S and Muhammad Saeed and Naeem Ahmed},
  title   = {{KMS-Net}: {Kolmogorov}--{Arnold}-Based Multi-Scale Attention
             Network for Cardiac Segmentation},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {41230--41247},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3674209}
}
```

---

## Credits

- KAN convolution sliding-window implementation adapted from [detkov/Convolution-From-Scratch](https://github.com/detkov/Convolution-From-Scratch)  
- KANLinear spline implementation inspired by [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756) (Liu et al., 2024)  
- EMA module based on [Efficient Multi-Scale Attention](https://arxiv.org/abs/2305.13563) (Ouyang et al., ICASSP 2023)  
- SS2D inspired by [VMamba](https://arxiv.org/abs/2401.10166) (Jiao et al., NeurIPS 2024)  
- U-KAN baseline from [U-KAN](https://arxiv.org/abs/2406.02918) (Li et al., AAAI 2025)

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

- **Hassan Ali** — ali.hassan@nuaa.edu.cn (Nanjing University of Aeronautics and Astronautics)  
- **Abid Mehmood** — abid.mehmood@dsu.edu (Dakota State University)
