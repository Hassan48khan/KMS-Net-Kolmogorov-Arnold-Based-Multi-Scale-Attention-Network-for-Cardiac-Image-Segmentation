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

## Repository Structure

```
KMS-Net/
├── KMS-Net.py            # Main model: SuperKANet (full KMS-Net architecture)
├── Baseline UKAN.py      # Baseline U-KAN model for comparison
├── KASPPS.py             # KAN-based ASPP with Squeeze-and-Excitation module
├── KANLinear.py          # KANLinear primitive (spline-based learnable layer)
├── convolution.py        # KAN convolution sliding-window helper
└── README.md
```

> **Note:** `KANLinear.py` is a required dependency imported by `KASPPS.py` and `KMS-Net.py`. Make sure it is present in the same directory or on your Python path.

---

## Installation

### Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12 (tested on PyTorch 2.x with CUDA 11.8)  
- `timm` (for `DropPath`, `trunc_normal_`)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm numpy
```

### Clone the Repository

```bash
git clone https://github.com/Hassan48khan/KMS-Net-Kolmogorov-Arnold-Based-Multi-Scale-Attention-Network-for-Cardiac-Image-Segmentation.git
cd KMS-Net-Kolmogorov-Arnold-Based-Multi-Scale-Attention-Network-for-Cardiac-Image-Segmentation
```

---

## Quick Start

### Instantiate KMS-Net

```python
import torch
from KMS_Net import SuperKANet   # rename file to KMS_Net.py to avoid the space

model = SuperKANet(
    num_classes=3,          # LV, RV, Myo for ACDC; or 1 for binary (CAMUS)
    input_channels=1,       # grayscale MRI / echo
    img_size=128,
    embed_dims=[64, 128, 256],
    device='cuda'
).cuda()

x = torch.randn(2, 1, 128, 128).cuda()
out = model(x)              # shape: (2, num_classes, 128, 128)
print(out.shape)
```

### Training Loop Skeleton

```python
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

model = SuperKANet(num_classes=3, input_channels=1, img_size=128, device='cuda').cuda()
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=250)

dice_loss = ...   # your Dice loss implementation
ce_loss   = nn.CrossEntropyLoss()

for epoch in range(250):
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()

        preds = model(images)
        loss  = dice_loss(preds, masks) + ce_loss(preds, masks)

        # Optional: KAN regularization
        loss += 1e-5 * model.regularization_loss(0.5, 0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
```

### Training Settings (Paper)

| Dataset | Optimizer | LR | Epochs | Augmentation |
|---|---|---|---|---|
| CAMUS | SGD (momentum=0.9) | 0.01 | 250 | None (strict protocol) |
| ACDC | Adam | 1e-3 | 250 | ±15° rotation, H/V flip (p=0.5) |

**Input:** All images resized to **128×128**. Batch size **8**.  
**Loss:** Dice loss + Cross-Entropy loss (equal weighting).  
**Hardware:** NVIDIA RTX 3070 Ti (16 GB VRAM).

---

## Datasets

| Dataset | Modality | Subjects | Structures | Access |
|---|---|---|---|---|
| [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) | 2D Echo | 500 | LVendo, LVepi, LA | Public |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Cardiac MRI | 150 | LV, RV, Myo | Public |
| [EchoNet-Dynamic](https://echonet.github.io/dynamic/) | 2D Echo (video) | 10,030 | LV | Public |

**CAMUS protocol:** 10-fold cross-validation, patient-disjoint splits (450 train / 50 test per fold), images resized to 128×128, **no data augmentation**.

**ACDC protocol:** Standard 100-patient split (70 train / 10 val / 20 test), results averaged over 5 independent runs.

---

## Module Details

### KANLinear

The fundamental building block. Each layer learns a spline-based univariate function for every input–output feature pair:

```
output = base_activation(x) @ base_weight  +  B-spline(x) @ spline_weight
```

- **Grid size:** 7 (increased from the original U-KAN default of 5 for richer expressiveness)  
- **Spline order:** 3 (cubic B-splines)  
- Supports adaptive grid update via `update_grid()` and KAN regularization via `regularization_loss()`

### KASPPS

Drop-in replacement for standard ASPP at the bottleneck:

```
Input (C×H×W)
  ├── KAN Conv 3×3, dilation=6,  grid=3  → BN → ReLU → SE
  ├── KAN Conv 3×3, dilation=12, grid=6  → BN → ReLU → SE
  ├── KAN Conv 3×3, dilation=18, grid=9  → BN → ReLU → SE
  └── Global Avg Pool → Conv1×1 → BN → ReLU → SE → Upsample
          └──────────── Concat → Conv1×1 → BN → ReLU ────────►
```

Smaller grid sizes capture fine anatomical textures; larger grids capture broader structural context.

### MSAG (Multi-Scale Attention Gate)

Applied on every skip connection in the decoder:

```python
f_out = f + f ⊙ sigmoid(Conv1×1(concat(
    BN(Conv1×1(f)),
    BN(Conv3×3(f)),
    BN(DilatedConv3×3_d=2(f))
)))
```

Three parallel receptive fields (point-wise, regular, dilated) are fused into a soft attention mask, suppressing background while sharpening cardiac boundaries.

### SS2D (Selective State-Space 2D)

Provides near-linear-complexity global context (O(N) vs. O(N²) for self-attention):

1. **Multi-directional scanning** — unfolds the feature map into 4 sequences (left→right, top→bottom, right→left, bottom→top)  
2. **S6 state-space block** — recurrence: `h'(t) = Ah(t) + Bx(t)`, `y(t) = Ch(t)`  
3. **Cross-directional fusion** — merges the 4 enhanced sequences back to 2D

### EMA (Efficient Multi-Scale Attention)

Groups channels into G subgroups, applies horizontal + vertical pooling for anisotropic spatial attention, fuses with 3×3 local convolution via softmax cross-spatial similarity, and reweights each group output.

---

## Computational Cost

| Model | Params (M) | GFLOPs | Inference (ms/img) | FPS |
|---|:---:|:---:|:---:|:---:|
| UNet | 17.3 | 68.4 | 12.6 | 79.4 |
| TransUNet | 92.4 | 156.8 | 37.2 | 26.9 |
| Swin-UNet | 27.8 | 48.1 | 9.7 | 103.1 |
| U-Mamba | 25.6 | 41.3 | 9.0 | 111.1 |
| U-KAN (baseline) | 28.4 | 52.7 | 11.5 | 87.0 |
| **KMS-Net (Ours)** | **32.8** | **61.9** | **14.5** | **69.0** |

Inference measured over 300 forward passes on RTX 3070 Ti (128×128, batch size 1).  
KMS-Net runs at ~69 FPS, well within the 20–60 FPS needed for real-time clinical echocardiography review.

---

## Ablation Study (ACDC)

| Configuration | Mean Dice (%) | HD95 (mm) | Mean IoU (%) |
|---|:---:|:---:|:---:|
| Baseline (U-KAN) | 90.12 | 6.94 | 82.34 |
| + ResConv + EMA | 91.89 | 3.65 | 84.87 |
| + SS2D | 92.21 | 2.48 | 85.67 |
| + Standard ASPP | 92.37 | 1.94 | 85.78 |
| + KASPPS (ours) | 92.53 | 1.46 | 85.89 |
| **+ MSAG (full KMS-Net)** | **92.65** | **1.08** | **85.95** |

Every component contributes statistically significant improvement (p < 0.01, paired t-test).

---

## Known Issues & Fixes

### 1. File naming — spaces in filename
`Baseline UKAN.py` and `KMS-Net.py` contain characters that break Python imports. Rename them:

```bash
mv "Baseline UKAN.py" baseline_ukan.py
mv "KMS-Net.py"       kms_net.py
mv "KASPPS.py"        kaspps.py
```

Then update imports accordingly.

### 2. Missing `KANLinear.py` dependency
`KASPPS.py` imports `from KANLinear import KANLinear` and `import convolution`. These files must be present in the same directory. The `KANLinear` class is already fully implemented inside `KMS-Net.py`; extract it into a standalone `KANLinear.py` so that `KASPPS.py` can import it cleanly:

```python
# KANLinear.py — extract the KANLinear class from KMS-Net.py and save here
```

### 3. `convolution.py` device handling
The `multiple_convs_kan_conv2d` function currently creates the output tensor on a hardcoded `device` argument. When calling from `KASPPS.py`, pass `device=x.device` explicitly to avoid CPU/GPU mismatches:

```python
# In KASPPS.py KAN_Convolutional_Layer.forward():
return convolution.multiple_convs_kan_conv2d(
    x, self.convs, ..., device=x.device   # ← always pass x.device
)
```

### 4. `KASPPS.py` — standalone test block
The `if __name__ == "__main__"` block at the bottom of `KASPPS.py` requires `convolution.multiple_convs_kan_conv2d` to be implemented. If you run the file directly for smoke testing, ensure `convolution.py` is importable from the working directory.

### 5. `Baseline UKAN.py` — `Sigmoid` in final layer
The baseline model uses `nn.Sigmoid()` and stores it as `self.soft` but never calls it in `forward()`. The raw logits are returned by `self.final(out)`. This is intentional for use with BCEWithLogitsLoss, but add a docstring note to avoid confusion:

```python
# self.soft is defined but not applied in forward().
# Use BCEWithLogitsLoss during training, or apply torch.sigmoid() at inference.
```

### 6. `KMS-Net.py` — `SS2D` simplified implementation
The `SS2D` module in `KMS-Net.py` is a lightweight approximation (depthwise conv + SiLU + projection) rather than a full multi-directional selective scan. This is by design for efficiency but differs from the VMamba SS2D described in the paper. Add a comment clarifying this if sharing the code.

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
