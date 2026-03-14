---
license: mit
tags:
  - facial-expression-recognition
  - emotion-recognition
  - computer-vision
  - pytorch
  - mediapipe
  - efficientnet
  - real-time
  - image-classification
pipeline_tag: image-classification
---

<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=VisageCNN&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Real-Time%20Facial%20Expression%20Recognition&descAlignY=60&descAlign=50)

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=06B6D4&center=true&vCenter=true&width=750&lines=Hybrid+CNN+%2B+MediaPipe+Landmark+Architecture;7+Emotion+Classes+%E2%80%94+Real-Time+at+30+FPS;Bidirectional+Cross-Attention+%7C+EfficientNet-B0+%2B+478+Landmarks;Optimized+for+RTX+3050+%E2%80%94+4GB+VRAM" alt="Typing SVG" /></a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00BCD4?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
[![GitHub](https://img.shields.io/badge/GitHub-VisageCNN-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Huuffy/VisageCNN)

</div>

---

## What Is This?

**HybridEmotionNet** — a dual-branch neural network for real-time facial emotion recognition that fuses **EfficientNet-B0 appearance features** with **MediaPipe 3D landmark geometry** via bidirectional cross-attention.

Processes webcam frames at **30+ FPS**, extracts **478 3D landmarks**, crops the face, and classifies into 7 emotions with temporal smoothing.

---

## Architecture

![Architecture](https://huggingface.co/Huuffy/VisageCNN/resolve/main/Architecture%20digram.png)

```
Face crop (224×224) ──► EfficientNet-B0 ──► [B, 256] appearance
478 landmarks (xyz)  ──► MLP encoder    ──► [B, 256] geometry
                               │
               Bidirectional Cross-Attention (4 heads each)
               ┌──────────────────────────────────────────┐
               │  coord → CNN  (geometry queries appear.) │
               │  CNN  → coord (appear. queries geometry) │
               └──────────────────────────────────────────┘
                               │
               Fusion MLP: 512 → 384 → 256 → 128
                               │
               Classifier:   128 → 7 emotions
```

| Component | Detail |
|-----------|--------|
| CNN branch | EfficientNet-B0, ImageNet init, blocks 0–2 frozen |
| Coord branch | MLP 1434 → 512 → 384 → 256, BN + Dropout |
| Fusion | Bidirectional cross-attention + MLP |
| Parameters | 6.2M total / 5.75M trainable |
| Model size | 72 MB |

---

## Files in This Repo

| File | Size | Required |
|------|------|---------|
| `models/weights/hybrid_best_model.pth` | 72 MB | Yes — model weights |
| `models/scalers/hybrid_coordinate_scaler.pkl` | 18 KB | Yes — landmark scaler |
| `Architecture digram.png` | — | No — docs only |

---

## Quick Start

### 1 — Clone the code

```bash
git clone https://github.com/Huuffy/VisageCNN.git
cd VisageCNN
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 2 — Download weights

```python
from huggingface_hub import hf_hub_download
import shutil, pathlib

for remote, local in [
    ("models/weights/hybrid_best_model.pth",        "models/weights/hybrid_best_model.pth"),
    ("models/scalers/hybrid_coordinate_scaler.pkl", "models/scalers/hybrid_coordinate_scaler.pkl"),
]:
    src = hf_hub_download(repo_id="Huuffy/VisageCNN", filename=remote)
    pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, local)
```

Or with the HF CLI:
```bash
hf download Huuffy/VisageCNN models/weights/hybrid_best_model.pth --local-dir .
hf download Huuffy/VisageCNN models/scalers/hybrid_coordinate_scaler.pkl --local-dir .
```

### 3 — Run inference

```bash
python inference/run_hybrid.py
```

Press **Q** to quit.

---

## Emotion Classes

| Label | Emotion | Key Signals |
|-------|---------|-------------|
| 0 | Angry | Furrowed brows, tightened jaw |
| 1 | Disgust | Raised upper lip, wrinkled nose |
| 2 | Fear | Wide eyes, raised brows, open mouth |
| 3 | Happy | Raised cheeks, open smile |
| 4 | Neutral | Relaxed, no strong deformation |
| 5 | Sad | Lowered brow corners, downturned lips |
| 6 | Surprised | Raised brows, wide eyes, dropped jaw |

---

## Training Dataset

~30k clean images — FER2013 noise removed across all classes:

| Class | Images | Sources |
|-------|--------|---------|
| Angry | 6,130 | RAF-DB + AffectNet + AffectNet-Short + CK+ |
| Surprised | 5,212 | RAF-DB + AffectNet |
| Sad | 4,941 | RAF-DB + AffectNet + AffectNet-Short + CK+ |
| Disgust | 3,782 | AffectNet-Short + RAF-DB + CK+ |
| Neutral | 3,475 | RAF-DB + AffectNet |
| Fear | 3,418 | AffectNet-Short + RAF-DB + CK+ |
| Happy | 3,124 | RAF-DB + AffectNet |

Max class imbalance: **1.97×**

---

## Training Config

| Setting | Value |
|---------|-------|
| Loss | Focal Loss γ=2.0 + label smoothing 0.12 |
| Optimizer | AdamW, weight decay 0.05 |
| LR | OneCycleLR — CNN 5e-5, fusion 5e-4 |
| Batch | 128 + grad accumulation ×2 (eff. 256) |
| Augmentation | CutMix + noise + rotation + zoom |
| Mixed precision | torch.amp (AMP) |
| Early stopping | patience=40 on val accuracy |

---

## Retrain From Scratch

```bash
# Build dataset (downloads ~30k clean images from HuggingFace)
pip install datasets
python scripts/prepare_dataset.py

# Delete old cache and train
rmdir /s /q models\cache
python scripts/train_hybrid.py
```

Full training guide: [GitHub README](https://github.com/Huuffy/VisageCNN)

---

<div align="center">

**Built with curiosity and a lot of training runs**

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer)

</div>
