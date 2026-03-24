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

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=06B6D4&center=true&vCenter=true&width=750&lines=Hybrid+CNN+%2B+MediaPipe+Landmark+Architecture;7+Emotion+Classes+%E2%80%94+Real-Time+at+30+FPS;Bidirectional+Cross-Attention+%7C+EfficientNet-B2+%2B+478+Landmarks;87.9%25+Validation+Accuracy+%7C+Disgust+92%25+Recall" alt="Typing SVG" /></a>

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

**HybridEmotionNet ** — a dual-branch neural network for real-time facial emotion recognition that fuses **EfficientNet-B2 appearance features** with **MediaPipe 3D landmark geometry** via bidirectional cross-attention.

Processes webcam frames at **30+ FPS**, extracts **478 3D landmarks**, crops the face at 224×224, and classifies into 7 emotions with EMA + sliding window temporal smoothing.

** highlights:** 87.9% validation accuracy · Disgust recall 51%→90% · Fear recall 65%→75% · 75k balanced training images · ViT-scored quality filtering

---

## Architecture

![Architecture](https://huggingface.co/Huuffy/VisageCNN/resolve/main/Architecture%20digram.png)

```
Face crop (224×224) ──► EfficientNet-B2 ──► [B, 256] appearance
                         blocks 0-1 frozen
                         blocks 2-8 fine-tuned

478 landmarks (xyz)  ──► MLP encoder    ──► [B, 256] geometry
                         1434 → 512 → 384 → 256

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
| CNN branch | EfficientNet-B2, ImageNet init, blocks 0–1 frozen, gradient checkpointing |
| Coord branch | MLP 1434 → 512 → 384 → 256, BN + Dropout |
| Fusion | Bidirectional cross-attention + MLP |
| Parameters | ~8M total |
| Model size | ~90 MB |

---

## Performance 

| Metric | Value |
|--------|-------|
| Validation accuracy | **87.9%** |
| Macro F1 | **0.88** |
| Inference speed | ~12 ms/frame on RTX 3050 |

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Angry | 0.85 | 0.83 | 0.84 |
| Disgust | 0.97 | 0.90 | 0.94 |
| Fear | 0.89 | 0.75 | 0.82 |
| Happy | 0.97 | 0.99 | 0.98 |
| Neutral | 0.85 | 0.91 | 0.88 |
| Sad | 0.78 | 0.88 | 0.83 |
| Surprised | 0.83 | 0.90 | 0.86 |

---

## Files in This Repo

| File | Size | Required |
|------|------|---------|
| `models/weights/hybrid_best_model.pth` | ~90 MB | Yes — best macro F1 checkpoint |
| `models/weights/hybrid_swa_final.pth` | ~90 MB | Optional — SWA ensemble model |
| `models/scalers/hybrid_coordinate_scaler.pkl` | 18 KB | Yes — landmark scaler |

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
    ("models/weights/hybrid_swa_final.pth",         "models/weights/hybrid_swa_final.pth"),
    ("models/scalers/hybrid_coordinate_scaler.pkl", "models/scalers/hybrid_coordinate_scaler.pkl"),
]:
    src = hf_hub_download(repo_id="Huuffy/VisageCNN", filename=remote)
    pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, local)
```

Or with the HF CLI:
```bash
huggingface-cli download Huuffy/VisageCNN models/weights/hybrid_best_model.pth --local-dir .
huggingface-cli download Huuffy/VisageCNN models/weights/hybrid_swa_final.pth --local-dir .
huggingface-cli download Huuffy/VisageCNN models/scalers/hybrid_coordinate_scaler.pkl --local-dir .
```

### 3 — Run inference

```bash
# Standard
python inference/run_hybrid.py

# With SWA ensemble
python inference/run_hybrid.py --ensemble
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

75,376 total images — 10,768 per class × 7 emotions, perfectly balanced.

**Sources:** AffectNet · RAF-DB · FER2013 · AffectNet-Short · ScullyowesHenry · RAF-DB Kaggle

All images passed a two-stage quality filter:
1. MediaPipe FaceMesh (dual confidence: 0.5 normal + 0.2 lenient for extreme expressions)
2. ViT confidence scoring (`dima806/facial_emotions_image_detection`) with per-class asymmetric mislabel thresholds

Final class balance achieved via ViT-scored capping — lowest-confidence images removed first, preserving the highest quality examples per class.

---

## Training Config

| Setting | Value |
|---------|-------|
| Loss | Focal Loss γ=2.0 + label smoothing 0.12 |
| Optimizer | AdamW, weight decay 0.05 |
| LR | OneCycleLR — CNN 5e-5, fusion 5e-4 |
| Batch | 96 + grad accumulation ×2 (eff. 192) |
| Augmentation | CutMix + noise + rotation + zoom |
| Mixed precision | torch.amp (AMP) |
| Best model saved by | Macro F1 (not val accuracy) |
| SWA | Epochs 30–70, BN update after training |
| Early stopping | patience=15 on macro F1 |

---

## Retrain From Scratch

```bash
# Delete old cache and train
rmdir /s /q models\cache
python scripts/train_hybrid.py
```

Full guide: [GitHub README](https://github.com/Huuffy/VisageCNN)

---

<div align="center">

**Built with curiosity and a lot of training runs**

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer)

</div>
