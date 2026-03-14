<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=VisageCNN&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Real-Time%20Facial%20Expression%20Recognition&descAlignY=60&descAlign=50)

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=06B6D4&center=true&vCenter=true&width=750&lines=Hybrid+CNN+%2B+MediaPipe+Landmark+Architecture;7+Emotion+Classes+%E2%80%94+Real-Time+at+30+FPS;Cross-Attention+Fusion+%7C+EfficientNet-B0+%2B+478+Landmarks;Optimized+for+RTX+3050+%E2%80%94+4GB+VRAM" alt="Typing SVG" /></a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00BCD4?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
[![HuggingFace](https://img.shields.io/badge/Weights-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Huuffy/VisageCNN)

</div>

---

## What Is This Project?

VisageCNN is an ongoing research and engineering project aimed at building a **better-than-baseline facial expression recognition system** for real-time use. The core idea is: standard CNN-only approaches that operate purely on raw pixel data miss a lot of structural information about the human face. By combining **appearance features** (what the face looks like) with **geometric features** (how the face is shaped and positioned in 3D space), the model understands expressions the way humans intuitively do — through both visual texture and facial structure.

The system processes webcam frames live at **30 FPS**, extracts **478 3D facial landmarks** using MediaPipe, crops the face region, fuses both sources of information through a **bidirectional cross-attention mechanism**, and classifies the result into one of 7 emotion categories with temporal smoothing to avoid flickering.

> **Current Status:** Active development — v10. Dataset fully rebuilt from high-quality clean sources (AffectNet, RAF-DB, CK+, AffectNet-Short) — FER2013 noise eliminated across all classes. Training pipeline is stable and tested on RTX 3050 4GB.

---

## Architecture Deep Dive

The heart of this project is **HybridEmotionNet** — a dual-branch neural network that fuses two complementary representations of the same face.

<div align="center">

![Architecture Diagram](Architecture%20digram.png)

</div>

### Why Hybrid?

| Approach | Strength | Weakness |
|----------|----------|----------|
| **CNN only** (raw pixels) | Captures texture, color, lighting | Sensitive to pose, occlusion, resolution |
| **Landmarks only** (coordinates) | Invariant to lighting, color, pose | Loses subtle muscle texture cues |
| **Hybrid (this project)** ✅ | Gets both — structure + appearance | Higher complexity, requires MediaPipe |

### Architecture Details

```
Input: webcam frame
  │
  ├── MediaPipe FaceMesh ──► 478 × (x,y,z) = 1,434 landmark coordinates
  │                              │
  │                              ▼
  │                     CoordinateBranch (MLP)
  │                     1434 → 512 → 384 → 256
  │                              │
  │                              ▼ [B, 256] geometry features
  │
  ├── Face crop (224×224) ──► EfficientNet-B0 (ImageNet pretrained)
  │    (adaptive padding)         blocks 0–2 frozen, 3–8 fine-tuned
  │                              │
  │                              ▼ [B, 256] appearance features
  │
  └── Bidirectional Cross-Attention
        coord → CNN  (coord attends to appearance)
        CNN  → coord (appearance attends to geometry)
              │
              ▼ [B, 512] fused features
        Fusion MLP: 512 → 384 → 256 → 128
              │
              ▼
        Classifier: 128 → 7 emotions
```

The **bidirectional cross-attention** is key: each branch dynamically weights the other's features. A face in unusual lighting relies more on geometry; a subtle micro-expression with minimal landmark shift relies more on CNN texture.

---

## Emotion Classes

| # | Emotion | Key Facial Signals |
|---|---------|-------------------|
| 1 | Angry | Furrowed brows, tightened jaw, compressed lips |
| 2 | Disgust | Raised upper lip, wrinkled nose, lowered brow |
| 3 | Fear | Wide eyes, raised brows, open mouth |
| 4 | Happy | Raised cheeks, crow's feet, open smile |
| 5 | Neutral | Relaxed face, no strong deformation |
| 6 | Sad | Lowered brow corners, downward lip corners |
| 7 | Surprised | Raised brows, wide eyes, dropped jaw |

---

## Project Structure

```
v10/
├── visage_er/                  # Core ML library package
│   ├── config.py               # All hyperparameters and paths (single source of truth)
│   ├── utils.py                # Logging, SQLite experiment DB, visualization helpers
│   ├── models/
│   │   ├── hybrid_model.py     # HybridEmotionNet — production model
│   │   └── enhanced_model.py   # CoordinateEmotionNet — coordinate-only model
│   ├── data/
│   │   └── processor.py        # Dataset loading, augmentation, weighted sampling, caching
│   ├── core/
│   │   └── face_processor.py   # MediaPipe landmark extraction, quality validation
│   ├── training/
│   │   └── trainer.py          # Full training loop with AMP, early stopping, metrics
│   └── app/
│       └── gui_app.py          # Tkinter GUI with live overlay and session analytics
│
├── scripts/                    # CLI entry points
│   ├── prepare_dataset.py      # Master pipeline: download → import → filter → balance
│   ├── download_data.py        # Multi-source HuggingFace + Kaggle downloader
│   ├── import_ckplus.py        # CK+48 local folder importer
│   ├── filter_dataset.py       # ViT-based confidence quality filter
│   ├── balance_dataset.py      # Class balance capper (reversible)
│   ├── train_hybrid.py         # Train hybrid model (recommended)
│   └── train.py                # Train coordinate-only model
│
├── inference/                  # Real-time inference entry points
│   ├── run_hybrid.py           # Live webcam — hybrid model ← use this
│   └── run_coordinate.py       # Live webcam — coordinate-only model
│
├── dataset/
│   ├── train/<emotion>/        # Training images (~30k, all clean sources)
│   └── val/<emotion>/          # Validation images
│
├── models/                     # Saved model artifacts (weights gitignored — see HuggingFace)
│   ├── weights/                # .pth checkpoint files
│   │   └── hybrid_best_model.pth       # ← download from HuggingFace
│   ├── scalers/                # Preprocessing scalers (.pkl)
│   │   └── hybrid_coordinate_scaler.pkl
│   ├── cache/                  # MediaPipe feature cache (built on first run, reused every epoch)
│   │   ├── train/
│   │   └── val/
│   └── experiments/            # Per-run JSON metrics + training plots
│
├── checkpoints/                # Epoch-level training checkpoints
├── logs/                       # Training and inference logs + TensorBoard
└── requirements.txt
```

Dataset directories must use **capitalized** emotion names: `Angry/`, `Disgust/`, `Fear/`, `Happy/`, `Neutral/`, `Sad/`, `Surprised/`.

---

## Installation & Setup

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | CPU-only | 4 GB+ (NVIDIA) |
| Webcam | — | Required for real-time inference |

### Step 1 — Clone & create environment

```bash
git clone https://github.com/Huuffy/VisageCNN.git
cd VisageCNN
python -m venv venv
```

Activate the environment:

| Platform | Command |
|----------|---------|
| Windows | `venv\Scripts\activate` |
| Linux / macOS | `source venv/bin/activate` |

### Step 2 — Install PyTorch

PyTorch must be installed separately because the correct build depends on your hardware.

**NVIDIA GPU (CUDA 12.6) — recommended for training:**
```bash
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

**NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (inference works, training is slow):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon (macOS M1/M2/M3):**
```bash
pip install torch torchvision
```

> Check your CUDA version: `nvidia-smi`

### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Download Pre-Trained Weights

Model weights are hosted on HuggingFace (72 MB — too large for GitHub).

### Option A — HuggingFace CLI

```bash
# Install the HF CLI (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# Download weights into the correct folders
hf download Huuffy/VisageCNN models/weights/hybrid_best_model.pth --local-dir .
hf download Huuffy/VisageCNN models/scalers/hybrid_coordinate_scaler.pkl --local-dir .
```

### Option B — Python

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

### Option C — Manual

Go to [huggingface.co/Huuffy/VisageCNN](https://huggingface.co/Huuffy/VisageCNN), download the two files, and place them at:
```
models/weights/hybrid_best_model.pth
models/scalers/hybrid_coordinate_scaler.pkl
```

---

## Real-Time Inference

```bash
# Hybrid model — recommended
python inference/run_hybrid.py

# With a specific checkpoint
python inference/run_hybrid.py --model models/weights/hybrid_best_model.pth

# Coordinate-only model (faster, less accurate)
python inference/run_coordinate.py --model-path models/weights/best_model.pth
```

Press **Q** to quit.

---

## GUI Application

The GUI provides a full session interface with landmark visualization, live probability bars, and session export.

```bash
python -c "from visage_er.app.gui_app import launch_app; launch_app()"
```

Session data is logged to `logs/expressions.db`.

---

## Dataset Setup

### Option A — Automated (recommended)

The full pipeline downloads ~30k clean images from HuggingFace public datasets, optionally imports a local CK+48 folder, runs a quality filter, and balances class counts in one command:

```bash
pip install datasets   # HuggingFace datasets library

# Full pipeline (downloads ~30k images, ~5-10 min)
python scripts/prepare_dataset.py

# Skip download if data already present
python scripts/prepare_dataset.py --skip-download

# Download specific sources only
python scripts/prepare_dataset.py --sources rafdb affectnet
```

**Sources used:**

| Source | Size | Type |
|--------|------|------|
| `AutumnQiu/fer2013` | 35k | 48×48 grayscale — Surprised/Happy only |
| `deanngkl/raf-db-7emotions` | 20k | Real-world color |
| `Piro17/affectnethq` | 28k | High-res color |
| `Mengyuh/ExpW_preprocessed` | 2.8k | Wild faces |
| `Mauregato/affectnet_short` | 29k | High-quality — used for Fear/Sad/Angry/Disgust |
| CK+48 (local, optional) | ~1k | Lab-controlled poses |

### Option B — Bring your own images

Organize labeled face images and run the split script:

```
dataset/train/
├── Angry/
├── Disgust/
├── Fear/
├── Happy/
├── Neutral/
├── Sad/
└── Surprised/
```

```bash
python scripts/prepare_dataset.py --skip-download
```

---

## Training

```bash
# Train hybrid model — best accuracy
python scripts/train_hybrid.py

# Custom parameters
python scripts/train_hybrid.py --epochs 300 --batch-size 96
```

The MediaPipe feature cache is built automatically on the first epoch into `models/cache/` — all subsequent epochs are **10-100× faster**. The best checkpoint is saved to `models/weights/hybrid_best_model.pth`.

**After changing the dataset, delete the cache first:**
```bash
# Windows
rmdir /s /q models\cache

# Linux / macOS
rm -rf models/cache/
```

### Key Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 128 | Fits in 4GB VRAM |
| Effective batch | 256 | Gradient accumulation ×2 |
| Max epochs | 300 | With early stopping (patience=40) |
| Base LR | 1e-4 | CNN branch 5e-5, fusion 5e-4 (differential) |
| Optimizer | AdamW | Weight decay 0.05 |
| Loss | Focal (γ=2.0) | + label smoothing 0.12 |
| Face crop size | 224×224 | EfficientNet-B0 input |
| Landmark features | 1,434 | 478 landmarks × (x, y, z) |
| Mixed precision | enabled | torch.amp |
| Augmentation | CutMix (p=0.3) | + noise, rotation, zoom, flip |

### Coordinate-only model (lightweight alternative)

```bash
python scripts/train.py --epochs 300 --experiment-name "my_run"
```

---

## Model Performance

> Results reflect the current training run on the cleaned dataset. Numbers will be updated as training completes.

| Metric | Value |
|--------|-------|
| Model size | ~72 MB |
| Parameters | 6,225,795 total / 5,749,795 trainable |
| Inference time | ~12 ms/frame |
| VRAM (training) | ~3.5 GB |
| FPS on RTX 3050 | 30+ |

### Dataset Composition (training set — ~30k images, all clean sources)

| Class | Images | Sources |
|-------|--------|---------|
| Angry | 6,130 | RAF-DB + AffectNet + AffectNet-Short + CK+ |
| Surprised | 5,212 | RAF-DB + AffectNet + FER2013 |
| Sad | 4,941 | RAF-DB + AffectNet + AffectNet-Short + CK+ |
| Disgust | 3,782 | AffectNet-Short + RAF-DB + CK+ |
| Neutral | 3,475 | RAF-DB + AffectNet |
| Fear | 3,418 | AffectNet-Short + RAF-DB + CK+ |
| Happy | 3,124 | RAF-DB + AffectNet |

Max class imbalance: **1.97×** — significantly better than the typical 5-10× seen in raw FER2013-only setups.

---

## Technical Decisions

**Why EfficientNet-B0 over MobileNetV3?**
EfficientNet-B0 provides a better accuracy-efficiency tradeoff for the appearance branch. Blocks 0–2 (low-level edges/textures) are frozen with pretrained ImageNet weights; blocks 3–8 are fine-tuned to learn expression-specific cues. This keeps VRAM usage manageable while preserving strong feature extraction.

**Why MediaPipe over a standalone face detector?**
MediaPipe gives us precise 3D landmark coordinates at ~5ms per frame. The (x, y, z) coordinates encode depth and pose information that 2D pixel grids cannot represent, and the landmarks are topology-consistent (landmark 33 is always the nose tip), so the MLP encoder can learn meaningful spatial relationships.

**Why Focal Loss?**
Standard cross-entropy is dominated by easy examples (Happy, Neutral). Focal Loss (γ=2.0) down-weights well-classified examples and focuses gradient updates on hard boundary cases — particularly the Disgust/Angry/Fear confusion triangle where subtle landmark differences matter most.

**Why cache landmarks to disk?**
MediaPipe inference takes ~20ms per image. With 30k training images and 300 epochs, re-running MediaPipe each epoch would add ~50 hours of wasted compute. The cache is built once in epoch 1 and reused for all subsequent epochs, reducing per-epoch data loading by 10-100×.

**Why bidirectional cross-attention?**
Unidirectional attention (landmark→CNN only) means the CNN branch has no say in which geometric features matter. Bidirectional attention lets each branch dynamically weight the other's contribution based on the input — a face in poor lighting can lean on geometry; a subtle micro-expression with minimal landmark shift can lean on CNN texture.

---

## Development Notes

- All hyperparameters live in `visage_er/config.py` — never hardcode values inline
- Delete `models/cache/` to force a full MediaPipe cache rebuild (required after dataset changes)
- Every training run is logged to `logs/experiments.db` — query it to compare runs
- `logs/tensorboard/` contains TensorBoard-compatible event files
- On Windows, DataLoader uses `num_workers=4` with persistent workers disabled
- `balance_dataset.py` moves excess images to `dataset/excess/` — fully reversible with `--restore`
- `filter_dataset.py` quarantines low-confidence images to `dataset/rejected/` — never deletes

---

## Roadmap

- [ ] Achieve 85%+ per-class accuracy across all 7 emotions simultaneously
- [ ] Transformer backbone (ViT-Tiny) as CNN branch alternative
- [ ] Multi-face support — currently processes the primary detected face only
- [ ] ONNX export + quantization for CPU deployment without PyTorch
- [ ] Web inference demo via FastAPI + WebSocket
- [ ] Continuous emotion intensity output (not just discrete labels)
- [ ] Compound emotion detection (e.g., happy-surprised simultaneously)
- [x] AffectNet + RAF-DB dataset integration
- [x] Confidence-based quality filtering pipeline
- [x] CK+48 import pipeline
- [x] Automated class balancing (reversible)
- [x] Bidirectional cross-attention fusion

---

## Dependencies

```
torch              # Deep learning framework + CUDA support
torchvision        # Pretrained CNN backbones (EfficientNet-B0)
mediapipe          # Face mesh — 478 3D landmark extraction
opencv-python      # Webcam capture, frame preprocessing
albumentations     # Coordinate and image augmentation pipeline
scikit-learn       # Metrics, RobustScaler, class weight computation
numpy              # Numerical operations
Pillow             # Image loading and utilities
matplotlib         # Training curve plots
seaborn            # Confusion matrix heatmaps
tqdm               # Training progress bars
datasets           # HuggingFace datasets (for data pipeline only)
huggingface-hub    # Weight download from HuggingFace
```

---

<div align="center">

**Built with curiosity and a lot of training runs**

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer)

</div>
