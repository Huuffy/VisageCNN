<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=VisageCNN&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Real-Time%20Facial%20Expression%20Recognition&descAlignY=60&descAlign=50)

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=06B6D4&center=true&vCenter=true&width=750&lines=Hybrid+CNN+%2B+MediaPipe+Landmark+Architecture;7+Emotion+Classes+%E2%80%94+Real-Time+at+30+FPS;Cross-Attention+Fusion+%7C+EfficientNet-B2+%2B+478+Landmarks;Optimized+for+RTX+3050+%E2%80%94+4GB+VRAM" alt="Typing SVG" /></a>

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

> **Current Status:**  — Phase 6 complete. Full dataset rebuild from 6 multi-source HuggingFace repositories (~75k images), ViT-scored quality filtering with per-class asymmetric thresholds, perfectly balanced classes (10,768/class × 7). **87.9% validation accuracy. Disgust recall recovered from 51% → 92%. Fear recall improved from 65% → 75%.**

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
  ├── Face crop (224×224) ──► EfficientNet-B2 (ImageNet pretrained)
  │    (adaptive padding)         blocks 0–1 frozen, blocks 2–8 fine-tuned
  │                              per-block gradient checkpointing
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
/
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
├── inference/                  # Real-time inference entry points
│   ├── run_hybrid.py           # Live webcam — hybrid model ← use this
│   └── run_coordinate.py       # Live webcam — coordinate-only model
│
├── models/                     # Saved model artifacts (weights gitignored — see HuggingFace)
│   ├── weights/
│   │   ├── hybrid_best_model.pth       # ← download from HuggingFace
│   │   └── hybrid_swa_final.pth        # ← SWA averaged model (optional ensemble)
│   ├── scalers/
│   │   └── hybrid_coordinate_scaler.pkl
│   └── cache/                  # MediaPipe feature cache (built on first run, reused every epoch)
│
├── dataset/
│   ├── train/<emotion>/        # 8,614 images per class (60,298 total)
│   └── val/<emotion>/          # 2,154 images per class (15,078 total)
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

**NVIDIA GPU (CUDA 12.6) — recommended:**
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

Model weights are hosted on HuggingFace (too large for GitHub).

### Option A — HuggingFace CLI

```bash
# Install the HF CLI
pip install huggingface_hub

# Download weights into the correct folders
huggingface-cli download Huuffy/VisageCNN models/weights/hybrid_best_model.pth --local-dir .
huggingface-cli download Huuffy/VisageCNN models/weights/hybrid_swa_final.pth --local-dir .
huggingface-cli download Huuffy/VisageCNN models/scalers/hybrid_coordinate_scaler.pkl --local-dir .
```

### Option B — Python

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

### Option C — Manual

Go to [huggingface.co/Huuffy/VisageCNN](https://huggingface.co/Huuffy/VisageCNN), download the files, and place them at:
```
models/weights/hybrid_best_model.pth
models/weights/hybrid_swa_final.pth        ← optional, for ensemble mode
models/scalers/hybrid_coordinate_scaler.pkl
```

---

## Real-Time Inference

```bash
# Hybrid model — recommended
python inference/run_hybrid.py

# With SWA ensemble (slightly smoother, ~same accuracy)
python inference/run_hybrid.py --ensemble

# With a specific checkpoint
python inference/run_hybrid.py --model models/weights/hybrid_best_model.pth
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

## Model Performance

>  results — trained on 75,376 balanced images (10,768 per class × 7 emotions).

| Metric | Value |
|--------|-------|
| **Validation accuracy** | **87.9%** |
| **Macro F1** | **0.88** |
| Model size | ~90 MB (B2 backbone) |
| Inference time | ~12 ms/frame |
| VRAM (training) | ~3.5 GB |
| FPS on RTX 3050 | 30+ |

### Per-Class Results (val set, epoch 88)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Angry | 0.85 | 0.83 | 0.84 |
| **Disgust** | **0.97** | **0.90** | **0.94** |
| Fear | 0.89 | 0.75 | 0.82 |
| Happy | 0.97 | 0.99 | 0.98 |
| Neutral | 0.85 | 0.91 | 0.88 |
| Sad | 0.78 | 0.88 | 0.83 |
| Surprised | 0.83 | 0.90 | 0.86 |

### Dataset Composition ( — balanced, 10,768 per class)

| Class | Images | Primary Sources |
|-------|--------|-----------------|
| Angry | 10,768 | AffectNet + RAF-DB + FER2013 + AffectNet-Short |
| Disgust | 10,768 | AffectNet + RAF-DB + FER2013 + AffectNet-Short |
| Fear | 10,768 | AffectNet + RAF-DB + FER2013 + AffectNet-Short |
| Happy | 10,768 | AffectNet + RAF-DB + FER2013 |
| Neutral | 10,768 | AffectNet + RAF-DB + FER2013 |
| Sad | 10,768 | AffectNet + RAF-DB + AffectNet-Short |
| Surprised | 10,768 | AffectNet + RAF-DB + FER2013 |

**Max class imbalance: 1.0×** — perfectly balanced. All images passed a two-stage quality filter (MediaPipe face detection + ViT confidence scoring with per-class asymmetric thresholds).

---

## Training

```bash
# Train hybrid model — best accuracy
python scripts/train_hybrid.py

# Custom parameters
python scripts/train_hybrid.py --epochs 100 --batch-size 96
```

The MediaPipe feature cache is built automatically on the first epoch into `models/cache/` — all subsequent epochs are **10-100× faster**. The best checkpoint (by macro F1) is saved to `models/weights/hybrid_best_model.pth`. SWA weights are saved to `models/weights/hybrid_swa_final.pth`.

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
| Batch size | 96 | Fits in 4GB VRAM |
| Effective batch | 192 | Gradient accumulation ×2 |
| Max epochs | 100 | With early stopping (patience=15) |
| Base LR | 1e-4 | CNN branch 5e-5, fusion 5e-4 (differential) |
| Optimizer | AdamW | Weight decay 0.05 |
| Loss | Focal (γ=2.0) | + label smoothing 0.12 |
| Face crop size | 224×224 | EfficientNet-B2 input |
| Landmark features | 1,434 | 478 landmarks × (x, y, z) |
| Mixed precision | enabled | torch.amp |
| Best model saved by | Macro F1 | Not val accuracy — better for hard classes |
| SWA | epochs 30–70 | Stochastic Weight Averaging snapshot ensemble |

---

## Technical Decisions

**Why EfficientNet-B2 over B0?**
EfficientNet-B2 has a larger receptive field and more channels than B0, capturing finer-grained texture cues — particularly important for subtle expressions like Disgust (wrinkled nose) and Fear (periorbital tension). Blocks 0–1 are frozen with pretrained ImageNet weights; blocks 2–8 are fine-tuned with per-block gradient checkpointing to keep VRAM under 4GB.

**Why MediaPipe over a standalone face detector?**
MediaPipe gives us precise 3D landmark coordinates at ~5ms per frame. The (x, y, z) coordinates encode depth and pose information that 2D pixel grids cannot represent, and the landmarks are topology-consistent (landmark 33 is always the nose tip), so the MLP encoder can learn meaningful spatial relationships.

**Why Focal Loss?**
Standard cross-entropy is dominated by easy examples (Happy, Neutral). Focal Loss (γ=2.0) down-weights well-classified examples and focuses gradient updates on hard boundary cases — particularly the Disgust/Angry/Fear confusion triangle where subtle landmark differences matter most.

**Why save by macro F1 instead of val accuracy?**
Val accuracy is dominated by high-count easy classes (Happy at 99%). A model that sacrifices Disgust (hard, few examples) for Happy can improve val accuracy while degrading on the classes that matter most. Macro F1 weighs all 7 classes equally — a drop in Disgust recall will always hurt the saved model.

**Why SWA (Stochastic Weight Averaging)?**
Late-training weight noise causes the model to jump between good solutions without converging. SWA averages weights from epochs 30–70 into a single flat, wide minimum that generalizes better than any single checkpoint, typically recovering 0.5–1.5% on hard classes.

**Why cache landmarks to disk?**
MediaPipe inference takes ~20ms per image. With 60k training images and 100 epochs, re-running MediaPipe each epoch would add ~33 hours of wasted compute. The cache is built once in epoch 1 and reused for all subsequent epochs, reducing per-epoch data loading by 10-100×.

**Why bidirectional cross-attention?**
Unidirectional attention (landmark→CNN only) means the CNN branch has no say in which geometric features matter. Bidirectional attention lets each branch dynamically weight the other's contribution based on the input — a face in poor lighting can lean on geometry; a subtle micro-expression with minimal landmark shift can lean on CNN texture.

---

## Development Notes

- All hyperparameters live in `visage_er/config.py` — never hardcode values inline
- Delete `models/cache/` to force a full MediaPipe cache rebuild (required after dataset changes)
- Every training run is logged to `logs/experiments.db` — query it to compare runs
- `logs/tensorboard/` contains TensorBoard-compatible event files
- On Windows, DataLoader uses `num_workers=0` (required for Windows multiprocessing compatibility)
- Both `hybrid_best_model.pth` (best macro F1) and `hybrid_swa_final.pth` (epoch 30-70 average) are available for inference

---

## Roadmap

- [x] Achieve 85%+ val accuracy overall
- [x] Disgust recall above 85% — recovered from 51% → 90%+ via multi-source data + asymmetric ViT filtering
- [x] Fear recall above 70% — improved from 65% → 75% via targeted data + Fear-specific thresholds
- [x] AffectNet + RAF-DB + FER2013 dataset integration (6 HuggingFace sources)
- [x] ViT-scored quality filtering pipeline with per-class asymmetric thresholds
- [x] Perfectly balanced dataset (10,768 per class)
- [x] Bidirectional cross-attention fusion
- [x] Save-by-macro-F1 training strategy
- [x] SWA (epoch 30–70) for hard-class recovery
- [ ] Fear recall above 80% — next target
- [ ] Transformer backbone (ViT-Tiny) as CNN branch alternative
- [ ] Multi-face support — currently processes the primary detected face only
- [ ] ONNX export + quantization for CPU deployment without PyTorch
- [ ] Web inference demo via FastAPI + WebSocket
- [ ] Continuous emotion intensity output (not just discrete labels)
- [ ] Compound emotion detection (e.g., happy-surprised simultaneously)

---

## Dependencies

```
torch              # Deep learning framework + CUDA support
torchvision        # Pretrained CNN backbones (EfficientNet-B2)
mediapipe          # Face mesh — 478 3D landmark extraction
opencv-python      # Webcam capture, frame preprocessing
scikit-learn       # StandardScaler, metrics
numpy              # Numerical operations
Pillow             # Image loading and utilities
matplotlib         # Training curve plots
seaborn            # Confusion matrix heatmaps
tqdm               # Training progress bars
huggingface-hub    # Weight download from HuggingFace
```

---

<div align="center">

**Built with curiosity and a lot of training runs**

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer)

</div>
