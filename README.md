<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=VisageCNN&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Real-Time%20Facial%20Expression%20Recognition&descAlignY=60&descAlign=50)

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=06B6D4&center=true&vCenter=true&width=750&lines=Hybrid+CNN+%2B+MediaPipe+Landmark+Architecture;7+Emotion+Classes+%E2%80%94+Real-Time+at+30+FPS;Cross-Attention+Fusion+%7C+MobileNetV3+%2B+478+Landmarks;Optimized+for+RTX+3050+%E2%80%94+4GB+VRAM" alt="Typing SVG" /></a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00BCD4?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## 🎯 What Is This Project?

VisageCNN is an ongoing research and engineering project aimed at building a **better-than-baseline facial expression recognition system** for real-time use. The core idea is: standard CNN-only approaches that operate purely on raw pixel data miss a lot of structural information about the human face. By combining **appearance features** (what the face looks like) with **geometric features** (how the face is shaped and positioned in 3D space), the model understands expressions the way humans intuitively do — through both visual texture and facial structure.

The system processes webcam frames live at **30 FPS**, extracts **478 3D facial landmarks** using MediaPipe, crops the face region, fuses both sources of information through a **cross-attention mechanism**, and classifies the result into one of 7 emotion categories with temporal smoothing to avoid flickering.

> **Current Status:** Active development — v10. The hybrid model consistently outperforms the coordinate-only model. Per-class accuracy target is 85%+ across all 7 emotions. The training pipeline is stable and tested on RTX 3050 4GB.

---

## 🧠 Architecture Deep Dive

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

The cross-attention layer is key: it lets the CNN branch ask "where should I focus given the facial structure?" and the landmark branch ask "which geometric patterns are most discriminative given what the face looks like?" — they inform each other rather than simply concatenating.

---

## 🎭 Emotion Classes

The model is trained to recognize **7 universal emotion categories**:

| # | Emotion | Key Facial Signals |
|---|---------|-------------------|
| 1 | 😠 Angry | Furrowed brows, tightened jaw, compressed lips |
| 2 | 🤢 Disgust | Raised upper lip, wrinkled nose, lowered brow |
| 3 | 😨 Fear | Wide eyes, raised brows, open mouth |
| 4 | 😊 Happy | Raised cheeks, crow's feet, open smile |
| 5 | 😐 Neutral | Relaxed face, no strong deformation |
| 6 | 😢 Sad | Lowered brow corners, downward lip corners |
| 7 | 😲 Surprised | Raised brows, wide eyes, dropped jaw |

---

## 📁 Project Structure

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
├── scripts/                    # CLI training entry points
│   ├── train_hybrid.py         # Train hybrid model (recommended)
│   ├── train.py                # Train coordinate-only model
│   └── prepare_data.py         # Build 80/20 train/val dataset split
│
├── inference/                  # Real-time inference entry points
│   ├── run_hybrid.py           # Live webcam — hybrid model ← use this
│   └── run_coordinate.py       # Live webcam — coordinate-only model
│
├── dataset/
│   ├── train/<emotion>/        # Training images
│   └── val/<emotion>/          # Validation images
│
├── models/                     # Saved model artifacts (weights gitignored)
│   ├── weights/                # .pth checkpoint files
│   │   └── hybrid_best_model.pth
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

## ⚙️ Training Pipeline

```
dataset/train/  (raw labeled images)
        │
        ▼
scripts/prepare_data.py     ← Run once — creates 80/20 train/val split
        │
        ▼
scripts/train_hybrid.py     ← Main training entry point
   │
   ├── CachedHybridDataset  ← Runs MediaPipe once, caches to models/cache/
   │                           All subsequent epochs load from disk (10-100× faster)
   ├── WeightedRandomSampler← Over-samples rare classes automatically
   ├── AMP (torch.autocast) ← Mixed precision — halves VRAM usage
   ├── OneCycleLR scheduler ← Warmup + annealing, prevents early instability
   ├── AdamW optimizer      ← Weight decay = 0.01
   ├── Gradient accumulation← Steps=2, effective batch = 2 × BATCH_SIZE
   ├── Focal loss (γ=2.0)   ← Down-weights easy examples, focuses on hard ones
   ├── Label smoothing (0.1)← Prevents overconfident predictions
   └── Early stopping       ← Patience=40, saves best model automatically
        │
        ▼
models/weights/hybrid_best_model.pth  ← Best checkpoint
models/experiments/<run>/             ← JSON metrics + PNG plots
expressions.db                      ← SQLite log of all training runs
```

### Key Training Parameters (`visage_er/config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 96 | Fits in 4GB VRAM |
| Effective batch | 192 | Gradient accumulation ×2 |
| Max epochs | 300 | With early stopping |
| Base LR | 0.0001 | OneCycleLR peak |
| Optimizer | AdamW | Weight decay 0.01 |
| Face crop size | 112×112 | MobileNetV3 input |
| Landmark features | 1434 | 478 landmarks × (x, y, z) |
| Mixed precision | ✅ enabled | torch.amp |
| Dropout | 0.3 | Fusion network |
| Attention heads | 8 | Cross-attention |
| Early stop patience | 40 epochs | On val accuracy |

---

## 🚀 Installation & Setup

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | CPU-only | 4 GB+ (NVIDIA) |
| Webcam | — | Required for real-time inference |

---

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

---

### Step 2 — Install PyTorch

PyTorch must be installed separately because the correct build depends on your hardware.

**NVIDIA GPU (CUDA 12.6) — recommended for training:**
```bash
pip install torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

**NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (Windows / Linux / macOS — inference works, training is slow):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon (macOS M1/M2/M3 — uses MPS backend):**
```bash
pip install torch torchvision
```

> To check your CUDA version: `nvidia-smi` (Windows/Linux) or check the CUDA toolkit installed on your system.

---

### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Verify installation

```python
python -c "
import torch, cv2, mediapipe
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('OpenCV:', cv2.__version__)
print('MediaPipe:', mediapipe.__version__)
"
```

---

## 📂 Dataset Setup

Organize your labeled face images under `dataset/train/`:

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

Emotion folder names must be **exactly** as shown above (capitalized). Then generate the validation split:

```bash
python scripts/prepare_data.py
# Splits 20% of each class into dataset/val/
# Keeps 80% in dataset/train/ for training
```

---

## 🏋️ Custom Training

### GPU training (recommended)

```bash
# Train hybrid model — best accuracy
python scripts/train_hybrid.py

# Customize training parameters
python scripts/train_hybrid.py \
  --epochs 300 \
  --batch-size 96 \
```

The MediaPipe feature cache is built automatically on the first epoch into `models/cache/` — all subsequent epochs are **10-100× faster**. The best checkpoint is saved to `models/weights/hybrid_best_model.pth`.

### CPU training

CPU training works but is significantly slower. Reduce batch size to avoid memory issues:

```bash
python scripts/train_hybrid.py --epochs 50 --batch-size 16
```

Training on CPU is recommended only for testing that the pipeline works, not for production runs.

### Coordinate-only model (lightweight alternative)

```bash
python scripts/train.py --epochs 300 --experiment-name "my_run"
```

Faster to train, smaller model (~8 MB vs ~31 MB), but lower accuracy on subtle expressions.

### Monitor training

Training progress is printed per epoch. Results are also saved to:
- `models/experiments/<run>/` — loss and accuracy plots (PNG) + full metrics (JSON)
- `logs/` — timestamped training log files
- `logs/experiments.db` — SQLite record of every training run for comparison

---

## 🎥 Real-Time Inference

```bash
# Hybrid model — recommended
python inference/run_hybrid.py

# With a specific checkpoint
python inference/run_hybrid.py --model models/weights/hybrid_best_model.pth

# Coordinate-only model
python inference/run_coordinate.py --model-path models/weights/best_model.pth
```

Press **Q** to quit. Works on any platform where a webcam is available.

---

## 🖥️ GUI Application

The GUI provides a full session interface with landmark visualization, live probability bars, and session export.

**Windows:**
```bash
python -c "from visage_er.app.gui_app import launch_app; launch_app()"
```

**Linux** — requires a display server. If running headless, set `DISPLAY` first:
```bash
export DISPLAY=:0
python -c "from visage_er.app.gui_app import launch_app; launch_app()"
```

**macOS** — Tkinter is included with Python on macOS. If you see a blank window, install `python-tk` via Homebrew:
```bash
brew install python-tk
python -c "from visage_er.app.gui_app import launch_app; launch_app()"
```

The GUI connects to the trained model at `models/weights/hybrid_best_model.pth` by default. Session data is logged to `logs/expressions.db`.

---

## 📊 Model Performance

| Metric | Coordinate Model | Hybrid Model |
|--------|-----------------|--------------|
| Training accuracy | ~95% | ~96% |
| Validation accuracy | ~88% | ~91% |
| Inference time | ~5ms/frame | ~12ms/frame |
| Model size | ~8MB | ~31MB |
| VRAM usage | ~1.5GB | ~3.5GB |
| FPS on RTX 3050 | 60+ | 30+ |

The hybrid model trades some speed for significantly better generalization, especially on hard-to-distinguish classes like Disgust and Fear where landmark geometry alone is insufficient.

### Per-Class Difficulty Ranking

```
Easiest  ██████████████████████ Happy      (~96% val acc)
         █████████████████████  Surprised  (~94% val acc)
         ████████████████████   Neutral    (~93% val acc)
         ██████████████████     Angry      (~89% val acc)
         █████████████████      Sad        (~87% val acc)
         ████████████████       Fear       (~85% val acc)
Hardest  ████████████████       Disgust    (~84% val acc)
```

Disgust and Fear are the hardest classes — they share many landmark deformations and differ mostly in subtle eye-region and brow texture cues that the CNN branch helps resolve.

---

## 🔬 Key Technical Decisions

**Why MediaPipe over a standalone face detector?**
MediaPipe gives us precise 3D landmark coordinates at ~5ms per frame — faster than running a separate face detector plus pose estimator. The (x, y, z) coordinates encode depth and pose information that 2D pixel grids cannot represent, and the landmarks are topology-consistent (landmark 33 is always the nose tip, etc.), so the MLP encoder can learn meaningful spatial relationships.

**Why MobileNetV3-Small for the CNN branch?**
At 4GB VRAM, we need efficiency. MobileNetV3-Small with pretrained ImageNet weights gives strong appearance features with minimal memory footprint (~1.5M parameters), leaving room for the cross-attention fusion network and large batch sizes that improve generalization.

**Why OneCycleLR over cosine annealing?**
OneCycleLR's warmup phase prevents early loss spikes from the randomly-initialized fusion and classifier layers while the pretrained CNN branch is still adapting to the emotion domain. It also tends to converge to flatter minima which generalize better.

**Why cache landmarks to disk?**
MediaPipe inference takes ~20ms per image. With 5,000+ training images and 300 epochs, re-running MediaPipe each epoch would add 50+ hours of wasted compute. The cache is built once in epoch 1 and reused for all subsequent epochs, reducing per-epoch data loading time by 10-100×.

**Why cross-attention instead of simple concatenation?**
Concatenation treats both feature vectors as equally important regardless of context. Cross-attention lets each branch dynamically weight the other's features based on the current input — a face in unusual lighting can rely more on geometry, while a subtle micro-expression with minimal landmark shift can rely more on CNN texture features.

---

## 🛠️ Development Notes

- All hyperparameters live in `visage_er/config.py` — never hardcode values inline
- Delete `models/hybrid_cache_*/` to force a full cache rebuild (e.g., after changing image preprocessing)
- Every training run is logged to `expressions.db` — query it to compare runs
- `logs/tensorboard/` contains TensorBoard-compatible event files
- On Windows, DataLoader `num_workers` is forced to 0 to avoid multiprocessing spawn issues
- The `balance_dataset.py` script is idempotent — safe to re-run

---

## 🗺️ Roadmap

- [ ] Achieve 85%+ per-class accuracy across **all** 7 emotions simultaneously
- [ ] Transformer backbone (ViT-Tiny) as an alternative to MobileNetV3 in the CNN branch
- [ ] Multi-face support — currently processes only the primary detected face
- [ ] ONNX export + quantization for CPU deployment without PyTorch dependency
- [ ] Web inference demo via FastAPI + WebSocket stream
- [ ] Larger dataset integration (AffectNet, RAF-DB, SFEW)
- [ ] Continuous emotion intensity output (not just discrete one-hot labels)
- [ ] Compound emotion detection (e.g., happy-surprised simultaneously)

---

## 📦 Dependencies

```
torch              # Deep learning framework + CUDA support
torchvision        # Pretrained CNN backbones (MobileNetV3-Small)
mediapipe          # Face mesh — 478 3D landmark extraction
opencv-python      # Webcam capture, frame preprocessing
albumentations     # Coordinate and image augmentation pipeline
scikit-learn       # Metrics, StandardScaler, class weight computation
numpy              # Numerical operations
Pillow             # Image loading and utilities
matplotlib         # Training curve plots
seaborn            # Confusion matrix heatmaps
tqdm               # Training progress bars
pytz               # Timezone-aware timestamps in GUI session logs
psutil             # System memory and CPU monitoring
```

---

<div align="center">


**Built with curiosity and a lot of training runs**


![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer)

</div>
