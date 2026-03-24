"""Training pipeline for HybridEmotionNet.

Builds a MediaPipe face mesh cache on the first epoch, then trains the
EfficientNet-B2 + MLP hybrid model using OneCycleLR scheduling, AMP, and
weighted random sampling.
"""

import warnings
import os

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
import cv2
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import pickle
import random
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from visage_er.config import Config
from visage_er.models.hybrid_model import create_hybrid_model, HybridEmotionNet


class CachedHybridDataset(Dataset):
    """PyTorch Dataset that caches MediaPipe coordinates and face crops to disk.

    On the first call, each image is processed by MediaPipe FaceMesh and the
    resulting coordinate array and face crop are saved as a compressed .npz file.
    All subsequent epochs load directly from the cache, eliminating per-epoch
    MediaPipe overhead.

    Args:
        data_path: Root directory containing one subdirectory per emotion class.
        is_training: When True, applies data augmentation in ``__getitem__``.
        cache_dir: Directory for .npz cache files. Defaults to the split-specific
            subdirectory under ``Config.CACHE_PATH``.
    """

    FACE_CROP_SIZE = Config.FACE_CROP_SIZE

    def __init__(self, data_path: Path, is_training: bool = True, cache_dir: Path = None):
        self.data_path = data_path
        self.is_training = is_training
        self.cache_dir = cache_dir or (
            Config.MODELS_PATH / "cache" / ('train' if is_training else 'val')
        )

        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.samples = self._load_samples()
        self.cache_ready = self._check_cache()

        if not self.cache_ready:
            logging.info(
                f"Building cache for {'train' if is_training else 'val'} "
                f"set ({len(self.samples)} samples)…"
            )
            self._build_cache()
            self.cache_ready = True
        else:
            logging.info(
                f"Cache ready for {'train' if is_training else 'val'} "
                f"set ({len(self.samples)} samples)"
            )

        self.scaler = self._load_or_fit_scaler()

    def _load_samples(self) -> List[Tuple[Path, int, str]]:
        """Scan the dataset directory and return a list of (path, class_idx, emotion) tuples."""
        samples = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        for class_idx, emotion in enumerate(Config.EMOTION_CLASSES):
            emotion_path = self.data_path / emotion

            if not emotion_path.exists():
                for child in self.data_path.iterdir():
                    if child.is_dir() and child.name.lower() == emotion.lower():
                        emotion_path = child
                        break
                else:
                    logging.warning(f"Class directory not found: {emotion}")
                    continue

            for img_file in sorted(emotion_path.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    samples.append((img_file, class_idx, emotion))

        return samples

    def _check_cache(self) -> bool:
        """Return True if the cache exists and covers all samples in this split."""
        if not self.cache_dir.exists():
            return False

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return manifest.get('count', 0) == len(self.samples)

    @staticmethod
    def _prepare_bgr(img_path: Path) -> Optional[np.ndarray]:
        """Read an image and return a BGR uint8 array suitable for MediaPipe.

        Handles greyscale-to-BGR conversion and bicubic upscaling for images
        whose shortest side is below 112 px, which is the minimum MediaPipe
        FaceMesh reliably handles.

        Args:
            img_path: Path to the source image file.

        Returns:
            A HxWx3 uint8 BGR array, or None if the file cannot be read.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        min_dim = min(h, w)
        if min_dim < 112:
            scale = 256.0 / min_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return img

    def _build_cache(self):
        """Run MediaPipe on every image and save coordinates + face crop as .npz files.

        When MediaPipe fails to detect a face (e.g. for very small FER2013
        48×48 images), the sample is still saved with a zero coordinate vector
        and the full image as the face crop so the CNN branch can still learn
        from it.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )

        valid_count = 0
        landmark_failures = 0

        for idx, (img_path, class_idx, emotion) in enumerate(
            tqdm(self.samples, desc="Building cache")
        ):
            cache_file = self.cache_dir / f"{idx}.npz"

            if cache_file.exists():
                valid_count += 1
                continue

            img = self._prepare_bgr(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                landmark_failures += 1
                coords = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
                face_crop = cv2.resize(img, (self.FACE_CROP_SIZE, self.FACE_CROP_SIZE))
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                landmarks = results.multi_face_landmarks[0]

                coords = []
                for lm in landmarks.landmark:
                    coords.extend([lm.x * w, lm.y * h, lm.z * w])
                coords = np.array(coords, dtype=np.float32)

                coords_3d = coords.reshape(-1, 3)
                half_w, half_h = w / 2.0, h / 2.0
                coords_3d[:, 0] = (coords_3d[:, 0] - half_w) / half_w
                coords_3d[:, 1] = (coords_3d[:, 1] - half_h) / half_h
                coords_3d[:, 2] = coords_3d[:, 2] * 0.1
                coords = coords_3d.flatten()

                if len(coords) < Config.COORDINATE_DIM:
                    padded = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
                    padded[:len(coords)] = coords
                    coords = padded
                elif len(coords) > Config.COORDINATE_DIM:
                    coords = coords[:Config.COORDINATE_DIM]

                xs = [lm.x * w for lm in landmarks.landmark]
                ys = [lm.y * h for lm in landmarks.landmark]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                pad_w = int((x_max - x_min) * 0.2)
                pad_h = int((y_max - y_min) * 0.2)
                x_min = max(0, x_min - pad_w)
                x_max = min(w, x_max + pad_w)
                y_min = max(0, y_min - pad_h)
                y_max = min(h, y_max + pad_h)

                face_crop = img[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    face_crop = img

                face_crop = cv2.resize(face_crop, (self.FACE_CROP_SIZE, self.FACE_CROP_SIZE))
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            np.savez_compressed(
                cache_file,
                coords=coords,
                face_crop=face_crop_rgb,
                label=class_idx,
            )
            valid_count += 1

        face_mesh.close()

        manifest = {
            'count': len(self.samples),
            'valid': valid_count,
            'landmark_failures': landmark_failures,
            'created': datetime.now().isoformat(),
        }
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        logging.info(
            f"Cache built: {valid_count} valid "
            f"({landmark_failures} with landmark fallback)"
        )

    def _load_or_fit_scaler(self):
        """Load an existing RobustScaler or fit one from the training cache."""
        scaler_path = Config.MODELS_PATH / "scalers" / "hybrid_coordinate_scaler.pkl"

        if scaler_path.exists() and not self.is_training:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)

        if self.is_training:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()

            indices = random.sample(range(len(self.samples)), min(2000, len(self.samples)))
            coords_list = [
                np.load(self.cache_dir / f"{i}.npz")['coords']
                for i in indices
                if (self.cache_dir / f"{i}.npz").exists()
            ]

            if coords_list:
                scaler.fit(np.array(coords_list))
                scaler_path.parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logging.info(f"Scaler fitted on {len(coords_list)} samples")

            return scaler

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return (coords_tensor, crop_tensor, label_tensor) for one sample.

        Applies augmentation when ``is_training=True``.
        """
        cache_file = self.cache_dir / f"{idx}.npz"
        _, class_idx, _ = self.samples[idx]

        if not cache_file.exists():
            return (
                torch.zeros(Config.COORDINATE_DIM, dtype=torch.float32),
                torch.zeros(3, self.FACE_CROP_SIZE, self.FACE_CROP_SIZE, dtype=torch.float32),
                torch.tensor(class_idx, dtype=torch.long),
            )

        data = np.load(cache_file)
        coords = data['coords'].astype(np.float32)
        face_crop = data['face_crop']
        label = int(data['label'])

        if self.scaler is not None:
            try:
                coords = self.scaler.transform([coords])[0]
            except Exception:
                pass

        face_crop = face_crop.astype(np.float32) / 255.0

        if self.is_training:
            # Per-class augmentation boost: minority / ambiguous classes get
            # higher augmentation diversity to compensate for fewer samples.
            # Index mapping: 0=Angry 1=Disgust 2=Fear 3=Happy 4=Neutral 5=Sad 6=Surprised
            _AUG_BOOST = {0: 1.4, 1: 1.9, 2: 1.7, 5: 1.2, 6: 1.1}
            boost = _AUG_BOOST.get(label, 1.0)

            def _p(base: float) -> float:
                return min(1.0, base * boost)

            if random.random() < _p(0.5):
                face_crop = face_crop[:, ::-1, :].copy()
                coords_3d = coords.reshape(-1, 3)
                coords_3d[:, 0] = -coords_3d[:, 0]
                coords = coords_3d.flatten()

            if random.random() < _p(0.4):
                angle = random.uniform(-12, 12)
                h, w = face_crop.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                face_crop = cv2.warpAffine(face_crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            if random.random() < _p(0.5):
                face_crop = np.clip(face_crop * random.uniform(0.75, 1.25), 0, 1)

            if random.random() < _p(0.4):
                mean = face_crop.mean()
                face_crop = np.clip(mean + random.uniform(0.7, 1.3) * (face_crop - mean), 0, 1)

            if random.random() < _p(0.35):
                h, w = face_crop.shape[:2]
                eh = random.randint(h // 8, h // 3)
                ew = random.randint(w // 8, w // 3)
                y = random.randint(0, h - eh)
                x = random.randint(0, w - ew)
                face_crop[y:y + eh, x:x + ew] = np.random.uniform(0, 1, (eh, ew, 3))

            if random.random() < _p(0.5):
                coords += np.random.normal(0, 0.015, coords.shape).astype(np.float32)

            if random.random() < _p(0.35):
                scale = random.uniform(0.92, 1.08)
                coords_3d = coords.reshape(-1, 3)
                coords_3d[:, :2] *= scale
                coords = coords_3d.flatten()

            if random.random() < _p(0.4):
                zoom = random.uniform(0.65, 1.35)
                h, w = face_crop.shape[:2]
                new_h, new_w = max(1, int(h * zoom)), max(1, int(w * zoom))
                resized = cv2.resize(face_crop, (new_w, new_h))

                out = np.zeros((h, w, 3), dtype=face_crop.dtype)
                if zoom > 1.0:
                    y0 = (new_h - h) // 2
                    x0 = (new_w - w) // 2
                    out = resized[y0:y0 + h, x0:x0 + w]
                else:
                    y0 = (h - new_h) // 2
                    x0 = (w - new_w) // 2
                    out[y0:y0 + new_h, x0:x0 + new_w] = resized

                face_crop = out.copy()

        face_crop = (face_crop - self.img_mean) / self.img_std
        face_crop = face_crop.transpose(2, 0, 1)

        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(face_crop.copy(), dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Focal Loss down-weights well-classified examples so the model focuses on
    hard, mis-classified ones — especially useful for visually ambiguous classes
    like Disgust (often confused with Angry).

    Args:
        gamma: Default focusing parameter used when gamma_per_class is None.
            0 = standard cross-entropy. 2.0 is the value recommended by the
            original paper (Lin et al. 2017).
        alpha: Optional per-class weight tensor (same role as CrossEntropyLoss
            ``weight``). When supplied it additionally up-weights rare classes.
        label_smoothing: Label smoothing factor applied before focal weighting.
        gamma_per_class: Optional tensor of shape [num_classes] with a
            per-class gamma value. When provided, each sample's gamma is looked
            up from this tensor using its ground-truth label.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 label_smoothing: float = 0.0, gamma_per_class: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.gamma_per_class = gamma_per_class

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        pt = torch.exp(-ce)
        gamma = self.gamma_per_class[targets] if self.gamma_per_class is not None else self.gamma
        return (((1.0 - pt) ** gamma) * ce).mean()


_HARD_CLASSES = frozenset({0, 1, 2})  # Angry, Disgust, Fear


def cutmix_batch(
    coords: torch.Tensor,
    crops: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
    prob: float = 0.3,
):
    """Apply CutMix augmentation to one training batch.

    A random rectangular region of each image is replaced with the corresponding
    region from a randomly shuffled sample in the same batch.  Labels are mixed
    proportionally to the area ratio.  CutMix is only applied with probability
    ``prob`` so the un-augmented path stays dominant.

    The effective probability is reduced when hard classes (Angry, Disgust,
    Fear) dominate the batch, preventing CutMix from corrupting the clean
    gradients those classes need.

    Args:
        coords: Coordinate tensor [B, 1434].
        crops: Face crop tensor [B, 3, H, W].
        targets: Class index tensor [B].
        alpha: Beta distribution parameter controlling cut size.
        prob: Base probability of applying CutMix to a given batch.

    Returns:
        Tuple of (coords, crops, targets_a, targets_b, lam).
        When CutMix is not applied, targets_a == targets_b and lam == 1.0.
    """
    hard_fraction = (targets.unsqueeze(1).eq(
        torch.tensor(sorted(_HARD_CLASSES), device=targets.device)
    ).any(dim=1)).float().mean().item()
    effective_prob = prob * max(0.0, 1.0 - hard_fraction * 2.0)
    if random.random() > effective_prob:
        return coords, crops, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))
    B, _, H, W = crops.shape
    rand_idx = torch.randperm(B, device=crops.device)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)

    mixed_crops = crops.clone()
    mixed_crops[:, :, y1:y2, x1:x2] = crops[rand_idx, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (H * W)
    mixed_coords = lam * coords + (1.0 - lam) * coords[rand_idx]

    return mixed_coords, mixed_crops, targets, targets[rand_idx], lam


class HybridTrainer:
    """End-to-end trainer for HybridEmotionNet.

    Uses differential learning rates (lower for the pretrained CNN branch),
    OneCycleLR scheduling, gradient clipping, AMP, and weighted random sampling
    to handle class imbalance.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = create_hybrid_model(pretrained_cnn=True)

        self.swa_model = AveragedModel(self.model)
        self.swa_start_epoch = 30
        self.swa_end_epoch = 70
        self.swa_update_freq = 3

        gamma_per_class = torch.tensor(
            [2.5, 3.5, 3.0, 1.0, 1.0, 2.0, 1.5],  # Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
            dtype=torch.float32,
        ).to(self.device)
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(self.device)
        self.criterion = FocalLoss(
            gamma=2.0,
            alpha=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING,
            gamma_per_class=gamma_per_class,
        )
        self.val_criterion = nn.CrossEntropyLoss()

        cnn_params = list(self.model.cnn_branch.parameters())
        cnn_trainable = [p for p in cnn_params if p.requires_grad]
        other_params = [p for p in self.model.parameters()
                        if not any(p is cp for cp in cnn_params)]

        self.base_lr = 0.0005
        self.optimizer = optim.AdamW([
            {'params': cnn_trainable, 'lr': self.base_lr * 0.1},
            {'params': other_params, 'lr': self.base_lr},
        ], weight_decay=Config.WEIGHT_DECAY)

        self.use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.best_val_acc = 0.0
        self.best_macro_f1 = 0.0
        self.patience_counter = 0
        self.best_model_path = Config.MODELS_PATH / "weights" / "hybrid_best_model.pth"

        self.logger = logging.getLogger(__name__)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(
            f"HybridTrainer ready | {trainable:,} trainable params | "
            f"base_lr={self.base_lr} | CNN lr={self.base_lr * 0.1} | AMP={self.use_amp}"
        )

    def _create_data_loaders(self):
        """Instantiate cached datasets and weighted DataLoaders for train and val splits."""
        train_dataset = CachedHybridDataset(Config.DATASET_PATH / "train", is_training=True)
        val_dataset = CachedHybridDataset(Config.DATASET_PATH / "val", is_training=False)

        class_counts = {}
        for _, class_idx, _ in train_dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        total = len(train_dataset.samples)
        weights = [
            total / (len(class_counts) * class_counts[ci])
            for _, ci, _ in train_dataset.samples
        ]
        sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)

        batch_size = Config.BATCH_SIZE

        num_workers = Config.NUM_WORKERS
        persistent = num_workers > 0

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY, persistent_workers=persistent,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY, persistent_workers=persistent,
        )

        return train_loader, val_loader, batch_size

    def train_epoch(self, train_loader, epoch, scheduler) -> Tuple[float, float]:
        """Run one training epoch.

        Args:
            train_loader: Training DataLoader.
            epoch: Current epoch index (0-based).
            scheduler: OneCycleLR scheduler instance.

        Returns:
            Tuple of (average loss, accuracy %).
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for coords, crops, targets in pbar:
            coords = coords.to(self.device)
            crops = crops.to(self.device)
            targets = targets.to(self.device)

            coords, crops, targets_a, targets_b, lam = cutmix_batch(coords, crops, targets)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(coords, crops)
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1.0 - lam) * self.criterion(outputs, targets_b)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(coords, crops)
                loss = lam * self.criterion(outputs, targets_a) + \
                       (1.0 - lam) * self.criterion(outputs, targets_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets_a.size(0)
            correct += (predicted == targets_a).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.1f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            })

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader) -> Tuple[float, float, list, list, dict, dict]:
        """Run validation and return loss, accuracy, and per-class metrics.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Tuple of (loss, accuracy %, predictions list, targets list,
            per-class accuracy dict, extra metrics dict).
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for coords, crops, targets in tqdm(val_loader, desc="Validation", leave=False):
                coords = coords.to(self.device)
                crops = crops.to(self.device)
                targets = targets.to(self.device)

                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(coords, crops)
                        loss = self.val_criterion(outputs, targets)
                else:
                    outputs = self.model(coords, crops)
                    loss = self.val_criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        per_class = {}
        for i, emotion in enumerate(Config.EMOTION_CLASSES):
            mask = np.array(all_targets) == i
            per_class[emotion] = (
                100. * (np.array(all_preds)[mask] == i).sum() / mask.sum()
                if mask.sum() > 0 else 0.0
            )

        from sklearn.metrics import (
            f1_score, precision_score, recall_score,
            confusion_matrix as sk_cm,
        )
        t, p = np.array(all_targets), np.array(all_preds)
        macro_f1    = float(f1_score(t, p, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(t, p, average='weighted', zero_division=0))
        f1_per   = f1_score(t, p, average=None, zero_division=0)
        prec_per = precision_score(t, p, average=None, zero_division=0)
        rec_per  = recall_score(t, p, average=None, zero_division=0)
        cm       = sk_cm(t, p).tolist()

        per_class_f1   = {e: float(f1_per[i])   for i, e in enumerate(Config.EMOTION_CLASSES)}
        per_class_prec = {e: float(prec_per[i]) for i, e in enumerate(Config.EMOTION_CLASSES)}
        per_class_rec  = {e: float(rec_per[i])  for i, e in enumerate(Config.EMOTION_CLASSES)}

        extra = {
            'macro_f1':         macro_f1,
            'weighted_f1':      weighted_f1,
            'per_class_f1':     per_class_f1,
            'per_class_prec':   per_class_prec,
            'per_class_rec':    per_class_rec,
            'confusion_matrix': cm,
        }

        return total_loss / len(val_loader), 100. * correct / total, all_preds, all_targets, per_class, extra

    def train(self, num_epochs: int = 300, resume_path: str = None):
        """Execute the full training loop.

        Args:
            num_epochs: Maximum number of training epochs.
            resume_path: Path to a checkpoint saved by this trainer to resume
                from. When provided, model/optimizer/scheduler state and epoch
                counter are restored and training continues from the next epoch.
        """
        self.logger.info("Initialising data loaders…")
        train_loader, val_loader, batch_size = self._create_data_loaders()
        self.train_loader = train_loader

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.base_lr * 0.1, self.base_lr],
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100,
        )

        start_epoch = 0
        if resume_path:
            self.logger.info(f"Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            self.best_val_acc = ckpt.get('best_val_acc', ckpt.get('val_acc', 0.0))
            self.best_macro_f1 = ckpt.get('macro_f1', 0.0)
            self.patience_counter = ckpt.get('patience_counter', 0)
            if ckpt.get('swa_model_state_dict') is not None:
                self.swa_model.load_state_dict(ckpt['swa_model_state_dict'])
            self.logger.info(
                f"Resumed at epoch {start_epoch} | "
                f"best_val={self.best_val_acc:.1f}% | patience={self.patience_counter}"
            )

        self.logger.info(
            f"Training {num_epochs} epochs | batch={batch_size} | "
            f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)}"
        )

        exp_dir = (
            Config.MODELS_PATH / "experiments" /
            f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)

        latest_checkpoint_path = Config.MODELS_PATH / "weights" / "hybrid_latest.pth"

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_val_gap': [],
            'per_class': [],
            'macro_f1': [], 'weighted_f1': [],
            'per_class_f1': [], 'per_class_prec': [], 'per_class_rec': [],
            'confusion_matrix': [],
        }
        history_path = exp_dir / "training_history.json"

        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch, scheduler)
            val_loss, val_acc, preds, targets, per_class, extra = self.validate(val_loader)

            if self.swa_start_epoch <= epoch <= self.swa_end_epoch and (epoch - self.swa_start_epoch) % self.swa_update_freq == 0:
                self.swa_model.update_parameters(self.model)
                self.logger.info(f"  SWA: averaged weights at epoch {epoch + 1}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_val_gap'].append(train_acc - val_acc)
            history['per_class'].append(per_class)
            history['macro_f1'].append(extra['macro_f1'])
            history['weighted_f1'].append(extra['weighted_f1'])
            history['per_class_f1'].append(extra['per_class_f1'])
            history['per_class_prec'].append(extra['per_class_prec'])
            history['per_class_rec'].append(extra['per_class_rec'])
            history['confusion_matrix'].append(extra['confusion_matrix'])

            with open(history_path, 'w') as _hf:
                json.dump(history, _hf, default=str)

            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.logger.info(f"  Train  loss={train_loss:.4f}  acc={train_acc:.1f}%")
            self.logger.info(f"  Val    loss={val_loss:.4f}  acc={val_acc:.1f}%")
            self.logger.info(f"  Gap    {train_acc - val_acc:.1f}%")

            for emotion, acc in per_class.items():
                status = "ok" if acc >= 80 else "low" if acc < 60 else "improving"
                self.logger.info(f"    {emotion}: {acc:.1f}% [{status}]")

            try:
                report = classification_report(
                    targets, preds, target_names=Config.EMOTION_CLASSES, zero_division=0,
                )
                self.logger.info(f"\nClassification Report (epoch {epoch + 1}):\n{report}")
            except Exception as e:
                self.logger.warning(f"Could not generate classification report: {e}")

            macro_f1 = extra['macro_f1']
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1
                self.best_val_acc = val_acc
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'best_val_acc': self.best_val_acc,
                    'macro_f1': macro_f1,
                    'patience_counter': self.patience_counter,
                    'per_class_acc': per_class,
                    'swa_model_state_dict': self.swa_model.state_dict(),
                    'config': {
                        'num_classes': Config.NUM_CLASSES,
                        'coordinate_dim': Config.COORDINATE_DIM,
                    },
                }, self.best_model_path)

                self.logger.info(f"  New best macro F1: {macro_f1:.4f} (val={val_acc:.1f}%) → {self.best_model_path}")
            else:
                self.patience_counter += 1

            # Save latest checkpoint every epoch so training can be resumed
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': self.best_val_acc,
                'patience_counter': self.patience_counter,
                'swa_model_state_dict': self.swa_model.state_dict(),
            }, latest_checkpoint_path)

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if hasattr(self, 'swa_model'):
            self.logger.info("Running BN update for SWA model...")
            self.swa_model.train()
            with torch.no_grad():
                for coords, crops, _ in self.train_loader:
                    coords = coords.to(self.device)
                    crops = crops.to(self.device)
                    self.swa_model(coords, crops)
            swa_path = Config.MODELS_PATH / "weights" / "hybrid_swa_final.pth"
            torch.save(self.swa_model.module.state_dict(), swa_path)
            self.logger.info(f"SWA model saved: {swa_path}")

        print(f"\nTraining complete | Best val accuracy: {self.best_val_acc:.1f}%")
        print(f"Model saved: {self.best_model_path}")
        print(f"History saved: {history_path}")


def main():
    """Entry point — parse arguments and start training."""
    import argparse

    parser = argparse.ArgumentParser(description='Train HybridEmotionNet')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument(
        '--resume', type=str, default=None, metavar='CHECKPOINT',
        help='Resume training from checkpoint (e.g. models/weights/hybrid_latest.pth)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    Config.create_directories()

    print("=" * 60)
    print("HYBRID CNN + COORDINATE MODEL TRAINING")
    print("=" * 60)
    print(f"Device:    {Config.DEVICE}")
    print(f"Epochs:    {args.epochs}")
    print(f"Resume:    {args.resume or 'no (fresh run)'}")
    print(f"Model:     HybridEmotionNet (EfficientNet-B2 + Coordinate MLP)")
    print(f"LR:        {0.0005} (fusion/coord) / {0.00005} (CNN)")
    print(f"Scheduler: OneCycleLR with 5% warmup")
    print(f"Classes:   {Config.EMOTION_CLASSES}")
    print("=" * 60)

    trainer = HybridTrainer()
    trainer.train(num_epochs=args.epochs, resume_path=args.resume)


if __name__ == "__main__":
    main()
