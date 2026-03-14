"""
Hybrid Trainer v2 - Fixed training pipeline for HybridEmotionNet
Key fixes: proper LR, OneCycleLR scheduler, warmup, face crop caching
"""

# Suppress noisy warnings
import warnings
import os
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
import numpy as np
import cv2
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report
import mediapipe as mp
from typing import Tuple, Optional, List
import pickle
import random
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from visage_er.config import Config
from visage_er.models.hybrid_model import create_hybrid_model, HybridEmotionNet
from visage_er.core.face_processor import EnhancedFaceMeshProcessor


class CachedHybridDataset(Dataset):
    """Dataset that caches coordinates + face crops to disk on first epoch.
    This eliminates MediaPipe overhead after epoch 1.
    """

    FACE_CROP_SIZE = 112

    def __init__(self, data_path: Path, is_training: bool = True, cache_dir: Path = None):
        self.data_path = data_path
        self.is_training = is_training
        self.cache_dir = cache_dir or (Config.MODELS_PATH / "cache" / ('train' if is_training else 'val'))

        # ImageNet normalization for MobileNetV3
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load sample paths
        self.samples = self._load_samples()

        # Check if cache exists
        self.cache_ready = self._check_cache()

        if not self.cache_ready:
            logging.info(f"Building cache for {'train' if is_training else 'val'} set ({len(self.samples)} samples)...")
            self._build_cache()
            self.cache_ready = True
        else:
            logging.info(f"Cache loaded for {'train' if is_training else 'val'} set ({len(self.samples)} samples)")

        # Load coordinate scaler
        self.scaler = self._load_or_fit_scaler()

    def _load_samples(self) -> List[Tuple[Path, int, str]]:
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
                    logging.warning(f"Directory not found: {emotion}")
                    continue

            for img_file in sorted(emotion_path.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    samples.append((img_file, class_idx, emotion))

        return samples

    def _check_cache(self) -> bool:
        """Check if cache is complete"""
        if not self.cache_dir.exists():
            return False

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        return manifest.get('count', 0) == len(self.samples)

    def _build_cache(self):
        """Extract and cache all coordinates + face crops"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )

        valid_count = 0
        skipped = 0

        for idx, (img_path, class_idx, emotion) in enumerate(tqdm(self.samples, desc="Caching data")):
            cache_file = self.cache_dir / f"{idx}.npz"

            if cache_file.exists():
                valid_count += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                skipped += 1
                continue

            landmarks = results.multi_face_landmarks[0]

            # Extract coordinates
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x * w, lm.y * h, lm.z * w])
            coords = np.array(coords, dtype=np.float32)

            # Normalize coordinates
            coords_3d = coords.reshape(-1, 3)
            half_w, half_h = w / 2.0, h / 2.0
            coords_3d[:, 0] = (coords_3d[:, 0] - half_w) / half_w
            coords_3d[:, 1] = (coords_3d[:, 1] - half_h) / half_h
            coords_3d[:, 2] = coords_3d[:, 2] * 0.1
            coords = coords_3d.flatten()

            # Ensure correct dimension
            if len(coords) < Config.COORDINATE_DIM:
                padded = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
                padded[:len(coords)] = coords
                coords = padded
            elif len(coords) > Config.COORDINATE_DIM:
                coords = coords[:Config.COORDINATE_DIM]

            # Extract face crop (raw, unnormalized for augmentation flexibility)
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

            # Save as compressed npz
            np.savez_compressed(cache_file,
                               coords=coords,
                               face_crop=face_crop_rgb,  # uint8 RGB
                               label=class_idx)
            valid_count += 1

        face_mesh.close()

        # Save manifest
        manifest = {
            'count': len(self.samples),
            'valid': valid_count,
            'skipped': skipped,
            'created': datetime.now().isoformat()
        }
        with open(self.cache_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)

        logging.info(f"Cache built: {valid_count} valid, {skipped} skipped")

    def _load_or_fit_scaler(self):
        scaler_path = Config.MODELS_PATH / "scalers" / "hybrid_coordinate_scaler.pkl"

        if scaler_path.exists() and not self.is_training:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)

        if self.is_training:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()

            coords_list = []
            indices = random.sample(range(len(self.samples)), min(500, len(self.samples)))
            for idx in indices:
                cache_file = self.cache_dir / f"{idx}.npz"
                if cache_file.exists():
                    data = np.load(cache_file)
                    coords_list.append(data['coords'])

            if coords_list:
                scaler.fit(np.array(coords_list))
                scaler_path.parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logging.info(f"Scaler fitted on {len(coords_list)} cached samples")

            return scaler
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_file = self.cache_dir / f"{idx}.npz"
        _, class_idx, _ = self.samples[idx]

        if not cache_file.exists():
            return (
                torch.zeros(Config.COORDINATE_DIM, dtype=torch.float32),
                torch.zeros(3, self.FACE_CROP_SIZE, self.FACE_CROP_SIZE, dtype=torch.float32),
                torch.tensor(class_idx, dtype=torch.long)
            )

        data = np.load(cache_file)
        coords = data['coords'].astype(np.float32)
        face_crop = data['face_crop']  # uint8 RGB [112, 112, 3]
        label = int(data['label'])

        # Apply scaler to coordinates
        if self.scaler is not None:
            try:
                coords = self.scaler.transform([coords])[0]
            except:
                pass

        # Process face crop: uint8 -> float32, normalize
        face_crop = face_crop.astype(np.float32) / 255.0

        # Augmentation during training
        if self.is_training:
            # Random horizontal flip
            if random.random() < 0.5:
                face_crop = face_crop[:, ::-1, :].copy()
                # Also flip X coordinates
                coords_3d = coords.reshape(-1, 3)
                coords_3d[:, 0] = -coords_3d[:, 0]
                coords = coords_3d.flatten()

            # Random brightness
            if random.random() < 0.3:
                face_crop = face_crop * random.uniform(0.85, 1.15)
                face_crop = np.clip(face_crop, 0, 1)

            # Add slight noise to coordinates
            if random.random() < 0.4:
                coords += np.random.normal(0, 0.01, coords.shape).astype(np.float32)

        # ImageNet normalization
        face_crop = (face_crop - self.img_mean) / self.img_std
        face_crop = face_crop.transpose(2, 0, 1)  # HWC -> CHW

        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(face_crop.copy(), dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


class HybridTrainer:
    """Fixed training pipeline with proper LR and scheduling"""

    def __init__(self):
        self.device = Config.DEVICE
        self.model = create_hybrid_model(pretrained_cnn=True)

        # Class-weighted loss
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.val_criterion = nn.CrossEntropyLoss()

        # Differential LR: CNN gets lower LR since pretrained
        cnn_params = list(self.model.cnn_branch.parameters())
        cnn_trainable = [p for p in cnn_params if p.requires_grad]
        other_params = [p for p in self.model.parameters()
                       if not any(p is cp for cp in cnn_params)]

        # Much higher LR for this smaller model
        self.base_lr = 0.001  # 10x higher than before
        self.optimizer = optim.AdamW([
            {'params': cnn_trainable, 'lr': self.base_lr * 0.1},   # CNN: 0.0001
            {'params': other_params, 'lr': self.base_lr}            # Rest: 0.001
        ], weight_decay=0.01)

        # AMP
        self.use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_path = Config.MODELS_PATH / "weights" / "hybrid_best_model.pth"

        self.logger = logging.getLogger(__name__)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Hybrid Trainer | {trainable:,} trainable params | "
                        f"LR: {self.base_lr} (main) / {self.base_lr*0.1} (CNN) | "
                        f"AMP: {self.use_amp}")

    def _create_data_loaders(self):
        train_path = Config.DATASET_PATH / "train"
        val_path = Config.DATASET_PATH / "val"

        train_dataset = CachedHybridDataset(train_path, is_training=True)
        val_dataset = CachedHybridDataset(val_path, is_training=False)

        # Weighted sampling
        class_counts = {}
        for _, class_idx, _ in train_dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        total = len(train_dataset.samples)
        weights = [total / (len(class_counts) * class_counts[ci]) for _, ci, _ in train_dataset.samples]
        sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)

        # Larger batch size since model is small
        batch_size = 128  # Can afford this with 2.7M model

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=sampler, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader, batch_size

    def train_epoch(self, train_loader, epoch, scheduler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (coords, crops, targets) in enumerate(pbar):
            coords = coords.to(self.device)
            crops = crops.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(coords, crops)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(coords, crops)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.1f}%',
                'LR': f'{current_lr:.6f}'
            })

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

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

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        per_class = {}
        for i, emotion in enumerate(Config.EMOTION_CLASSES):
            mask = np.array(all_targets) == i
            if mask.sum() > 0:
                per_class[emotion] = 100. * (np.array(all_preds)[mask] == i).sum() / mask.sum()
            else:
                per_class[emotion] = 0.0

        return epoch_loss, epoch_acc, all_preds, all_targets, per_class

    def train(self, num_epochs: int = 300):
        self.logger.info("Creating cached data loaders...")
        train_loader, val_loader, batch_size = self._create_data_loaders()

        # OneCycleLR: proper warmup + cosine decay over all epochs
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.base_lr * 0.1, self.base_lr],  # CNN max, Other max
            total_steps=total_steps,
            pct_start=0.05,       # 5% warmup
            anneal_strategy='cos',
            div_factor=10,        # start_lr = max_lr / 10
            final_div_factor=100  # end_lr = max_lr / 1000
        )

        self.logger.info(f"Training for {num_epochs} epochs | Batch size: {batch_size}")
        self.logger.info(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
        self.logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
        self.logger.info(f"OneCycleLR: warmup 5%, max_lr=[{self.base_lr*0.1}, {self.base_lr}]")

        # Setup experiment directory
        exp_dir = Config.MODELS_PATH / "experiments" / f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'per_class': []}

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch, scheduler)
            val_loss, val_acc, preds, targets, per_class = self.validate(val_loader)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['per_class'].append(per_class)

            # Logging
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            self.logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
            self.logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.1f}%")
            self.logger.info(f"  Gap: {train_acc - val_acc:.1f}%")

            for emotion, acc in per_class.items():
                status = "[OK]" if acc >= 80 else "[IMPROVING]" if acc >= 60 else "[LOW]"
                self.logger.info(f"    {emotion}: {acc:.1f}% {status}")

            # Classification report every 10 epochs
            if (epoch + 1) % 10 == 0:
                try:
                    report = classification_report(
                        targets, preds, target_names=Config.EMOTION_CLASSES, zero_division=0
                    )
                    self.logger.info(f"\nClassification Report (Epoch {epoch+1}):\n{report}")
                except Exception as e:
                    self.logger.warning(f"Report error: {e}")

            # Save best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'per_class_acc': per_class,
                    'config': {
                        'num_classes': Config.NUM_CLASSES,
                        'coordinate_dim': Config.COORDINATE_DIM
                    }
                }, self.best_model_path)

                self.logger.info(f"  ** New best! Val Acc: {val_acc:.1f}% **")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save history
        with open(exp_dir / "training_history.json", 'w') as f:
            json.dump(history, f, default=str)

        print("\n" + "=" * 60)
        print("HYBRID TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Val Accuracy: {self.best_val_acc:.1f}%")
        print(f"Model saved: {self.best_model_path}")
        print(f"Experiment: {exp_dir}")
        print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Hybrid CNN+Coordinate Model v2')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    Config.create_directories()

    print("=" * 60)
    print("HYBRID CNN + COORDINATE MODEL TRAINING v2")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Model: HybridEmotionNet (MobileNetV3 + Coordinate MLP)")
    print(f"LR: 0.001 (main) / 0.0001 (CNN)")
    print(f"Scheduler: OneCycleLR with 5% warmup")
    print(f"Classes: {Config.EMOTION_CLASSES}")
    print("=" * 60)

    trainer = HybridTrainer()
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
