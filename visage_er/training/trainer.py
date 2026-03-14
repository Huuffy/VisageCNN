import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

from ..config import Config
from ..utils import setup_logging
from ..models.enhanced_model import (
    create_enhanced_model,
    EnhancedModelUtils,
    AdaptiveFocalLoss,
    UncertaintyLoss
)
from ..data.processor import create_enhanced_data_loaders
from ..core.face_processor import EnhancedFaceMeshProcessor

def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class EnhancedEmotionTrainer:
    """Enhanced training pipeline with AMP, label smoothing, gradient accumulation, and per-class metrics"""

    def __init__(self, use_focal_loss=True, use_mixup=True):
        self.device = Config.DEVICE

        # Initialize enhanced model
        self.model = create_enhanced_model()
        self.model_utils = EnhancedModelUtils

        # Enhanced loss functions
        self.use_focal_loss = use_focal_loss
        self.use_mixup = use_mixup

        # Class weights to boost weaker classes (Sad, Fear)
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(self.device)

        if use_focal_loss:
            self.criterion = AdaptiveFocalLoss(
                num_classes=Config.NUM_CLASSES,
                alpha=Config.FOCAL_LOSS_ALPHA,
                gamma=Config.FOCAL_LOSS_GAMMA
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=Config.LABEL_SMOOTHING
            )

        # Label smoothing cross entropy for validation (always clean)
        self.val_criterion = nn.CrossEntropyLoss()

        # Enhanced optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.BASE_LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        # Advanced scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )

        # Mixed precision training (AMP)
        self.use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Gradient accumulation
        self.grad_accum_steps = Config.GRADIENT_ACCUMULATION_STEPS

        # Training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epochs': [],
            'per_class_acc': []
        }

        # Enhanced tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_path = Config.MODELS_PATH / "enhanced_best_model.pth"

        # Setup logging
        self._setup_logging()

        if self.use_amp:
            self.logger.info("Mixed precision training (AMP) ENABLED")
        self.logger.info(f"Gradient accumulation steps: {self.grad_accum_steps}")

    def _setup_logging(self):
        """Setup enhanced logging configuration"""
        log_file = Config.LOGS_PATH / f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging()
        self.logger = logging.getLogger(__name__)

        # Add file handler for training-specific logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def train_epoch(self, train_loader, epoch):
        """Enhanced training with AMP, gradient accumulation, and mixup"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Update learning rate dynamically
        current_lr = Config.update_learning_rate(self.optimizer, epoch)

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{Config.EPOCHS}")

        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Apply mixup data augmentation (60% of batches to reduce underfitting)
            if self.use_mixup and random.random() < 0.6:
                data, targets_a, targets_b, lam = mixup_data(data, target, Config.MIXUP_ALPHA)

                # Forward pass with AMP
                if self.use_amp:
                    with autocast('cuda'):
                        output = self.model(data)
                        loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                        loss = loss / self.grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    output = self.model(data)
                    loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                    loss = loss / self.grad_accum_steps
                    loss.backward()
            else:
                # Forward pass with AMP
                if self.use_amp:
                    with autocast('cuda'):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss = loss / self.grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.grad_accum_steps
                    loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRADIENT_CLIP_VALUE)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRADIENT_CLIP_VALUE)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Statistics (use unscaled loss for logging)
            total_loss += loss.item() * self.grad_accum_steps
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.grad_accum_steps:.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, current_lr

    def validate(self, val_loader):
        """Enhanced validation with per-class accuracy and confusion matrix"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)

                if self.use_amp:
                    with autocast('cuda'):
                        output = self.model(data)
                        loss = self.val_criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.val_criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        # Calculate per-class accuracy
        per_class_acc = {}
        for class_idx, emotion in enumerate(Config.EMOTION_CLASSES):
            class_mask = np.array(all_targets) == class_idx
            if class_mask.sum() > 0:
                class_correct = (np.array(all_predictions)[class_mask] == class_idx).sum()
                class_acc = 100. * class_correct / class_mask.sum()
                per_class_acc[emotion] = class_acc
            else:
                per_class_acc[emotion] = 0.0

        return epoch_loss, epoch_acc, all_predictions, all_targets, per_class_acc

    def train(self, num_epochs=None):
        """Main enhanced training loop with all improvements"""
        if num_epochs is None:
            num_epochs = Config.EPOCHS

        self.logger.info("Starting enhanced training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model_utils.count_parameters(self.model):,}")
        self.logger.info(f"Mixed Precision: {self.use_amp}")
        self.logger.info(f"Gradient Accumulation: {self.grad_accum_steps} steps")
        self.logger.info(f"Effective Batch Size: {Config.BATCH_SIZE * self.grad_accum_steps}")

        # Create enhanced data loaders
        train_loader, val_loader = create_enhanced_data_loaders(
            use_weighted_sampling=True,
            cache_coordinates=True
        )

        # Enhanced training loop
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training phase
            train_loss, train_acc, current_lr = self.train_epoch(train_loader, epoch)

            # Validation phase
            val_loss, val_acc, val_predictions, val_targets, per_class_acc = self.validate(val_loader)

            # Save history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            self.train_history['epochs'].append(epoch + 1)
            self.train_history['per_class_acc'].append(per_class_acc)

            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Train-Val Gap: {train_acc - val_acc:.2f}% (overfitting if >15%)")

            # Log per-class accuracy
            self.logger.info("Per-class accuracy:")
            for emotion, acc in per_class_acc.items():
                status = "OK" if acc > 30 else "LOW"
                self.logger.info(f"  {emotion}: {acc:.1f}% [{status}]")

            # Log confusion matrix every 10 epochs
            if (epoch + 1) % 10 == 0:
                try:
                    cm = confusion_matrix(val_targets, val_predictions)
                    report = classification_report(
                        val_targets, val_predictions,
                        target_names=Config.EMOTION_CLASSES,
                        zero_division=0
                    )
                    self.logger.info(f"\nClassification Report (Epoch {epoch+1}):\n{report}")
                except Exception as e:
                    self.logger.warning(f"Could not generate classification report: {e}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                self.model_utils.save_enhanced_model(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_loss, {'val_acc': val_acc, 'per_class_acc': per_class_acc},
                    self.best_model_path
                )

                self.logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.logger.info("Enhanced training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

def main():
    """Main enhanced training function"""
    Config.create_directories()

    trainer = EnhancedEmotionTrainer(
        use_focal_loss=True,
        use_mixup=True
    )

    trainer.train()

if __name__ == "__main__":
    main()
