"""Shared utilities for VisageCNN — logging, dataset analysis, model evaluation,
visualisation, and experiment database management.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from pathlib import Path
import json
from datetime import datetime
import cv2
from typing import Dict, List, Optional
import sqlite3
import pickle
from .config import Config


def setup_logging() -> logging.Logger:
    """Initialise file and console logging with UTF-8 support.

    Returns:
        Root logger configured with both handlers.
    """
    Config.LOGS_PATH.mkdir(parents=True, exist_ok=True)

    log_file = Config.LOGS_PATH / f"visagecnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    except Exception:
        file_handler = logging.FileHandler(log_file)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(Config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=Config.LOG_LEVEL,
        handlers=[file_handler, console_handler],
        force=True,
    )

    return logging.getLogger(__name__)


def create_project_structure():
    """Create all required project directories."""
    Config.create_directories()
    additional = [
        Config.MODELS_PATH / "experiments",
        Config.LOGS_PATH / "tensorboard",
        Config.LOGS_PATH / "training",
        Config.LOGS_PATH / "inference",
    ]
    for directory in additional:
        directory.mkdir(parents=True, exist_ok=True)


def print_system_info():
    """Print hardware and framework information to stdout."""
    device_info = Config.get_device_info()

    print("\n" + "=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)

    if device_info['device'] == 'cuda':
        print(f"GPU:               {device_info['device_name']}")
        print(f"Total VRAM:        {device_info['memory_total']} GB")
        print(f"Allocated VRAM:    {device_info['memory_allocated']} GB")
        print(f"Compute:           SM {device_info['compute_capability']}.x")
    else:
        print(f"CPU Cores:         {device_info['cores']}")

    print(f"PyTorch:           {torch.__version__}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version:      {torch.version.cuda}")

    print("=" * 70 + "\n")


class DatasetAnalyzer:
    """Utilities for analysing emotion dataset distribution and balance."""

    @staticmethod
    def analyze_dataset_distribution(dataset_path: Path) -> Dict:
        """Count images per emotion class and compute per-class percentages.

        Args:
            dataset_path: Path to a directory containing one subdirectory per class.

        Returns:
            Dictionary with keys 'distribution', 'percentages', 'total_samples',
            and 'num_classes'.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        distribution = {emotion: 0 for emotion in Config.EMOTION_CLASSES}
        total_samples = 0

        if not dataset_path.exists():
            return {
                'distribution': distribution,
                'percentages': {e: 0.0 for e in Config.EMOTION_CLASSES},
                'total_samples': 0,
                'num_classes': 0,
            }

        for emotion_dir in dataset_path.iterdir():
            if not emotion_dir.is_dir():
                continue

            matched = next(
                (e for e in Config.EMOTION_CLASSES if e.lower() == emotion_dir.name.lower()),
                None,
            )
            if matched is None:
                continue

            count = sum(
                1 for f in emotion_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            distribution[matched] = count
            total_samples += count

        percentages = (
            {e: (c / total_samples) * 100 for e, c in distribution.items()}
            if total_samples > 0
            else {e: 0.0 for e in Config.EMOTION_CLASSES}
        )

        return {
            'distribution': distribution,
            'percentages': percentages,
            'total_samples': total_samples,
            'num_classes': sum(1 for v in distribution.values() if v > 0),
        }

    @staticmethod
    def plot_distribution(analysis_result: Dict, save_path: Optional[Path] = None):
        """Plot per-class sample counts and percentage breakdown.

        Args:
            analysis_result: Output of :meth:`analyze_dataset_distribution`.
            save_path: Optional path to save the figure as a PNG.
        """
        if analysis_result['total_samples'] == 0:
            return

        non_zero = {k: v for k, v in analysis_result['distribution'].items() if v > 0}
        if not non_zero:
            return

        emotions = list(non_zero.keys())
        counts = list(non_zero.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bars = ax1.bar(emotions, counts, color=colors)
        ax1.set_title('Sample Counts per Class', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotion')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(int(bar.get_height())), ha='center', va='bottom',
            )

        percentages = [analysis_result['percentages'][e] for e in emotions]
        ax2.pie(percentages, labels=emotions, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelEvaluator:
    """Utilities for evaluating a trained model on a DataLoader."""

    @staticmethod
    def evaluate(model, data_loader, device=Config.DEVICE) -> Dict:
        """Run model inference over an entire DataLoader and collect predictions.

        Args:
            model: Trained PyTorch model.
            data_loader: DataLoader yielding (inputs, targets) batches.
            device: Target device for inference.

        Returns:
            Dictionary with keys 'predictions', 'targets', and 'probabilities'.
        """
        model.eval()
        all_predictions, all_targets, all_probabilities = [], [], []

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities),
        }

    @staticmethod
    def calculate_metrics(predictions, targets) -> Dict:
        """Compute accuracy, precision, recall, F1, and per-class breakdowns.

        Args:
            predictions: Array of predicted class indices.
            targets: Array of ground-truth class indices.

        Returns:
            Dictionary with overall and per-class metrics.
        """
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None
        )

        per_class = {}
        for i, emotion in enumerate(Config.EMOTION_CLASSES):
            per_class[emotion] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0,
            }

        return {
            'accuracy': accuracy,
            'avg_precision': float(np.mean(precision)),
            'avg_recall': float(np.mean(recall)),
            'avg_f1_score': float(np.mean(f1)),
            'per_class_metrics': per_class,
        }

    @staticmethod
    def plot_confusion_matrix(targets, predictions, save_path: Optional[Path] = None):
        """Plot a confusion matrix heatmap.

        Args:
            targets: Ground-truth class indices.
            predictions: Predicted class indices.
            save_path: Optional path to save the figure.
        """
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=Config.EMOTION_CLASSES,
            yticklabels=Config.EMOTION_CLASSES,
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class Visualizer:
    """Utilities for visualising training history."""

    @staticmethod
    def plot_training_history(history: Dict, save_path: Optional[Path] = None):
        """Plot loss, accuracy, learning rate, and validation gap curves.

        Args:
            history: Dictionary containing lists for 'epochs', 'train_loss',
                'val_loss', 'train_acc', 'val_acc', and optionally 'learning_rates'.
            save_path: Optional path to save the figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')

        axes[0, 0].plot(history['epochs'], history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(history['epochs'], history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history['epochs'], history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(history['epochs'], history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if 'learning_rates' in history:
            axes[1, 0].plot(history['epochs'], history['learning_rates'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        if len(history['train_loss']) == len(history['val_loss']):
            gap = np.array(history['val_loss']) - np.array(history['train_loss'])
            axes[1, 1].plot(history['epochs'], gap, 'm-', linewidth=2)
            axes[1, 1].set_title('Val − Train Loss (Generalisation Gap)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DatabaseManager:
    """SQLite-backed experiment tracker for training runs."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialise the database, creating tables if they do not exist.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ``Config.LOGS_PATH / "experiments.db"``.
        """
        self.db_path = db_path or (Config.LOGS_PATH / "experiments.db")
        self._create_tables()

    def _create_tables(self):
        """Create experiments and metrics tables if they do not already exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS experiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        model_type TEXT,
                        dataset_size INTEGER,
                        batch_size INTEGER,
                        learning_rate REAL,
                        epochs INTEGER,
                        best_val_acc REAL,
                        best_val_loss REAL,
                        training_time REAL,
                        model_path TEXT,
                        notes TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id INTEGER,
                        epoch INTEGER,
                        train_loss REAL,
                        train_acc REAL,
                        val_loss REAL,
                        val_acc REAL,
                        learning_rate REAL,
                        FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                    )
                ''')
        except Exception as e:
            logging.warning(f"Failed to initialise experiment database: {e}")

    def log_experiment(self, experiment_data: Dict) -> int:
        """Insert an experiment record and return its row ID.

        Args:
            experiment_data: Dictionary with keys matching the experiments table columns.

        Returns:
            Row ID of the inserted record, or -1 on failure.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiments (
                        model_type, dataset_size, batch_size, learning_rate,
                        epochs, best_val_acc, best_val_loss, training_time,
                        model_path, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experiment_data.get('model_type'),
                    experiment_data.get('dataset_size'),
                    experiment_data.get('batch_size'),
                    experiment_data.get('learning_rate'),
                    experiment_data.get('epochs'),
                    experiment_data.get('best_val_acc'),
                    experiment_data.get('best_val_loss'),
                    experiment_data.get('training_time'),
                    experiment_data.get('model_path'),
                    experiment_data.get('notes'),
                ))
                return cursor.lastrowid
        except Exception as e:
            logging.warning(f"Failed to log experiment: {e}")
            return -1


def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, filepath):
    """Serialise a training checkpoint to disk.

    Args:
        model: PyTorch model instance.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance, or None.
        epoch: Current epoch number.
        loss: Scalar loss value for this checkpoint.
        metrics: Dictionary of evaluation metrics.
        filepath: Destination path for the checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_classes': Config.NUM_CLASSES,
            'hidden_size': Config.HIDDEN_SIZE,
            'num_heads': Config.NUM_HEADS,
            'dropout_rate': Config.DROPOUT_RATE,
        },
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """Load a serialised checkpoint and restore model (and optionally optimiser) state.

    Args:
        filepath: Path to the checkpoint file.
        model: Model instance to restore weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional LR scheduler to restore state.
        device: Device to map tensors onto.

    Returns:
        The full checkpoint dictionary.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint
