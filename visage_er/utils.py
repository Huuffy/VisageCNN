"""
Enhanced Utilities for VisageCNN - RTX 3050 + Windows Optimized
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from pathlib import Path
import json
from datetime import datetime
import cv2
from typing import Dict, List, Tuple, Optional
import sqlite3
import pickle
from .config import Config

def setup_logging():
    """Setup enhanced logging configuration with Windows Unicode support"""
    Config.LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    log_file = Config.LOGS_PATH / f"visagecnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create handlers with UTF-8 encoding for Windows compatibility
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    except Exception:
        # Fallback for systems without UTF-8 support
        file_handler = logging.FileHandler(log_file)
    
    console_handler = logging.StreamHandler()
    
    # Set formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        handlers=[file_handler, console_handler],
        force=True  # Override existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Enhanced logging system initialized")
    return logger

def create_project_structure():
    """Create enhanced project directory structure"""
    Config.create_directories()
    
    # Create additional enhanced directories
    additional_dirs = [
        Config.MODELS_PATH / "experiments",
        Config.LOGS_PATH / "tensorboard",
        Config.LOGS_PATH / "training",
        Config.LOGS_PATH / "inference"
    ]
    
    for directory in additional_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("[OK] Enhanced project structure created successfully")

def print_system_info():
    """Print enhanced system information"""
    device_info = Config.get_device_info()
    
    print("\n" + "=" * 70)
    print("ENHANCED SYSTEM INFORMATION")
    print("=" * 70)
    
    if device_info['device'] == 'cuda':
        print(f"GPU: {device_info['device_name']}")
        print(f"Total Memory: {device_info['memory_total']} GB")
        print(f"Allocated Memory: {device_info['memory_allocated']} GB")
        print(f"Compute Capability: {device_info['compute_capability']}")
    else:
        print(f"CPU Cores: {device_info['cores']}")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    
    print("=" * 70 + "\n")

class DatasetAnalyzer:
    """Enhanced dataset analysis utilities"""
    
    @staticmethod
    def analyze_dataset_distribution(dataset_path: Path) -> Dict:
        """Analyze enhanced dataset distribution with proper file detection"""
        distribution = {}
        total_samples = 0
        
        # Common image file extensions (case-insensitive)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        
        print(f"[INFO] Scanning dataset path: {dataset_path}")
        
        # Check if dataset path exists
        if not dataset_path.exists():
            print(f"[ERROR] Dataset path does not exist: {dataset_path}")
            return {
                'distribution': {},
                'percentages': {},
                'total_samples': 0,
                'num_classes': 0
            }
        
        # Initialize distribution for all emotion classes
        for emotion in Config.EMOTION_CLASSES:
            distribution[emotion] = 0
        
        # Scan each emotion directory
        for emotion_dir in dataset_path.iterdir():
            if emotion_dir.is_dir():
                emotion_name = emotion_dir.name.lower()
                
                # Check if this is a valid emotion directory
                if emotion_name in [e.lower() for e in Config.EMOTION_CLASSES]:
                    # Find the correct case-sensitive emotion name
                    correct_emotion_name = None
                    for emotion in Config.EMOTION_CLASSES:
                        if emotion.lower() == emotion_name:
                            correct_emotion_name = emotion
                            break
                    
                    print(f"  [INFO] Checking {emotion_dir.name} directory...")
                    
                    # Count image files with proper extensions
                    image_files = []
                    
                    # Check all files in the directory
                    for file_path in emotion_dir.iterdir():
                        if file_path.is_file():
                            file_ext = file_path.suffix.lower()
                            if file_ext in image_extensions:
                                image_files.append(file_path)
                    
                    count = len(image_files)
                    if correct_emotion_name:
                        distribution[correct_emotion_name] = count
                        total_samples += count
                    
                    print(f"    [OK] Found {count} images in {emotion_dir.name}")
                    
                    # Debug: Show first few files found
                    if count > 0:
                        sample_files = [f.name for f in image_files[:3]]
                        print(f"    [FILES] Sample files: {sample_files}")
                    elif count == 0:
                        # List all files to help debug
                        all_files = list(emotion_dir.iterdir())
                        if all_files:
                            print(f"    [WARN] No images found, but directory contains {len(all_files)} files:")
                            for i, f in enumerate(all_files[:5]):  # Show first 5 files
                                print(f"      - {f.name} (extension: {f.suffix})")
                            if len(all_files) > 5:
                                print(f"      ... and {len(all_files) - 5} more files")
                        else:
                            print(f"    [WARN] Directory is empty")
                else:
                    print(f"  [SKIP] Skipping unknown directory: {emotion_dir.name}")
        
        # Calculate percentages (avoid division by zero)
        if total_samples > 0:
            percentages = {emotion: (count/total_samples)*100 
                          for emotion, count in distribution.items()}
        else:
            percentages = {emotion: 0.0 for emotion in Config.EMOTION_CLASSES}
        
        print(f"\n[SUMMARY] Dataset analysis complete:")
        print(f"  Total samples found: {total_samples}")
        print(f"  Classes with data: {len([k for k, v in distribution.items() if v > 0])}")
        
        return {
            'distribution': distribution,
            'percentages': percentages,
            'total_samples': total_samples,
            'num_classes': len([k for k, v in distribution.items() if v > 0])
        }
    
    @staticmethod
    def plot_enhanced_distribution(analysis_result: Dict, save_path: Optional[Path] = None):
        """Plot enhanced dataset distribution"""
        if analysis_result['total_samples'] == 0:
            print("[WARN] Cannot plot distribution - no data found")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filter out emotions with zero samples for cleaner plots
        non_zero_emotions = {k: v for k, v in analysis_result['distribution'].items() if v > 0}
        
        if not non_zero_emotions:
            print("[WARN] No emotions with data to plot")
            return
        
        emotions = list(non_zero_emotions.keys())
        counts = list(non_zero_emotions.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
        
        # Bar plot
        bars = ax1.bar(emotions, counts, color=colors)
        ax1.set_title('Dataset Distribution - Sample Counts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotions')
        ax1.set_ylabel('Sample Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Pie chart
        percentages = [analysis_result['percentages'][emotion] for emotion in emotions]
        ax2.pie(percentages, labels=emotions, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Dataset Distribution - Percentages', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class ModelEvaluator:
    """Enhanced model evaluation utilities"""
    
    @staticmethod
    def evaluate_enhanced_model(model, data_loader, device=Config.DEVICE):
        """Comprehensive enhanced model evaluation"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
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
            'probabilities': np.array(all_probabilities)
        }
    
    @staticmethod
    def calculate_enhanced_metrics(predictions, targets):
        """Calculate enhanced evaluation metrics"""
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, average=None)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, emotion in enumerate(Config.EMOTION_CLASSES):
            per_class_metrics[emotion] = {
                'precision': precision[i] if i < len(precision) else 0,
                'recall': recall[i] if i < len(recall) else 0,
                'f1_score': f1[i] if i < len(f1) else 0,
                'support': support[i] if i < len(support) else 0
            }
        
        # Average metrics
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        return {
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'per_class_metrics': per_class_metrics
        }
    
    @staticmethod
    def plot_enhanced_confusion_matrix(targets, predictions, save_path: Optional[Path] = None):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=Config.EMOTION_CLASSES,
                   yticklabels=Config.EMOTION_CLASSES)
        plt.title('Enhanced Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('Actual Emotion')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class Visualizer:
    """Enhanced visualization utilities"""
    
    @staticmethod
    def plot_enhanced_training_history(history: Dict, save_path: Optional[Path] = None):
        """Plot enhanced training history with multiple metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0, 0].plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rates' in history:
            axes[1, 0].plot(history['epochs'], history['learning_rates'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss difference plot
        if len(history['train_loss']) == len(history['val_loss']):
            loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
            axes[1, 1].plot(history['epochs'], loss_diff, 'm-', linewidth=2)
            axes[1, 1].set_title('Validation - Training Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class DatabaseManager:
    """Enhanced database management for experiments"""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Config.LOGS_PATH / "experiments.db"
        
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create enhanced experiment tracking tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Experiments table
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
                
                # Metrics table
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
            logging.warning(f"Failed to create database tables: {e}")
    
    def log_experiment(self, experiment_data: Dict) -> int:
        """Log enhanced experiment data"""
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
                    experiment_data.get('notes')
                ))
                return cursor.lastrowid
        except Exception as e:
            logging.warning(f"Failed to log experiment: {e}")
            return -1

def save_enhanced_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, filepath):
    """Save enhanced model checkpoint"""
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
            'dropout_rate': Config.DROPOUT_RATE
        }
    }
    
    torch.save(checkpoint, filepath)
    logging.info(f"[CHECKPOINT] Enhanced checkpoint saved: {filepath}")

def load_enhanced_checkpoint(filepath, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """Load enhanced model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

# Initialize enhanced utilities
logger = setup_logging()
