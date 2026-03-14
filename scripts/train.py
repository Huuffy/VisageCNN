#!/usr/bin/env python3
"""
Enhanced VisageCNN Training Script
Main entry point for training the enhanced emotion recognition model
"""

# Suppress noisy warnings before any imports
import warnings
import os
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["GLOG_minloglevel"] = "2"  # Suppress mediapipe C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF Lite logs

import sys
import argparse
import time
from pathlib import Path
import torch
from datetime import datetime
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visage_er.config import Config
from visage_er.training.trainer import EnhancedEmotionTrainer
from visage_er.utils import (
    setup_logging,
    create_project_structure,
    print_system_info,
    DatasetAnalyzer,
    DatabaseManager
)

def parse_arguments():
    """Parse enhanced command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced VisageCNN Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=Config.BASE_LEARNING_RATE,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=Config.WEIGHT_DECAY,
                       help='Weight decay for optimizer')

    # Enhanced model parameters
    parser.add_argument('--hidden-size', type=int, default=Config.HIDDEN_SIZE,
                       help='Hidden size for enhanced model')
    parser.add_argument('--num-heads', type=int, default=Config.NUM_HEADS,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=Config.NUM_LAYERS,
                       help='Number of transformer layers')
    parser.add_argument('--dropout-rate', type=float, default=Config.DROPOUT_RATE,
                       help='Dropout rate')

    # Enhanced training options
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use adaptive focal loss')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                       help='Use mixup data augmentation')
    parser.add_argument('--use-weighted-sampling', action='store_true', default=True,
                       help='Use weighted random sampling')
    parser.add_argument('--cache-coordinates', action='store_true', default=True,
                       help='Cache extracted coordinates')

    # Advanced options
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--notes', type=str, default='',
                       help='Notes about this experiment')

    # Data paths
    parser.add_argument('--train-data', type=str, default=str(Config.TRAIN_DATA_PATH),
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, default=str(Config.VAL_DATA_PATH),
                       help='Path to validation data')

    # Output options
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save training plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    return parser.parse_args()

def validate_dataset(train_path, val_path):
    """Validate enhanced dataset structure and content with improved error handling"""
    print("\n" + "="*50)
    print("ENHANCED DATASET VALIDATION")
    print("="*50)

    train_path = Path(train_path)
    val_path = Path(val_path)

    # Check if paths exist
    if not train_path.exists():
        print(f"[!!] Training data path not found: {train_path}")
        print("🔧 Please ensure the path exists and contains emotion subdirectories")
        raise FileNotFoundError(f"Training data path not found: {train_path}")

    if not val_path.exists():
        print(f"[!!] Validation data path not found: {val_path}")
        print("🔧 Please ensure the path exists and contains emotion subdirectories")
        raise FileNotFoundError(f"Validation data path not found: {val_path}")

    # Analyze training dataset
    print("📊 Analyzing training dataset...")
    try:
        train_analysis = DatasetAnalyzer.analyze_dataset_distribution(train_path)
        print(f"[OK] Training samples: {train_analysis['total_samples']}")
        print(f"[OK] Classes found: {train_analysis['num_classes']}")
    except Exception as e:
        print(f"[!!] Error analyzing training dataset: {e}")
        train_analysis = {'total_samples': 0, 'num_classes': 0, 'distribution': {}, 'percentages': {}}

    # Analyze validation dataset
    print("📊 Analyzing validation dataset...")
    try:
        val_analysis = DatasetAnalyzer.analyze_dataset_distribution(val_path)
        print(f"[OK] Validation samples: {val_analysis['total_samples']}")
        print(f"[OK] Classes found: {val_analysis['num_classes']}")
    except Exception as e:
        print(f"[!!] Error analyzing validation dataset: {e}")
        val_analysis = {'total_samples': 0, 'num_classes': 0, 'distribution': {}, 'percentages': {}}

    # Check if we have any data at all
    if train_analysis['total_samples'] == 0 and val_analysis['total_samples'] == 0:
        print("\n" + "="*50)
        print("[!!] CRITICAL ERROR: No images found in datasets!")
        print("="*50)
        print("\n🔧 Troubleshooting checklist:")
        print("1. Verify your dataset directory structure:")
        print("   ├── dataset/")
        print("   │   ├── train/")
        print("   │   │   ├── angry/*.jpg")
        print("   │   │   ├── disgust/*.jpg")
        print("   │   │   ├── fear/*.jpg")
        print("   │   │   ├── happy/*.jpg")
        print("   │   │   ├── neutral/*.jpg")
        print("   │   │   ├── sad/*.jpg")
        print("   │   │   └── surprised/*.jpg")
        print("   │   └── val/")
        print("   │       ├── angry/*.jpg")
        print("   │       ├── disgust/*.jpg")
        print("   │       └── ... (same structure)")
        print("\n2. Check supported image formats:")
        print("   - .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        print("\n3. Verify file permissions and directory access")
        print("\n4. Check emotion folder names match exactly:")
        print(f"   Expected: {', '.join(Config.EMOTION_CLASSES)}")
        print("\n5. Run this debug command:")
        print("   python -c \"from pathlib import Path; print(list(Path('dataset/train').iterdir()))\"")
        print("="*50)

        return None, None

    # Warning for insufficient data
    if train_analysis['total_samples'] < 100:
        print(f"\n[!] WARNING: Very few training samples ({train_analysis['total_samples']})")
        print("   Consider adding more data for better model performance")

    if val_analysis['total_samples'] < 20:
        print(f"\n[!] WARNING: Very few validation samples ({val_analysis['total_samples']})")
        print("   Consider adding more validation data for reliable evaluation")

    # Check class balance only if we have data
    if train_analysis['total_samples'] > 0:
        print("\n📈 Class Distribution:")
        class_imbalance = False
        for emotion in Config.EMOTION_CLASSES:
            train_count = train_analysis['distribution'].get(emotion, 0)
            val_count = val_analysis['distribution'].get(emotion, 0)
            train_pct = train_analysis['percentages'].get(emotion, 0)
            val_pct = val_analysis['percentages'].get(emotion, 0)

            status = "[OK]" if train_count > 0 else "[!!]"
            print(f"  {status} {emotion:>10}: Train: {train_count:>4} ({train_pct:>5.1f}%) | "
                  f"Val: {val_count:>4} ({val_pct:>5.1f}%)")

            if train_count == 0:
                class_imbalance = True

        # Warnings for class imbalance
        train_percentages = [v for v in train_analysis['percentages'].values() if v > 0]
        if len(train_percentages) > 1 and (max(train_percentages) - min(train_percentages) > 30):
            print("\n[!] WARNING: Significant class imbalance detected in training set")
            print("   Weighted sampling is enabled by default to help with this")
            class_imbalance = True

        if class_imbalance:
            print("\n💡 Tips for handling class imbalance:")
            print("   - Use weighted sampling (enabled by default)")
            print("   - Consider data augmentation for minority classes")
            print("   - Monitor per-class metrics during training")

    print("="*50)
    return train_analysis, val_analysis

def setup_experiment(args):
    """Setup enhanced experiment environment"""
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"enhanced_exp_{timestamp}"

    # Create experiment directory
    experiment_dir = Config.MODELS_PATH / "experiments" / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    exp_config = {
        'experiment_name': args.experiment_name,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_size': args.hidden_size,
            'num_heads': args.num_heads,
            'dropout_rate': args.dropout_rate,
            'use_focal_loss': args.use_focal_loss,
            'use_mixup': args.use_mixup,
            'device': str(Config.DEVICE)
        },
        'notes': args.notes
    }

    config_file = experiment_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(exp_config, f, indent=2)

    return experiment_dir, exp_config

def main():
    """Main enhanced training function"""
    # Parse arguments
    args = parse_arguments()

    # Print header
    print("=" * 70)
    print("ENHANCED VISAGECNN TRAINING PIPELINE")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name or 'Default'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Setup environment
    create_project_structure()
    setup_logging()
    print_system_info()

    # Validate dataset with enhanced error handling
    try:
        train_analysis, val_analysis = validate_dataset(args.train_data, args.val_data)

        # Exit early if no data found
        if train_analysis is None or val_analysis is None:
            print("\n[!!] Training cannot proceed without valid dataset")
            print("Please fix the dataset issues above and try again.")
            return 1

    except Exception as e:
        print(f"[!!] Dataset validation failed: {e}")
        print("\n🔧 Common solutions:")
        print("1. Check if dataset directories exist and are accessible")
        print("2. Verify image files have supported extensions (.jpg, .png, etc.)")
        print("3. Ensure emotion subdirectories are named correctly")
        return 1

    # Check minimum data requirements
    if train_analysis['total_samples'] < 10:
        print("[!!] Insufficient training data (minimum 10 samples required)")
        return 1

    if val_analysis['total_samples'] < 5:
        print("[!!] Insufficient validation data (minimum 5 samples required)")
        return 1

    # Setup experiment
    experiment_dir, exp_config = setup_experiment(args)

    # Update config with command line arguments
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.BASE_LEARNING_RATE = args.learning_rate
    Config.WEIGHT_DECAY = args.weight_decay
    Config.HIDDEN_SIZE = args.hidden_size
    Config.NUM_HEADS = args.num_heads
    Config.NUM_LAYERS = args.num_layers
    Config.DROPOUT_RATE = args.dropout_rate

    # Print enhanced configuration
    Config.print_config()

    try:
        # Initialize enhanced trainer
        print("\n🚀 Initializing Enhanced Trainer...")
        trainer = EnhancedEmotionTrainer(
            use_focal_loss=args.use_focal_loss,
            use_mixup=args.use_mixup
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            print(f"📂 Resuming from checkpoint: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=Config.DEVICE)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"[OK] Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"[!] Warning: Could not resume from checkpoint: {e}")
                print("Starting training from scratch...")

        # Log experiment to database
        try:
            db_manager = DatabaseManager()
            experiment_id = db_manager.log_experiment({
                'model_type': 'EnhancedCoordinateEmotionNet',
                'dataset_size': train_analysis['total_samples'],
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'model_path': str(experiment_dir),
                'notes': args.notes
            })
            print(f"📝 Experiment logged with ID: {experiment_id}")
        except Exception as e:
            print(f"[!] Warning: Could not log to database: {e}")

        # Start enhanced training
        print("\n🎯 Starting Enhanced Training...")
        start_time = time.time()

        trainer.train(num_epochs=args.epochs)

        training_time = time.time() - start_time

        # Save training history
        try:
            history_file = experiment_dir / "training_history.json"
            with open(history_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                history_json = {}
                for key, value in trainer.train_history.items():
                    if isinstance(value, list):
                        history_json[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                    else:
                        history_json[key] = value
                json.dump(history_json, f, indent=2)
            print(f"📊 Training history saved to: {history_file}")
        except Exception as e:
            print(f"[!] Warning: Could not save training history: {e}")

        # Generate enhanced training plots
        if args.save_plots and len(trainer.train_history['epochs']) > 0:
            try:
                from visage_er.utils import Visualizer
                plot_file = experiment_dir / "training_plots.png"
                Visualizer.plot_enhanced_training_history(trainer.train_history, plot_file)
                print(f"📈 Training plots saved to: {plot_file}")
            except Exception as e:
                print(f"[!] Warning: Could not generate plots: {e}")

        # Final summary
        print("\n" + "=" * 70)
        print("ENHANCED TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"[OK] Experiment: {args.experiment_name}")
        print(f"[OK] Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
        print(f"[OK] Best Validation Loss: {trainer.best_val_loss:.4f}")
        print(f"[OK] Total Training Time: {training_time/3600:.2f} hours")
        print(f"[OK] Best Model Saved: {trainer.best_model_path}")
        print(f"[OK] Experiment Directory: {experiment_dir}")
        print(f"[OK] Training Samples Used: {train_analysis['total_samples']}")
        print(f"[OK] Validation Samples Used: {val_analysis['total_samples']}")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (Ctrl+C)")
        print("Partial results may be saved in the experiment directory")
        return 1

    except Exception as e:
        print(f"\n[!!] Training failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check if you have enough GPU/CPU memory")
        print("2. Try reducing batch size if you get memory errors")
        print("3. Verify all required dependencies are installed")
        print("4. Check the full error traceback below:")
        print("-" * 50)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
