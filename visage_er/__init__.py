"""
VisageCNN - Enhanced PyTorch-based facial expression recognition system
"""

__version__ = "2.2.0"
__author__ = "VisageCNN Team"
__description__ = "Real-time facial expression recognition using enhanced PyTorch components"

# Core imports
from .config import Config

# Enhanced component imports (Primary)
from .models.enhanced_model import (
    EnhancedCoordinateEmotionNet,
    create_enhanced_model,
    EnhancedModelUtils,
    AdaptiveFocalLoss,
    UncertaintyLoss
)
from .data.processor import (
    AdvancedEmotionDataset,
    create_enhanced_data_loaders,
    AdvancedCoordinateAugmentation,
    DatasetPerformanceMonitor
)
from .core.face_processor import (
    EnhancedFaceMeshProcessor
    # Removed FaceQualityAssessment - not implemented yet
)

# Training and inference
from .training.trainer import EnhancedEmotionTrainer

# Utilities
from .utils import (
    DatasetAnalyzer,
    ModelEvaluator,
    DatabaseManager,
    Visualizer,
    setup_logging,
    create_project_structure,
    print_system_info
)

# Export all public classes and functions
__all__ = [
    # Core Configuration
    'Config',
    # Enhanced Model Components
    'EnhancedCoordinateEmotionNet',
    'create_enhanced_model',
    'EnhancedModelUtils',
    'AdaptiveFocalLoss',
    'UncertaintyLoss',
    # Enhanced Data Processing
    'AdvancedEmotionDataset',
    'create_enhanced_data_loaders',
    'AdvancedCoordinateAugmentation',
    'DatasetPerformanceMonitor',
    # Enhanced Face Processing
    'EnhancedFaceMeshProcessor',
    # Training
    'EnhancedEmotionTrainer',
    # Utilities
    'DatasetAnalyzer',
    'ModelEvaluator',
    'DatabaseManager',
    'Visualizer',
    'setup_logging',
    'create_project_structure',
    'print_system_info'
]

# Initialize logging
setup_logging()

def print_package_info():
    """Print enhanced package information"""
    print("=" * 70)
    print(f"VisageCNN v{__version__} - Enhanced Emotion Recognition")
    print("=" * 70)
    print("Enhanced Features:")
    print(" • Advanced multi-head attention mechanisms")
    print(" • Geometric feature extraction from facial landmarks")
    print(" • Emotion-specific expert networks")
    print(" • Advanced coordinate augmentation")
    print(" • Enhanced face processing and preprocessing")
    print(" • Weighted sampling for class balance")
    print(" • Coordinate caching for performance")
    print(" • Real-time inference optimization")
    print()
    print("Quick Start:")
    print(" 1. Create structure: python -c 'from visage_er import create_project_structure; create_project_structure()'")
    print(" 2. Train model: python train.py")
    print("=" * 70)

# Print info on import (only once)
if not hasattr(print_package_info, 'already_printed'):
    print_package_info()
    print_package_info.already_printed = True
