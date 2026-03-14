"""VisageCNN — real-time facial expression recognition using a hybrid CNN and landmark architecture."""

__version__ = "3.0.0"
__author__ = "Huuffy"

from .config import Config
from .models.hybrid_model import HybridEmotionNet, create_hybrid_model
from .utils import (
    DatasetAnalyzer,
    ModelEvaluator,
    DatabaseManager,
    Visualizer,
    setup_logging,
    create_project_structure,
    print_system_info,
)

__all__ = [
    "Config",
    "HybridEmotionNet",
    "create_hybrid_model",
    "DatasetAnalyzer",
    "ModelEvaluator",
    "DatabaseManager",
    "Visualizer",
    "setup_logging",
    "create_project_structure",
    "print_system_info",
]
