"""VisageCNN configuration — optimized for RTX 3050 Laptop GPU (4GB VRAM)."""

import torch
import os
from pathlib import Path
import logging


class Config:
    """Central configuration for the VisageCNN hybrid model pipeline."""

    PROJECT_ROOT = Path(__file__).parent.parent
    DATASET_PATH = PROJECT_ROOT / "dataset"
    MODELS_PATH = PROJECT_ROOT / "models"
    WEIGHTS_PATH = MODELS_PATH / "weights"
    SCALERS_PATH = MODELS_PATH / "scalers"
    CACHE_PATH = MODELS_PATH / "cache"
    LOGS_PATH = PROJECT_ROOT / "logs"
    CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"

    TRAIN_DATA_PATH = DATASET_PATH / "train"
    VAL_DATA_PATH = DATASET_PATH / "val"
    TRAIN_PATH = DATASET_PATH / "train"
    VAL_PATH = DATASET_PATH / "val"
    VALIDATION_PATH = DATASET_PATH / "val"

    NUM_CLASSES = 7
    EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

    NUM_LANDMARKS = 478
    COORDINATE_FEATURES = 3
    NUM_COORDINATE_FEATURES = 1434
    COORDINATE_DIM = 1434
    FEATURE_SIZE = 1434
    INPUT_SIZE = 1434

    USE_3D_COORDINATES = True
    EXTRACT_ONLY_XYZ = True
    NORMALIZE_COORDINATES = True
    HANDLE_VARIABLE_LANDMARKS = True

    NORMALIZE_Z_COORDINATES = True
    Z_COORDINATE_SCALE = 0.1
    COORDINATE_PREPROCESSING = True
    USE_COORDINATE_SMOOTHING = True
    APPLY_OUTLIER_REMOVAL = True

    Z_NORMALIZATION_METHOD = 'scale'
    Z_CLIP_RANGE = (-1.0, 1.0)
    Z_OUTLIER_THRESHOLD = 3.0

    SMOOTHING_WINDOW_SIZE = 5
    SMOOTHING_WEIGHT_DECAY = 0.9

    MIN_COORDINATE_QUALITY = 0.5
    MAX_COORDINATE_VARIANCE = 100.0

    USE_REFINED_LANDMARKS = True
    ENABLE_IRIS_LANDMARKS = True
    FACE_MESH_COMPLEXITY = 2
    MEDIAPIPE_MODEL_COMPLEXITY = 2
    MEDIAPIPE_REFINE_LANDMARKS = True
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

    BATCH_SIZE = 128
    GRADIENT_ACCUMULATION_STEPS = 2

    HIDDEN_SIZE = 640
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT_RATE = 0.3
    GEOMETRIC_FEATURE_DIM = 64
    EXPERT_HIDDEN_SIZE = 128
    HIDDEN_LAYERS = [640]
    DROPOUT_RATES = [0.3]

    ATTENTION_DIM = 256
    KEY_DIM = 32
    VALUE_DIM = 32

    EPOCHS = 300
    BASE_LEARNING_RATE = 0.0001
    MIN_LEARNING_RATE = 0.000005
    WEIGHT_DECAY = 0.05

    FOCAL_LOSS_ALPHA = 1.0
    FOCAL_LOSS_GAMMA = 2.0
    LABEL_SMOOTHING = 0.12
    MIXUP_ALPHA = 0.1

    EARLY_STOPPING_PATIENCE = 40
    GRADIENT_CLIP_VALUE = 1.0
    WARMUP_EPOCHS = 10

    CLASS_WEIGHTS = [1.0, 1.35, 1.4, 1.0, 1.0, 1.0, 1.15]

    AUGMENTATION_PROBABILITY = 0.7
    COORDINATE_NOISE_STD = 0.02
    GEOMETRIC_AUGMENTATION_STRENGTH = 0.1
    NOISE_LIMIT = 0.02
    ROTATION_LIMIT = 15
    AUGMENTATION_STRENGTH = 0.1
    ROTATION_RANGE = 15
    SCALE_RANGE = (0.9, 1.1)
    TRANSLATION_RANGE = 0.05
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    FLIP_PROBABILITY = 0.5

    FACE_CONFIDENCE_THRESHOLD = 0.7
    FACE_QUALITY_THRESHOLD = 0.6
    COORDINATE_SMOOTHING_ALPHA = 0.3
    MIN_FACE_SIZE = 50
    MAX_FACE_SIZE = 500

    NUM_WORKERS = 4
    PIN_MEMORY = True
    CACHE_COORDINATES = True
    COORDINATE_CACHE_SIZE = 24000
    PREFETCH_FACTOR = 4
    PERSISTENT_WORKERS = False
    DROP_LAST = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MAX_MEMORY_USAGE = 0.85
    MIXED_PRECISION = True
    CUDA_MEMORY_FRACTION = 1.0
    ALLOW_MEMORY_GROWTH = True
    EMPTY_CACHE_FREQUENCY = 50

    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    SAVE_TOP_K = 3
    SAVE_EVERY_N_EPOCHS = 10

    FACE_CROP_SIZE = 224
    IMAGE_SIZE = (224, 224)
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]

    COORDINATE_NORMALIZATION = True
    COORDINATE_SCALE_FACTOR = 1.0

    USE_RESIDUAL_CONNECTIONS = True
    USE_BATCH_NORMALIZATION = True
    USE_LAYER_NORMALIZATION = True

    NUM_EXPERTS = 7
    EXPERT_ACTIVATION = 'relu'
    EXPERT_USE_DROPOUT = True

    VALIDATION_FREQUENCY = 1
    SAVE_FREQUENCY = 5

    COORDINATE_MIN_VALUE = -1000
    COORDINATE_MAX_VALUE = 1000
    OUTLIER_THRESHOLD = 3.0

    MIN_LANDMARK_CONFIDENCE = 0.5
    MAX_FACE_ANGLE = 45
    MIN_FACE_AREA = 2500

    ENABLE_PROFILING = True
    LOG_MEMORY_USAGE = True
    PROFILE_EVERY_N_BATCHES = 100

    GEOMETRIC_FEATURES = [
        'eye_aspect_ratio',
        'mouth_aspect_ratio',
        'eyebrow_height',
        'face_symmetry',
        'facial_angles',
    ]

    QUALITY_CHECKS = [
        'face_size',
        'face_angle',
        'lighting_condition',
        'blur_detection',
        'occlusion_detection',
    ]

    @classmethod
    def create_directories(cls):
        """Create all required project directories if they do not exist."""
        directories = [
            cls.MODELS_PATH,
            cls.LOGS_PATH,
            cls.CHECKPOINTS_PATH,
            cls.DATASET_PATH,
            cls.TRAIN_DATA_PATH,
            cls.VAL_DATA_PATH,
            cls.MODELS_PATH / "experiments",
            cls.LOGS_PATH / "training",
            cls.LOGS_PATH / "inference",
            cls.WEIGHTS_PATH,
            cls.SCALERS_PATH,
            cls.CACHE_PATH,
            cls.CACHE_PATH / "train",
            cls.CACHE_PATH / "val",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def optimize_for_rtx3050(cls):
        """Apply RTX 3050 4GB GPU-specific CUDA optimizations."""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cls.CUDA_MEMORY_FRACTION)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)

    @classmethod
    def get_windows_safe_config(cls):
        """Return DataLoader kwargs compatible with Windows multiprocessing constraints."""
        return {
            'num_workers': cls.NUM_WORKERS,
            'pin_memory': cls.PIN_MEMORY,
            'persistent_workers': cls.PERSISTENT_WORKERS,
            'multiprocessing_context': None,
            'timeout': 0,
        }

    @classmethod
    def get_device_info(cls):
        """Return a dictionary with hardware device information."""
        if torch.cuda.is_available():
            return {
                'device': 'cuda',
                'device_name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory // (1024 ** 3),
                'memory_allocated': torch.cuda.memory_allocated(0) // (1024 ** 3),
                'compute_capability': torch.cuda.get_device_properties(0).major,
            }
        return {'device': 'cpu', 'cores': os.cpu_count()}

    @classmethod
    def update_learning_rate(cls, optimizer, epoch):
        """Apply cosine-annealed learning rate with linear warmup.

        Args:
            optimizer: PyTorch optimizer instance.
            epoch: Current training epoch (0-indexed).

        Returns:
            The learning rate applied for this epoch.
        """
        if epoch < cls.WARMUP_EPOCHS:
            lr = cls.BASE_LEARNING_RATE * (epoch + 1) / cls.WARMUP_EPOCHS
        else:
            import math
            T_cur = epoch - cls.WARMUP_EPOCHS
            T_max = cls.EPOCHS - cls.WARMUP_EPOCHS
            lr = cls.MIN_LEARNING_RATE + (cls.BASE_LEARNING_RATE - cls.MIN_LEARNING_RATE) * \
                 (1 + math.cos(math.pi * T_cur / T_max)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    @classmethod
    def get_coordinate_config(cls):
        """Return a dictionary of coordinate processing configuration values."""
        return {
            'num_landmarks': cls.NUM_LANDMARKS,
            'coordinate_features': cls.COORDINATE_FEATURES,
            'total_features': cls.NUM_COORDINATE_FEATURES,
            'use_3d': cls.USE_3D_COORDINATES,
            'normalize': cls.NORMALIZE_COORDINATES,
            'normalize_z': cls.NORMALIZE_Z_COORDINATES,
            'z_scale': cls.Z_COORDINATE_SCALE,
            'handle_variable': cls.HANDLE_VARIABLE_LANDMARKS,
        }

    @classmethod
    def get_augmentation_config(cls):
        """Return a dictionary of augmentation configuration values."""
        return {
            'noise_limit': cls.NOISE_LIMIT,
            'rotation_limit': cls.ROTATION_LIMIT,
            'augmentation_strength': cls.AUGMENTATION_STRENGTH,
            'rotation_range': cls.ROTATION_RANGE,
            'scale_range': cls.SCALE_RANGE,
            'translation_range': cls.TRANSLATION_RANGE,
            'brightness_range': cls.BRIGHTNESS_RANGE,
            'contrast_range': cls.CONTRAST_RANGE,
            'flip_probability': cls.FLIP_PROBABILITY,
            'augmentation_probability': cls.AUGMENTATION_PROBABILITY,
        }

    @classmethod
    def validate_config(cls):
        """Validate configuration values and raise on critical errors.

        Returns:
            True if no errors were found, False otherwise.
        """
        errors = []

        if cls.HIDDEN_SIZE <= 0:
            errors.append("HIDDEN_SIZE must be positive.")
        if cls.NUM_HEADS <= 0:
            errors.append("NUM_HEADS must be positive.")
        if not (0 <= cls.DROPOUT_RATE < 1):
            errors.append("DROPOUT_RATE must be in [0, 1).")
        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive.")
        if cls.BASE_LEARNING_RATE <= 0:
            errors.append("BASE_LEARNING_RATE must be positive.")

        for error in errors:
            logging.error(f"Config error: {error}")

        return len(errors) == 0


config = Config()

if torch.cuda.is_available():
    Config.optimize_for_rtx3050()
