"""
Enhanced Configuration for VisageCNN - RTX 3050 Laptop GPU 4GB Optimized
Performance tested and optimized for batch size 256
"""
import torch
import os
from pathlib import Path
import logging

class Config:
    """Enhanced configuration class optimized for RTX 3050 Laptop GPU (4GB VRAM)"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATASET_PATH = PROJECT_ROOT / "dataset"
    MODELS_PATH = PROJECT_ROOT / "models"
    WEIGHTS_PATH = MODELS_PATH / "weights"
    SCALERS_PATH = MODELS_PATH / "scalers"
    CACHE_PATH = MODELS_PATH / "cache"
    LOGS_PATH = PROJECT_ROOT / "logs"
    CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"
    
    # Dataset structure - All possible path variations for compatibility
    TRAIN_DATA_PATH = DATASET_PATH / "train"
    VAL_DATA_PATH = DATASET_PATH / "val"
    TRAIN_PATH = DATASET_PATH / "train"
    VAL_PATH = DATASET_PATH / "val"
    VALIDATION_PATH = DATASET_PATH / "val"
    
    # Enhanced model configuration
    NUM_CLASSES = 7
    EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    # MediaPipe coordinate processing to handle actual output (1434 features)
    NUM_LANDMARKS = 478  # MediaPipe returns 478 landmarks with refined mesh
    COORDINATE_FEATURES = 3  # x, y, z coordinates
    NUM_COORDINATE_FEATURES = 1434  # Actual MediaPipe output size
    COORDINATE_DIM = 1434
    FEATURE_SIZE = 1434
    INPUT_SIZE = 1434
    
    # MediaPipe processing flags
    USE_3D_COORDINATES = True
    EXTRACT_ONLY_XYZ = True
    NORMALIZE_COORDINATES = True
    HANDLE_VARIABLE_LANDMARKS = True
    
    # Coordinate processing attributes
    NORMALIZE_Z_COORDINATES = True
    Z_COORDINATE_SCALE = 0.1
    COORDINATE_PREPROCESSING = True
    USE_COORDINATE_SMOOTHING = True
    APPLY_OUTLIER_REMOVAL = True
    
    # Z-coordinate specific settings
    Z_NORMALIZATION_METHOD = 'scale'  # 'scale', 'standardize', or 'clip'
    Z_CLIP_RANGE = (-1.0, 1.0)
    Z_OUTLIER_THRESHOLD = 3.0
    
    # Coordinate smoothing parameters
    SMOOTHING_WINDOW_SIZE = 5
    SMOOTHING_WEIGHT_DECAY = 0.9
    
    # Quality assessment for coordinates
    MIN_COORDINATE_QUALITY = 0.5
    MAX_COORDINATE_VARIANCE = 100.0
    
    # MediaPipe settings optimized for highest quality
    USE_REFINED_LANDMARKS = True
    ENABLE_IRIS_LANDMARKS = True
    FACE_MESH_COMPLEXITY = 2                # Highest quality (0, 1, or 2)
    MEDIAPIPE_MODEL_COMPLEXITY = 2
    MEDIAPIPE_REFINE_LANDMARKS = True
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
    
    # RTX 3050 4GB OPTIMIZED MODEL PARAMETERS (Performance tested)
    BATCH_SIZE = 96                         # Increased to use available VRAM (~3.5GB target)
    GRADIENT_ACCUMULATION_STEPS = 2        # Effective batch size = 192
    
    # Model architecture (right-sized to prevent overfitting on ~4k samples)
    HIDDEN_SIZE = 640                      # Increased to use available VRAM
    NUM_HEADS = 8                          # Reduced from 16
    NUM_LAYERS = 4                         # Reduced from 8 to prevent overfitting
    DROPOUT_RATE = 0.3                     # Increased from 0.05 for regularization
    GEOMETRIC_FEATURE_DIM = 64             # Reduced from 128
    EXPERT_HIDDEN_SIZE = 128               # Reduced from 256
    HIDDEN_LAYERS = [640]  # Right-sized layers
    DROPOUT_RATES = [0.3]
    
    # Attention mechanism parameters
    ATTENTION_DIM = 256                    # Reduced from 512
    KEY_DIM = 32                           # Reduced from 64
    VALUE_DIM = 32                         # Reduced from 64
    
    # Training parameters
    EPOCHS = 300
    BASE_LEARNING_RATE = 0.0001
    MIN_LEARNING_RATE = 0.000005
    WEIGHT_DECAY = 0.01
    
    # Enhanced loss function parameters
    FOCAL_LOSS_ALPHA = 1.0
    FOCAL_LOSS_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.1                     # Reduced from 0.2 to reduce underfitting
    
    # Enhanced training strategies
    EARLY_STOPPING_PATIENCE = 40          # Longer patience for 300 epochs
    GRADIENT_CLIP_VALUE = 1.0
    WARMUP_EPOCHS = 10
    
    # Class weights to boost weak classes (Sad, Fear)
    # Order: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
    CLASS_WEIGHTS = [1.0, 1.0, 1.3, 1.0, 1.0, 1.5, 1.0]
    
    # Augmentation parameters
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
    
    # Enhanced face processing
    FACE_CONFIDENCE_THRESHOLD = 0.7
    FACE_QUALITY_THRESHOLD = 0.6
    COORDINATE_SMOOTHING_ALPHA = 0.3
    MIN_FACE_SIZE = 50
    MAX_FACE_SIZE = 500
    
    # Windows + RTX 3050 4GB Performance optimization
    NUM_WORKERS = 0                        # Disabled for Windows compatibility
    PIN_MEMORY = True                      # Enable for GPU acceleration
    CACHE_COORDINATES = True               # Enable caching for performance
    COORDINATE_CACHE_SIZE = 24000          # Increased from 16000 for 4GB VRAM
    
    # Data loading parameters optimized for Windows
    PREFETCH_FACTOR = 4                    # Increased for larger batches
    PERSISTENT_WORKERS = False             # Must be False when NUM_WORKERS = 0
    DROP_LAST = True
    
    # RTX 3050 4GB GPU configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Memory management optimized for 4GB VRAM
    MAX_MEMORY_USAGE = 0.85                # Use 85% of 4GB GPU memory
    MIXED_PRECISION = True                 # Essential for RTX 3050
    CUDA_MEMORY_FRACTION = 1.0             # 100% utilization as tested
    ALLOW_MEMORY_GROWTH = True
    EMPTY_CACHE_FREQUENCY = 50             # Less frequent clearing needed
    
    # Enhanced logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model saving
    SAVE_TOP_K = 3
    SAVE_EVERY_N_EPOCHS = 10
    
    # Data preprocessing
    IMAGE_SIZE = (224, 224)
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    # Coordinate normalization
    COORDINATE_NORMALIZATION = True
    COORDINATE_SCALE_FACTOR = 1.0
    
    # Advanced model parameters
    USE_RESIDUAL_CONNECTIONS = True
    USE_BATCH_NORMALIZATION = True
    USE_LAYER_NORMALIZATION = True
    
    # Expert network configuration
    NUM_EXPERTS = 7
    EXPERT_ACTIVATION = 'relu'
    EXPERT_USE_DROPOUT = True
    
    # Validation settings
    VALIDATION_FREQUENCY = 1
    SAVE_FREQUENCY = 5
    
    # Coordinate processing limits
    COORDINATE_MIN_VALUE = -1000
    COORDINATE_MAX_VALUE = 1000
    OUTLIER_THRESHOLD = 3.0
    
    # Quality assessment thresholds
    MIN_LANDMARK_CONFIDENCE = 0.5
    MAX_FACE_ANGLE = 45
    MIN_FACE_AREA = 2500
    
    # Performance monitoring (added)
    ENABLE_PROFILING = True
    LOG_MEMORY_USAGE = True
    PROFILE_EVERY_N_BATCHES = 100
    
    # Geometric feature extraction
    GEOMETRIC_FEATURES = [
        'eye_aspect_ratio',
        'mouth_aspect_ratio',
        'eyebrow_height',
        'face_symmetry',
        'facial_angles'
    ]
    
    # Quality assessment parameters
    QUALITY_CHECKS = [
        'face_size',
        'face_angle',
        'lighting_condition',
        'blur_detection',
        'occlusion_detection'
    ]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for enhanced pipeline"""
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
        
        created_count = 0
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created_count += 1
        
        if created_count > 0:
            print(f"[OK] Created {created_count} directories for enhanced pipeline")
        else:
            print("[OK] All directories already exist")
    
    @classmethod
    def optimize_for_rtx3050(cls):
        """Apply RTX 3050 4GB GPU specific optimizations"""
        if torch.cuda.is_available():
            # Set memory fraction for 4GB VRAM
            torch.cuda.set_per_process_memory_fraction(cls.CUDA_MEMORY_FRACTION)
            
            # Enable optimized operations for RTX 3050
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
            
            print(f"[OK] RTX 3050 4GB GPU optimizations applied")
            print(f"   - Memory fraction: {cls.CUDA_MEMORY_FRACTION}")
            print(f"   - Mixed precision: {cls.MIXED_PRECISION}")
            print(f"   - Batch size: {cls.BATCH_SIZE}")
            print(f"   - Gradient accumulation: {cls.GRADIENT_ACCUMULATION_STEPS}")
    
    @classmethod
    def get_windows_safe_config(cls):
        """Get Windows-safe configuration parameters"""
        return {
            'num_workers': cls.NUM_WORKERS,
            'pin_memory': cls.PIN_MEMORY,
            'persistent_workers': cls.PERSISTENT_WORKERS,
            'multiprocessing_context': None,
            'timeout': 0
        }
    
    @classmethod
    def get_device_info(cls):
        """Get enhanced device information"""
        if torch.cuda.is_available():
            return {
                'device': 'cuda',
                'device_name': torch.cuda.get_device_name(0),
                'memory_total': torch.cuda.get_device_properties(0).total_memory // (1024**3),
                'memory_allocated': torch.cuda.memory_allocated(0) // (1024**3),
                'compute_capability': torch.cuda.get_device_properties(0).major
            }
        else:
            return {
                'device': 'cpu',
                'cores': os.cpu_count()
            }
    
    @classmethod
    def update_learning_rate(cls, optimizer, epoch):
        """Enhanced dynamic learning rate update"""
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
        """Get coordinate processing configuration"""
        return {
            'num_landmarks': cls.NUM_LANDMARKS,
            'coordinate_features': cls.COORDINATE_FEATURES,
            'total_features': cls.NUM_COORDINATE_FEATURES,
            'use_3d': cls.USE_3D_COORDINATES,
            'normalize': cls.NORMALIZE_COORDINATES,
            'normalize_z': cls.NORMALIZE_Z_COORDINATES,
            'z_scale': cls.Z_COORDINATE_SCALE,
            'handle_variable': cls.HANDLE_VARIABLE_LANDMARKS
        }
    
    @classmethod
    def get_augmentation_config(cls):
        """Get augmentation configuration dictionary"""
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
            'augmentation_probability': cls.AUGMENTATION_PROBABILITY
        }
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters"""
        errors = []
        warnings = []
        
        # Check coordinate dimensions
        expected_features = cls.NUM_LANDMARKS * cls.COORDINATE_FEATURES
        if cls.NUM_COORDINATE_FEATURES != expected_features and cls.NUM_COORDINATE_FEATURES != 1434:
            warnings.append(f"Coordinate feature mismatch: expected {expected_features} or 1434, got {cls.NUM_COORDINATE_FEATURES}")
        
        # Updated validation for 4GB VRAM
        if cls.BATCH_SIZE > 512:
            warnings.append("Very large batch size - monitor memory usage")
        
        if cls.HIDDEN_SIZE > 1024:
            warnings.append("Very large hidden size - monitor memory usage")
        
        # Check essential parameters
        if cls.HIDDEN_SIZE <= 0:
            errors.append("HIDDEN_SIZE must be positive")
        
        if cls.NUM_HEADS <= 0:
            errors.append("NUM_HEADS must be positive")
        
        if cls.DROPOUT_RATE < 0 or cls.DROPOUT_RATE >= 1:
            errors.append("DROPOUT_RATE must be in [0, 1)")
        
        # Check training parameters
        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        
        if cls.BASE_LEARNING_RATE <= 0:
            errors.append("BASE_LEARNING_RATE must be positive")
        
        # Print results
        if errors:
            print("[!!] Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print("[!] Configuration validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if not errors and not warnings:
            print("[OK] Configuration validation passed")
        
        return len(errors) == 0
    
    @classmethod
    def print_config(cls):
        """Print RTX 3050 4GB optimized configuration"""
        print("=" * 70)
        print("RTX 3050 4GB PERFORMANCE OPTIMIZED VISAGECNN CONFIGURATION")
        print("=" * 70)
        print(f"Device: {cls.DEVICE}")
        print(f"GPU Memory: 4GB (100% utilization)")
        print(f"Batch Size: {cls.BATCH_SIZE} (performance tested)")
        print(f"Model Architecture: Enhanced (8 layers, 16 heads)")
        print(f"Hidden Size: {cls.HIDDEN_SIZE} (optimized for 4GB)")
        print(f"Face Mesh Complexity: {cls.FACE_MESH_COMPLEXITY} (highest quality)")
        print(f"Coordinate Features: {cls.NUM_COORDINATE_FEATURES} ({cls.NUM_LANDMARKS} landmarks)")
        print(f"3D Coordinates: {cls.USE_3D_COORDINATES}")
        print(f"Mixed Precision: {cls.MIXED_PRECISION}")
        print(f"Cache Size: {cls.COORDINATE_CACHE_SIZE}")
        print("=" * 70)

# Global configuration instance
config = Config()

# Apply RTX 3050 optimizations on import
if torch.cuda.is_available():
    Config.optimize_for_rtx3050()

# Validate configuration on import
try:
    Config.validate_config()
except Exception as e:
    print(f"[!] Configuration validation error: {e}")
