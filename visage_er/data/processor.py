"""
Enhanced Data Processor for VisageCNN - RTX 3050 + Windows Optimized
Fixed sklearn compatibility and Unicode logging issues
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import pickle
from collections import Counter
import random
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from ..config import Config
from ..core.face_processor import EnhancedFaceMeshProcessor
import platform
import time

class AdvancedCoordinateAugmentation:
    """Advanced coordinate augmentation with emotion-specific transformations"""

    def __init__(self):
        self.noise_std = Config.NOISE_LIMIT
        self.rotation_range = np.radians(Config.ROTATION_LIMIT)
        self.scale_range = Config.SCALE_RANGE
        self.translation_range = Config.TRANSLATION_RANGE
        self.augmentation_prob = Config.AUGMENTATION_PROBABILITY

        # Emotion-specific augmentation strengths
        self.emotion_strengths = {
            'happy': 0.8,    # Lower augmentation for well-represented emotions
            'sad': 1.2,      # Higher for subtle expressions
            'angry': 1.0,
            'surprised': 0.9,
            'fear': 1.3,     # Higher for complex expressions
            'disgust': 1.1,
            'neutral': 0.7   # Lower for baseline expression
        }

    def apply_augmentation(self, coordinates: np.ndarray, emotion: str = None) -> np.ndarray:
        """Apply comprehensive coordinate augmentation"""
        if random.random() > self.augmentation_prob:
            return coordinates

        coords = coordinates.copy()

        # Get emotion-specific strength
        strength = self.emotion_strengths.get(emotion, 1.0) if emotion else 1.0

        # Apply transformations
        coords = self._add_gaussian_noise(coords, strength)
        coords = self._apply_rotation_3d(coords, strength)
        coords = self._apply_scaling_3d(coords, strength)
        coords = self._apply_translation_3d(coords, strength)
        coords = self._apply_elastic_deformation_3d(coords, strength)

        return coords

    def _add_gaussian_noise(self, coords: np.ndarray, strength: float) -> np.ndarray:
        """Add Gaussian noise to coordinates"""
        noise = np.random.normal(0, self.noise_std * strength, coords.shape)
        return coords + noise

    def _apply_rotation_3d(self, coords: np.ndarray, strength: float) -> np.ndarray:
        """Apply rotation transformation preserving 3D coordinates"""
        if random.random() > 0.3:  # 30% chance
            return coords

        angle = random.uniform(-self.rotation_range * strength, self.rotation_range * strength)
        coords_3d = coords.reshape(-1, 3)
        center = np.mean(coords_3d[:, :2], axis=0)

        # Rotation matrix (rotate X,Y only, preserve Z)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        xy = coords_3d[:, :2] - center
        rotated_xy = np.column_stack([
            xy[:, 0] * cos_a - xy[:, 1] * sin_a,
            xy[:, 0] * sin_a + xy[:, 1] * cos_a
        ]) + center

        coords_3d[:, :2] = rotated_xy
        return coords_3d.flatten()

    def _apply_scaling_3d(self, coords: np.ndarray, strength: float) -> np.ndarray:
        """Apply scaling transformation preserving 3D coordinates"""
        if random.random() > 0.4:  # 40% chance
            return coords

        scale_factor = random.uniform(
            self.scale_range[0] + (1 - strength) * 0.05,
            self.scale_range[1] - (1 - strength) * 0.05
        )

        coords_3d = coords.reshape(-1, 3)
        center = np.mean(coords_3d, axis=0)
        scaled = (coords_3d - center) * scale_factor + center
        return scaled.flatten()

    def _apply_translation_3d(self, coords: np.ndarray, strength: float) -> np.ndarray:
        """Apply translation transformation preserving 3D coordinates"""
        if random.random() > 0.3:  # 30% chance
            return coords

        coords_3d = coords.reshape(-1, 3)
        coord_range = np.max(coords_3d, axis=0) - np.min(coords_3d, axis=0)

        translation = np.random.uniform(
            -self.translation_range * strength,
            self.translation_range * strength,
            3
        ) * coord_range

        translated = coords_3d + translation
        return translated.flatten()

    def _apply_elastic_deformation_3d(self, coords: np.ndarray, strength: float) -> np.ndarray:
        """Apply elastic deformation preserving 3D coordinates"""
        if random.random() > 0.2:  # 20% chance
            return coords

        coords_3d = coords.reshape(-1, 3)
        num_points = len(coords_3d)

        # Create random displacement field
        displacement = np.random.normal(0, 0.01 * strength, (num_points, 3))

        # Apply smoothing to make deformation more natural
        for i in range(num_points):
            start_idx = max(0, i - 2)
            end_idx = min(num_points, i + 3)
            displacement[i] = np.mean(displacement[start_idx:end_idx], axis=0)

        deformed = coords_3d + displacement
        return deformed.flatten()

class DatasetPerformanceMonitor:
    """Monitor dataset loading performance and statistics"""

    def __init__(self):
        self.load_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_samples_processed = 0

    def record_load_time(self, time_taken: float):
        """Record sample loading time"""
        self.load_times.append(time_taken)

    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.load_times:
            return {}

        return {
            'avg_load_time': np.mean(self.load_times),
            'total_samples': len(self.load_times),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_cache_accesses': self.cache_hits + self.cache_misses
        }

class AdvancedEmotionDataset(Dataset):
    """Enhanced emotion recognition dataset with coordinate caching and advanced augmentation"""

    def __init__(self, data_path: Path, is_training: bool = True,
                 use_weighted_sampling: bool = True, cache_coordinates: bool = True):
        self.data_path = Path(data_path)
        self.is_training = is_training
        self.use_weighted_sampling = use_weighted_sampling
        self.cache_coordinates = cache_coordinates

        # Initialize components
        self.face_processor = EnhancedFaceMeshProcessor()
        self.augmenter = AdvancedCoordinateAugmentation() if is_training else None
        self.monitor = DatasetPerformanceMonitor()

        # Initialize caching
        self.coordinate_cache = {}
        self.cache_file = Config.MODELS_PATH / f"coordinate_cache_{'train' if is_training else 'val'}.pkl"

        # Load dataset
        self.samples = self._load_dataset()

        # Setup weighted sampling
        self.sample_weights = None
        if use_weighted_sampling and is_training:
            self._setup_weighted_sampling()

        # Load coordinate cache
        if cache_coordinates:
            self._load_cache()

        # Initialize scaler
        self.scaler = StandardScaler()
        self._setup_coordinate_normalization()

    def _load_dataset(self) -> List[Tuple[Path, int, str]]:
        """Load and validate dataset with enhanced error handling"""
        samples = []
        for class_idx, emotion in enumerate(Config.EMOTION_CLASSES):
            emotion_path = self.data_path / emotion
            # Try case-insensitive matching if exact path doesn't exist
            if not emotion_path.exists():
                # Search for matching folder with different case
                parent = self.data_path
                matched = None
                if parent.exists():
                    for child in parent.iterdir():
                        if child.is_dir() and child.name.lower() == emotion.lower():
                            matched = child
                            break
                if matched:
                    emotion_path = matched
                else:
                    logging.warning(f"Emotion directory not found: {emotion_path}")
                    continue

            # Fixed: Get all image files without double counting
            image_files = []
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            for ext in extensions:
                image_files.extend(list(emotion_path.glob(ext)))

            # Validate image files
            valid_files = []
            for img_path in image_files:
                if self._validate_image(img_path):
                    valid_files.append(img_path)

            if len(valid_files) == 0:
                logging.warning(f"No valid images found for emotion: {emotion}")
                continue

            # Add to samples with emotion name for augmentation
            for img_path in valid_files:
                samples.append((img_path, class_idx, emotion))

            logging.info(f"{emotion}: {len(valid_files)} valid images loaded")

        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {self.data_path}")

        logging.info(f"Total samples loaded: {len(samples)}")
        return samples

    def _validate_image(self, img_path: Path) -> bool:
        """Validate if image file is readable"""
        try:
            img = cv2.imread(str(img_path))
            return img is not None and img.size > 0
        except Exception:
            return False

    def _setup_weighted_sampling(self):
        """Setup weighted sampling for class balance with sklearn fix"""
        # Get class distribution
        class_counts = Counter([sample[1] for sample in self.samples])

        # Convert to numpy arrays (required by sklearn)
        y_labels = np.array([sample[1] for sample in self.samples])
        unique_classes = np.unique(y_labels)

        try:
            # Calculate class weights using only classes that exist in the data
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,  # numpy array of actual classes
                y=y_labels
            )

            # Create a mapping for all possible classes
            class_weight_dict = {}
            for i, class_idx in enumerate(unique_classes):
                class_weight_dict[class_idx] = class_weights[i]

            # Create sample weights
            self.sample_weights = []
            for _, class_idx, _ in self.samples:
                weight = class_weight_dict.get(class_idx, 1.0)
                self.sample_weights.append(weight)

            self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)

            logging.info("[WEIGHTS] Weighted sampling configured:")
            for class_idx in unique_classes:
                count = class_counts.get(class_idx, 0)
                weight = class_weight_dict.get(class_idx, 1.0)
                emotion = Config.EMOTION_CLASSES[class_idx] if class_idx < len(Config.EMOTION_CLASSES) else f"class_{class_idx}"
                logging.info(f"   {emotion}: {count} samples, weight: {weight:.3f}")

        except Exception as e:
            logging.warning(f"Failed to compute class weights: {e}")
            logging.info("Using equal weights for all samples")
            self.sample_weights = torch.ones(len(self.samples), dtype=torch.float)

    def _load_cache(self):
        """Load coordinate cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.coordinate_cache = cache_data.get('coordinates', {})
                    logging.info(f"[CACHE] Loaded {len(self.coordinate_cache)} cached coordinates")
            except Exception as e:
                logging.warning(f"Failed to load coordinate cache: {e}")
                self.coordinate_cache = {}

    def _save_cache(self):
        """Save coordinate cache to disk"""
        if not self.cache_coordinates:
            return

        try:
            Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'coordinates': self.coordinate_cache,
                'timestamp': time.time(),
                'config': {
                    'num_landmarks': Config.NUM_LANDMARKS,
                    'coordinate_dim': Config.COORDINATE_DIM
                }
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            logging.warning(f"Failed to save coordinate cache: {e}")

    def _setup_coordinate_normalization(self):
        """Setup coordinate normalization with fixed array handling"""
        if not self.is_training:
            # Try to load existing scaler
            scaler_path = Config.MODELS_PATH / "enhanced_coordinate_scaler.pkl"
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        scaler_data = pickle.load(f)
                        self.scaler = scaler_data.get('scaler', StandardScaler())
                        logging.info("[SCALER] Loaded existing coordinate scaler")
                        return
                except Exception as e:
                    logging.warning(f"Failed to load scaler: {e}")

        # Collect sample coordinates with consistent dimensions
        sample_coords = []
        max_samples = min(500, len(self.samples))  # Reduce to avoid memory issues

        logging.info(f"[SCALER] Collecting coordinates from {max_samples} samples...")

        for i in range(0, max_samples, max(1, len(self.samples) // max_samples)):
            img_path, _, _ = self.samples[i]
            coords = self._extract_coordinates(img_path)

            if coords is not None:
                # Ensure consistent dimensions
                coords_fixed = self._ensure_consistent_dimensions(coords)
                if coords_fixed is not None and len(coords_fixed) == Config.COORDINATE_DIM:
                    sample_coords.append(coords_fixed)

                    if len(sample_coords) >= 100:  # Sufficient for normalization
                        break

        logging.info(f"[SCALER] Collected {len(sample_coords)} valid coordinate samples")

        if len(sample_coords) > 10:  # Need minimum samples for scaling
            try:
                # Verify all arrays have the same shape
                shapes = [coords.shape for coords in sample_coords]
                if len(set(shapes)) == 1:  # All shapes are the same
                    sample_array = np.array(sample_coords)
                    self.scaler.fit(sample_array)

                    # Save scaler for validation dataset
                    if self.is_training:
                        try:
                            scaler_data = {
                                'scaler': self.scaler,
                                'timestamp': time.time(),
                                'num_samples': len(sample_coords),
                                'coordinate_dim': Config.COORDINATE_DIM
                            }
                            Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
                            with open(Config.MODELS_PATH / "enhanced_coordinate_scaler.pkl", 'wb') as f:
                                pickle.dump(scaler_data, f)
                            logging.info(f"[SCALER] Saved coordinate scaler (fitted on {len(sample_coords)} samples)")
                        except Exception as e:
                            logging.warning(f"Failed to save scaler: {e}")
                else:
                    logging.error(f"[SCALER] Inconsistent coordinate shapes: {set(shapes)}")
                    logging.info("[SCALER] Using identity scaling (no normalization)")

            except Exception as e:
                logging.error(f"[SCALER] Failed to fit scaler: {e}")
                logging.info("[SCALER] Using identity scaling")
        else:
            logging.warning(f"[SCALER] Insufficient valid samples ({len(sample_coords)}), using identity scaling")

    def _ensure_consistent_dimensions(self, coordinates: np.ndarray) -> Optional[np.ndarray]:
        """Ensure coordinates have consistent dimensions"""
        if coordinates is None:
            return np.zeros(Config.COORDINATE_DIM, dtype=np.float32)

        target_size = Config.COORDINATE_DIM

        # Handle different input sizes
        if len(coordinates) == target_size:
            return coordinates.astype(np.float32)
        elif len(coordinates) == 1434:
            # MediaPipe's actual output - use as is
            return coordinates.astype(np.float32)
        elif len(coordinates) > target_size:
            # Truncate to target size
            logging.debug(f"Truncating coordinates from {len(coordinates)} to {target_size}")
            return coordinates[:target_size].astype(np.float32)
        else:
            # Pad to target size
            logging.debug(f"Padding coordinates from {len(coordinates)} to {target_size}")
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(coordinates)] = coordinates
            return padded

    def _extract_coordinates(self, img_path: Path) -> Optional[np.ndarray]:
        """Extract facial coordinates with consistent dimension handling"""
        cache_key = str(img_path)

        # Check cache first
        if self.cache_coordinates and cache_key in self.coordinate_cache:
            self.monitor.record_cache_hit()
            cached_coords = self.coordinate_cache[cache_key]
            return self._ensure_consistent_dimensions(cached_coords)

        self.monitor.record_cache_miss()
        start_time = time.time()

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None

            h, w = img.shape[:2]

            # Extract coordinates using enhanced face processor
            coordinates = self.face_processor.extract_coordinates_from_frame_enhanced(img)

            if coordinates is not None:
                # Normalize coordinates with actual frame dimensions
                normalized_coords = self.face_processor.normalize_coordinates_enhanced(
                    coordinates, frame_width=w, frame_height=h
                )

                if normalized_coords is not None:
                    # Ensure consistent dimensions
                    fixed_coords = self._ensure_consistent_dimensions(normalized_coords)

                    # Cache the result
                    if self.cache_coordinates and fixed_coords is not None:
                        self.coordinate_cache[cache_key] = fixed_coords

                        # Periodically save cache
                        if len(self.coordinate_cache) % 100 == 0:
                            self._save_cache()

                    self.monitor.record_load_time(time.time() - start_time)
                    return fixed_coords

        except Exception as e:
            logging.warning(f"Failed to extract coordinates from {img_path}: {e}")

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with consistent dimension handling"""
        img_path, class_idx, emotion = self.samples[idx]

        # Extract coordinates with dimension consistency
        coordinates = self._extract_coordinates(img_path)

        # Ensure we always have the right dimensions
        if coordinates is None or len(coordinates) != Config.COORDINATE_DIM:
            coordinates = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
            logging.debug(f"Using zero coordinates for {img_path}")

        # Apply augmentation if training
        if self.is_training and self.augmenter:
            coordinates = self.augmenter.apply_augmentation(coordinates, emotion)

        # Apply normalization with error handling
        try:
            if hasattr(self.scaler, 'transform'):
                coordinates = self.scaler.transform([coordinates])[0]
        except Exception as e:
            logging.debug(f"Normalization failed for {img_path}: {e}")
            # Use unnormalized coordinates as fallback
            pass

        # Convert to tensors
        coord_tensor = torch.tensor(coordinates, dtype=torch.float32)
        label_tensor = torch.tensor(class_idx, dtype=torch.long)

        return coord_tensor, label_tensor

    def __del__(self):
        """Cleanup - save cache when dataset is destroyed"""
        if hasattr(self, 'cache_coordinates') and self.cache_coordinates:
            self._save_cache()


def create_enhanced_data_loaders_windows_optimized(use_weighted_sampling=True, cache_coordinates=True):
    """Create enhanced data loaders optimized for Windows + RTX 3050"""
    logging.info("[LOADER] Creating Windows + RTX 3050 optimized data loaders...")

    # Create enhanced datasets
    train_dataset = AdvancedEmotionDataset(
        data_path=Config.TRAIN_PATH,
        is_training=True,
        use_weighted_sampling=use_weighted_sampling,
        cache_coordinates=cache_coordinates
    )

    val_dataset = AdvancedEmotionDataset(
        data_path=Config.VAL_PATH,
        is_training=False,
        use_weighted_sampling=False,
        cache_coordinates=cache_coordinates
    )

    # Validate datasets
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {Config.TRAIN_PATH}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation samples found in {Config.VAL_PATH}")

    logging.info(f"[CONFIG] Windows + RTX 3050 optimized dataset configuration:")
    logging.info(f" Training samples: {len(train_dataset)}")
    logging.info(f" Validation samples: {len(val_dataset)}")
    logging.info(f" Platform: {platform.system()}")

    # Windows-specific optimizations
    windows_config = Config.get_windows_safe_config()

    # Create sampler for balanced training
    sampler = None
    if use_weighted_sampling and train_dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        logging.info("[SAMPLER] Weighted sampling enabled for class balance")

    # RTX 3050 + Windows optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        **windows_config,
        drop_last=Config.DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        **windows_config,
        drop_last=False
    )

    logging.info("[SUCCESS] Windows + RTX 3050 optimized data loaders created")
    logging.info(f"   - Multiprocessing: Disabled (Windows safe)")
    logging.info(f"   - Memory Pinning: {windows_config['pin_memory']}")
    logging.info(f"   - Batch Size: {Config.BATCH_SIZE}")
    logging.info(f"   - Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")

    return train_loader, val_loader

def create_enhanced_data_loaders_original(use_weighted_sampling=True, cache_coordinates=True):
    """Original enhanced data loaders for Linux/Mac systems"""
    logging.info("[LOADER] Creating standard enhanced data loaders...")

    # Create enhanced datasets
    train_dataset = AdvancedEmotionDataset(
        data_path=Config.TRAIN_PATH,
        is_training=True,
        use_weighted_sampling=use_weighted_sampling,
        cache_coordinates=cache_coordinates
    )

    val_dataset = AdvancedEmotionDataset(
        data_path=Config.VAL_PATH,
        is_training=False,
        use_weighted_sampling=False,
        cache_coordinates=cache_coordinates
    )

    # Validate datasets
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {Config.TRAIN_PATH}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation samples found in {Config.VAL_PATH}")

    # Create sampler for balanced training
    sampler = None
    if use_weighted_sampling and train_dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

    # Standard data loaders with multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True,
        drop_last=Config.DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True,
        drop_last=False
    )

    logging.info("[SUCCESS] Standard enhanced data loaders created")
    logging.info(f"   - Workers: {Config.NUM_WORKERS}")
    logging.info(f"   - Memory Pinning: {Config.PIN_MEMORY}")

    return train_loader, val_loader

def create_enhanced_data_loaders(use_weighted_sampling=True, cache_coordinates=True):
    """Enhanced wrapper that detects platform and optimizes accordingly"""
    logging.info("[INIT] Initializing Enhanced Data Loaders...")

    if platform.system() == 'Windows':
        logging.info("[PLATFORM] Detected Windows - using optimized configuration")
        return create_enhanced_data_loaders_windows_optimized(
            use_weighted_sampling=use_weighted_sampling,
            cache_coordinates=cache_coordinates
        )
    else:
        logging.info("[PLATFORM] Detected Linux/Mac - using standard configuration")
        return create_enhanced_data_loaders_original(
            use_weighted_sampling=use_weighted_sampling,
            cache_coordinates=cache_coordinates
        )

# Utility functions for dataset analysis
def analyze_dataset_statistics(data_path: Path) -> Dict:
    """Analyze dataset statistics and provide insights"""
    stats = {
        'total_samples': 0,
        'class_distribution': {},
        'class_balance_ratio': 0.0,
        'recommendations': []
    }

    class_counts = {}
    for emotion in Config.EMOTION_CLASSES:
        emotion_path = data_path / emotion
        if emotion_path.exists():
            image_files = []
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            for ext in extensions:
                image_files.extend(list(emotion_path.glob(ext)))

            count = len(image_files)
            class_counts[emotion] = count
            stats['total_samples'] += count

    stats['class_distribution'] = class_counts

    if stats['total_samples'] > 0:
        min_count = min(class_counts.values()) if class_counts.values() else 0
        max_count = max(class_counts.values()) if class_counts.values() else 0
        stats['class_balance_ratio'] = min_count / max_count if max_count > 0 else 0

        # Generate recommendations
        if stats['class_balance_ratio'] < 0.5:
            stats['recommendations'].append("Consider data augmentation for minority classes")
        if stats['total_samples'] < 1000:
            stats['recommendations'].append("Small dataset - consider transfer learning")
        if any(count < 50 for count in class_counts.values()):
            stats['recommendations'].append("Some classes have very few samples - collect more data")

    return stats

def prepare_coordinate_cache(data_path: Path, cache_name: str = "coordinate_cache.pkl"):
    """Pre-populate coordinate cache for faster training"""
    logging.info(f"[CACHE] Pre-populating coordinate cache for {data_path}")

    face_processor = EnhancedFaceMeshProcessor()
    cache = {}
    processed = 0

    for emotion in Config.EMOTION_CLASSES:
        emotion_path = data_path / emotion
        if not emotion_path.exists():
            continue

        image_files = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        for ext in extensions:
            image_files.extend(list(emotion_path.glob(ext)))

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    coordinates = face_processor.extract_coordinates_from_frame_enhanced(img)
                    if coordinates is not None:
                        normalized_coords = face_processor.normalize_coordinates_enhanced(coordinates)
                        if normalized_coords is not None:
                            cache[str(img_path)] = normalized_coords
                            processed += 1

                            if processed % 100 == 0:
                                logging.info(f"[PROGRESS] Processed {processed} images...")
            except Exception as e:
                logging.warning(f"Failed to process {img_path}: {e}")

    # Save cache
    cache_path = Config.MODELS_PATH / cache_name
    Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

    cache_data = {
        'coordinates': cache,
        'timestamp': time.time(),
        'total_cached': len(cache)
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    logging.info(f"[SUCCESS] Cached {len(cache)} coordinate sets to {cache_path}")
    return cache_path

# Export main functions
__all__ = [
    'AdvancedEmotionDataset',
    'AdvancedCoordinateAugmentation',
    'DatasetPerformanceMonitor',
    'create_enhanced_data_loaders',
    'analyze_dataset_statistics',
    'prepare_coordinate_cache'
]
