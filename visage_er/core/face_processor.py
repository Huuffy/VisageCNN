"""
Enhanced Face Processor for VisageCNN - Fixed coordinate dimension handling
Robust error handling for missing config attributes
"""
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import time
from ..config import Config

class EnhancedFaceMeshProcessor:
    """Enhanced face mesh processor with robust attribute handling and quality assessment"""

    def __init__(self):
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Enhanced face mesh with more landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=getattr(Config, 'USE_REFINED_LANDMARKS', True),
            min_detection_confidence=getattr(Config, 'FACE_CONFIDENCE_THRESHOLD', 0.7),
            min_tracking_confidence=getattr(Config, 'FACE_CONFIDENCE_THRESHOLD', 0.7)
        )

        # Video processing face mesh (for real-time)
        self.face_mesh_video = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=getattr(Config, 'USE_REFINED_LANDMARKS', True),
            min_detection_confidence=getattr(Config, 'FACE_CONFIDENCE_THRESHOLD', 0.7),
            min_tracking_confidence=getattr(Config, 'FACE_CONFIDENCE_THRESHOLD', 0.7)
        )

        # Coordinate smoothing for temporal stability
        self.prev_coordinates = None
        self.smoothing_alpha = getattr(Config, 'COORDINATE_SMOOTHING_ALPHA', 0.3)

        # Quality assessment parameters
        self.min_quality = getattr(Config, 'MIN_COORDINATE_QUALITY', 0.5)
        self.max_variance = getattr(Config, 'MAX_COORDINATE_VARIANCE', 100.0)

        logging.info("Enhanced MediaPipe components initialized")

    def extract_coordinates_from_frame_enhanced(self, frame: np.ndarray, is_video: bool = False) -> Optional[np.ndarray]:
        """Extract enhanced 3D facial coordinates from frame with robust error handling"""
        if frame is None:
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Choose appropriate face mesh processor
            face_mesh_processor = self.face_mesh_video if is_video else self.face_mesh

            # Process the frame
            results = face_mesh_processor.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Extract 3D coordinates (x, y, z)
                coordinates_3d = []
                height, width = frame.shape[:2]

                for landmark in face_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * width
                    y = landmark.y * height
                    z = landmark.z * width  # Z is also normalized

                    coordinates_3d.extend([x, y, z])

                coordinates_array = np.array(coordinates_3d, dtype=np.float32)

                # Validate coordinate dimensions with flexible handling
                expected_size = getattr(Config, 'COORDINATE_DIM', 1434)

                if len(coordinates_array) != expected_size:
                    logging.debug(f"Coordinate size: expected {expected_size}, got {len(coordinates_array)}")
                    # Handle dimension mismatch gracefully
                    coordinates_array = self._handle_dimension_mismatch(coordinates_array, expected_size)

                # Apply coordinate smoothing for video processing
                if is_video and self.prev_coordinates is not None:
                    coordinates_array = self._smooth_coordinates(coordinates_array, self.prev_coordinates)

                self.prev_coordinates = coordinates_array.copy()

                return coordinates_array

        except Exception as e:
            logging.error(f"Error in enhanced coordinate extraction: {e}")

        return None

    def _handle_dimension_mismatch(self, coordinates: np.ndarray, expected_size: int) -> np.ndarray:
        """Handle coordinate dimension mismatches gracefully"""
        if len(coordinates) == expected_size:
            return coordinates
        elif len(coordinates) > expected_size:
            # Truncate if too large
            logging.debug(f"Truncating coordinates from {len(coordinates)} to {expected_size}")
            return coordinates[:expected_size]
        else:
            # Pad if too small
            logging.debug(f"Padding coordinates from {len(coordinates)} to {expected_size}")
            padded = np.zeros(expected_size, dtype=np.float32)
            padded[:len(coordinates)] = coordinates
            return padded

    def normalize_coordinates_enhanced(self, coordinates: np.ndarray, frame_width: int = 640, frame_height: int = 480) -> Optional[np.ndarray]:
        """Enhanced coordinate normalization using actual frame dimensions"""
        if coordinates is None or len(coordinates) == 0:
            return None

        try:
            coords = coordinates.copy()

            # Reshape to (num_landmarks, 3) for processing with safe attribute access
            if getattr(Config, 'USE_3D_COORDINATES', True):
                coords_3d = coords.reshape(-1, 3)

                # Normalize X and Y coordinates using actual frame dimensions
                half_w = frame_width / 2.0
                half_h = frame_height / 2.0
                coords_3d[:, 0] = (coords_3d[:, 0] - half_w) / half_w
                coords_3d[:, 1] = (coords_3d[:, 1] - half_h) / half_h

                # Normalize Z coordinates with safe attribute access
                if getattr(Config, 'NORMALIZE_Z_COORDINATES', True):
                    z_scale = getattr(Config, 'Z_COORDINATE_SCALE', 0.1)
                    coords_3d[:, 2] = coords_3d[:, 2] * z_scale

                # Apply outlier detection and clipping
                if getattr(Config, 'APPLY_OUTLIER_REMOVAL', True):
                    coords_3d = self._remove_outliers(coords_3d)

                # Flatten back to 1D array
                normalized_coords = coords_3d.flatten()
            else:
                # 2D normalization (legacy support)
                coords_2d = coords.reshape(-1, 2)
                half_w = frame_width / 2.0
                half_h = frame_height / 2.0
                coords_2d[:, 0] = (coords_2d[:, 0] - half_w) / half_w
                coords_2d[:, 1] = (coords_2d[:, 1] - half_h) / half_h
                normalized_coords = coords_2d.flatten()

            return normalized_coords.astype(np.float32)

        except Exception as e:
            logging.error(f"Error in enhanced coordinate normalization: {e}")
            return None

    def _smooth_coordinates(self, current_coords: np.ndarray, prev_coords: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to coordinates with enhanced detection"""
        if len(current_coords) != len(prev_coords):
            return current_coords

        # Calculate coordinate change magnitude
        change = np.linalg.norm(current_coords - prev_coords)

        # Apply smoothing with adaptive alpha based on change magnitude
        large_change_threshold = getattr(Config, 'MAX_COORDINATE_VARIANCE', 100.0) * 0.5

        if change > large_change_threshold:
            logging.warning("Large coordinate change detected, possible noise")
            alpha = 0.1  # Heavy smoothing
        else:
            alpha = self.smoothing_alpha

        smoothed_coords = alpha * current_coords + (1 - alpha) * prev_coords
        return smoothed_coords

    def _remove_outliers(self, coordinates: np.ndarray) -> np.ndarray:
        """Remove coordinate outliers using statistical methods with safe parameters"""
        coords = coordinates.copy()

        # Get outlier threshold with safe attribute access
        outlier_threshold = getattr(Config, 'Z_OUTLIER_THRESHOLD', 3.0)

        # Calculate statistics for each dimension
        for dim in range(coords.shape[1]):
            coord_dim = coords[:, dim]

            # Use IQR method for outlier detection
            try:
                q75, q25 = np.percentile(coord_dim, [75, 25])
                iqr = q75 - q25

                # Calculate bounds
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)

                # Clip outliers
                coords[:, dim] = np.clip(coord_dim, lower_bound, upper_bound)
            except Exception as e:
                logging.debug(f"Outlier removal failed for dimension {dim}: {e}")
                continue

        return coords

    def assess_face_quality(self, coordinates: np.ndarray, frame: np.ndarray = None) -> Dict:
        """Assess quality of detected face coordinates with robust error handling"""
        quality_metrics = {
            'overall_score': 0.0,
            'coordinate_stability': 0.0,
            'landmark_completeness': 0.0,
            'face_symmetry': 0.0,
            'coordinate_variance': 0.0
        }

        if coordinates is None or len(coordinates) == 0:
            return quality_metrics

        try:
            # Reshape coordinates for analysis with safe attribute access
            if getattr(Config, 'USE_3D_COORDINATES', True):
                coords = coordinates.reshape(-1, 3)
            else:
                coords = coordinates.reshape(-1, 2)

            # 1. Landmark completeness (all landmarks detected)
            expected_landmarks = getattr(Config, 'NUM_LANDMARKS', 478)
            quality_metrics['landmark_completeness'] = 1.0 if len(coords) == expected_landmarks else 0.5

            # 2. Coordinate variance (stability check)
            try:
                variance = np.var(coords, axis=0)
                max_variance = getattr(Config, 'MAX_COORDINATE_VARIANCE', 100.0)
                quality_metrics['coordinate_variance'] = 1.0 / (1.0 + np.mean(variance) / max_variance)
            except Exception:
                quality_metrics['coordinate_variance'] = 0.5

            # 3. Face symmetry analysis
            if len(coords) >= expected_landmarks // 2:
                try:
                    center_x = np.mean(coords[:, 0])
                    left_points = coords[coords[:, 0] < center_x]
                    right_points = coords[coords[:, 0] >= center_x]

                    if len(left_points) > 0 and len(right_points) > 0:
                        left_variance = np.var(left_points, axis=0)
                        right_variance = np.var(right_points, axis=0)
                        symmetry = 1.0 / (1.0 + abs(np.mean(left_variance) - np.mean(right_variance)))
                        quality_metrics['face_symmetry'] = symmetry
                except Exception:
                    quality_metrics['face_symmetry'] = 0.5

            # 4. Coordinate stability (if previous coordinates exist)
            if self.prev_coordinates is not None:
                try:
                    if getattr(Config, 'USE_3D_COORDINATES', True):
                        prev_coords = self.prev_coordinates.reshape(-1, 3)
                    else:
                        prev_coords = self.prev_coordinates.reshape(-1, 2)

                    if len(coords) == len(prev_coords):
                        stability = 1.0 / (1.0 + np.mean(np.linalg.norm(coords - prev_coords, axis=1)))
                        quality_metrics['coordinate_stability'] = stability
                except Exception:
                    quality_metrics['coordinate_stability'] = 0.5
            else:
                quality_metrics['coordinate_stability'] = 1.0

            # Calculate overall score
            quality_metrics['overall_score'] = np.mean([
                quality_metrics['landmark_completeness'],
                quality_metrics['coordinate_variance'],
                quality_metrics['face_symmetry'],
                quality_metrics['coordinate_stability']
            ])

        except Exception as e:
            logging.error(f"Error in quality assessment: {e}")

        return quality_metrics

    def extract_geometric_features(self, coordinates: np.ndarray) -> Optional[np.ndarray]:
        """Extract geometric features from facial landmarks with robust error handling"""
        if coordinates is None or len(coordinates) == 0:
            return None

        try:
            if getattr(Config, 'USE_3D_COORDINATES', True):
                coords = coordinates.reshape(-1, 3)
            else:
                coords = coordinates.reshape(-1, 2)

            geometric_features = []

            # Expected number of landmarks
            expected_landmarks = getattr(Config, 'NUM_LANDMARKS', 478)

            # Eye aspect ratio (both eyes) - using approximate indices
            try:
                if len(coords) >= 468:  # Standard MediaPipe landmarks
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

                    # Ensure indices are within bounds
                    left_eye_indices = [i for i in left_eye_indices if i < len(coords)]
                    right_eye_indices = [i for i in right_eye_indices if i < len(coords)]

                    if len(left_eye_indices) >= 6 and len(right_eye_indices) >= 6:
                        left_eye_coords = coords[left_eye_indices]
                        right_eye_coords = coords[right_eye_indices]

                        left_ear = self._calculate_eye_aspect_ratio(left_eye_coords)
                        right_ear = self._calculate_eye_aspect_ratio(right_eye_coords)
                        geometric_features.extend([left_ear, right_ear])
                    else:
                        geometric_features.extend([0.0, 0.0])
                else:
                    geometric_features.extend([0.0, 0.0])
            except Exception as e:
                logging.debug(f"Eye aspect ratio calculation failed: {e}")
                geometric_features.extend([0.0, 0.0])

            # Mouth aspect ratio
            try:
                if len(coords) >= 468:
                    mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
                    mouth_indices = [i for i in mouth_indices if i < len(coords)]

                    if len(mouth_indices) >= 6:
                        mouth_coords = coords[mouth_indices]
                        mar = self._calculate_mouth_aspect_ratio(mouth_coords)
                        geometric_features.append(mar)
                    else:
                        geometric_features.append(0.0)
                else:
                    geometric_features.append(0.0)
            except Exception as e:
                logging.debug(f"Mouth aspect ratio calculation failed: {e}")
                geometric_features.append(0.0)

            # Face symmetry measure
            try:
                if len(coords) >= expected_landmarks // 2:
                    center_x = np.mean(coords[:, 0])
                    symmetry = self._calculate_face_symmetry(coords, center_x)
                    geometric_features.append(symmetry)
                else:
                    geometric_features.append(0.0)
            except Exception as e:
                logging.debug(f"Face symmetry calculation failed: {e}")
                geometric_features.append(0.0)

            return np.array(geometric_features, dtype=np.float32)

        except Exception as e:
            logging.error(f"Error extracting geometric features: {e}")
            return np.array([0.0] * 4, dtype=np.float32)

    def _calculate_eye_aspect_ratio(self, eye_coords: np.ndarray) -> float:
        """Calculate eye aspect ratio with error handling"""
        if len(eye_coords) < 6:
            return 0.0

        try:
            # Approximate EAR calculation
            vertical_dist = np.mean([
                np.linalg.norm(eye_coords[1] - eye_coords[5]),
                np.linalg.norm(eye_coords[2] - eye_coords[4])
            ])
            horizontal_dist = np.linalg.norm(eye_coords[0] - eye_coords[3])

            if horizontal_dist > 0:
                return vertical_dist / horizontal_dist
        except Exception:
            pass
        return 0.0

    def _calculate_mouth_aspect_ratio(self, mouth_coords: np.ndarray) -> float:
        """Calculate mouth aspect ratio with error handling"""
        if len(mouth_coords) < 6:
            return 0.0

        try:
            # Approximate MAR calculation
            vertical_dist = np.mean([
                np.linalg.norm(mouth_coords[2] - mouth_coords[10]),
                np.linalg.norm(mouth_coords[4] - mouth_coords[8])
            ])
            horizontal_dist = np.linalg.norm(mouth_coords[0] - mouth_coords[6])

            if horizontal_dist > 0:
                return vertical_dist / horizontal_dist
        except Exception:
            pass
        return 0.0

    def _calculate_face_symmetry(self, coords: np.ndarray, center_x: float) -> float:
        """Calculate face symmetry measure with error handling"""
        try:
            left_coords = coords[coords[:, 0] < center_x]
            right_coords = coords[coords[:, 0] >= center_x]

            if len(left_coords) == 0 or len(right_coords) == 0:
                return 0.0

            # Calculate symmetry based on coordinate distribution
            left_spread = np.std(left_coords, axis=0)
            right_spread = np.std(right_coords, axis=0)

            symmetry = 1.0 / (1.0 + np.mean(np.abs(left_spread - right_spread)))
            return symmetry
        except Exception:
            return 0.0

# Export the enhanced face processor
__all__ = ['EnhancedFaceMeshProcessor']
