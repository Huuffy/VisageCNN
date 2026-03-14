#!/usr/bin/env python3
"""
VisageCNN Enhanced Real-time Inference Script
Smart inference with temporal smoothing, confidence thresholding, and multi-frame voting
"""

import sys
import torch
import numpy as np
import cv2
import time
from pathlib import Path
import logging
from typing import Dict, Optional, List
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from visage_er.config import Config
from visage_er.models.enhanced_model import EnhancedModelUtils
from visage_er.core.face_processor import EnhancedFaceMeshProcessor
import pickle

class SmartEmotionSmoother:
    """Temporal smoothing + multi-frame voting for stable real-time predictions"""

    def __init__(self, window_size: int = 10, ema_alpha: float = 0.3,
                 confidence_threshold: float = 0.35):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold

        # Sliding window of raw probabilities
        self.prob_history: deque = deque(maxlen=window_size)
        # EMA state
        self.ema_probs: Optional[np.ndarray] = None

    def update(self, raw_probs: np.ndarray) -> Dict:
        """Process new frame probabilities and return smoothed result"""
        # Update EMA
        if self.ema_probs is None:
            self.ema_probs = raw_probs.copy()
        else:
            self.ema_probs = self.ema_alpha * raw_probs + (1 - self.ema_alpha) * self.ema_probs

        # Add to sliding window
        self.prob_history.append(raw_probs.copy())

        # Multi-frame voting: average probabilities over the window
        if len(self.prob_history) >= 3:
            window_avg = np.mean(list(self.prob_history), axis=0)
            # Blend EMA with window average (60/40)
            blended = 0.6 * self.ema_probs + 0.4 * window_avg
        else:
            blended = self.ema_probs

        # Normalize
        blended = blended / (blended.sum() + 1e-8)

        # Get prediction
        predicted_class = int(np.argmax(blended))
        confidence = float(blended[predicted_class])

        # Confidence check
        is_confident = confidence >= self.confidence_threshold

        return {
            'class_idx': predicted_class,
            'confidence': confidence,
            'smoothed_probs': blended,
            'is_confident': is_confident
        }

    def reset(self):
        """Reset smoother state (e.g., when face is lost)"""
        self.prob_history.clear()
        self.ema_probs = None


class EnhancedEmotionInference:
    """Enhanced real-time emotion inference pipeline with smart smoothing"""

    def __init__(self, model_path: Optional[str] = None):
        self.device = Config.DEVICE
        self.emotion_classes = Config.EMOTION_CLASSES

        # Load enhanced model
        if model_path is None:
            model_path = Config.MODELS_PATH / "enhanced_best_model.pth"

        self.model, self.checkpoint = EnhancedModelUtils.load_enhanced_model(model_path, self.device)
        self.model.eval()

        # Initialize enhanced face processor
        self.face_processor = EnhancedFaceMeshProcessor()

        # Load coordinate scaler
        self.scaler = self._load_scaler()

        # Smart smoother for temporal stability
        self.smoother = SmartEmotionSmoother(
            window_size=10,
            ema_alpha=0.3,
            confidence_threshold=0.35
        )

        # Performance tracking
        self.inference_times: List[float] = []
        self.frame_count = 0
        self.no_face_streak = 0

        # AMP for inference
        self.use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()

        logging.info(f"Enhanced inference pipeline initialized on {self.device}")

    def _load_scaler(self):
        """Load the enhanced coordinate scaler"""
        scaler_path = Config.MODELS_PATH / "enhanced_coordinate_scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    if isinstance(scaler_data, dict):
                        return scaler_data['scaler']
                    return scaler_data
            except Exception as e:
                logging.warning(f"Could not load scaler: {e}")
        return None

    def predict_emotion(self, frame: np.ndarray) -> Dict:
        """Predict emotion from frame with smart smoothing and distance handling"""
        start_time = time.time()

        h, w = frame.shape[:2]

        # Enhanced coordinate extraction
        coordinates = self.face_processor.extract_coordinates_from_frame_enhanced(frame, is_video=True)

        if coordinates is None:
            self.no_face_streak += 1
            # Reset smoother if face lost for too long
            if self.no_face_streak > 15:
                self.smoother.reset()

            return {
                'emotion': 'No Face Detected',
                'confidence': 0.0,
                'probabilities': {},
                'inference_time': time.time() - start_time,
                'face_detected': False,
                'is_confident': False
            }

        self.no_face_streak = 0

        # Validate face size (skip tiny faces far from camera)
        coords_3d = coordinates.reshape(-1, 3)
        face_width = np.max(coords_3d[:, 0]) - np.min(coords_3d[:, 0])
        face_height = np.max(coords_3d[:, 1]) - np.min(coords_3d[:, 1])

        if face_width < 30 or face_height < 30:
            return {
                'emotion': 'Face Too Small',
                'confidence': 0.0,
                'probabilities': {},
                'inference_time': time.time() - start_time,
                'face_detected': True,
                'is_confident': False
            }

        # Enhanced coordinate normalization with actual frame dimensions
        normalized_coords = self.face_processor.normalize_coordinates_enhanced(
            coordinates, frame_width=w, frame_height=h
        )

        if normalized_coords is None:
            return {
                'emotion': 'Processing Error',
                'confidence': 0.0,
                'probabilities': {},
                'inference_time': time.time() - start_time,
                'face_detected': False,
                'is_confident': False
            }

        # Apply scaling if available
        if self.scaler is not None:
            try:
                normalized_coords = self.scaler.transform([normalized_coords])[0]
            except Exception as e:
                logging.debug(f"Scaler error: {e}")

        # Model inference with optional AMP
        with torch.no_grad():
            tensor_coords = torch.tensor(normalized_coords, dtype=torch.float32).unsqueeze(0)
            tensor_coords = tensor_coords.to(self.device)

            if self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    outputs = self.model(tensor_coords)
            else:
                outputs = self.model(tensor_coords)

            raw_probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

        # Smart smoothing
        smooth_result = self.smoother.update(raw_probs)

        predicted_class = smooth_result['class_idx']
        confidence = smooth_result['confidence']
        smoothed_probs = smooth_result['smoothed_probs']
        is_confident = smooth_result['is_confident']

        # Create probability dictionary
        prob_dict = {
            emotion: float(prob) for emotion, prob in zip(self.emotion_classes, smoothed_probs)
        }

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1

        emotion_label = self.emotion_classes[predicted_class] if is_confident else "Uncertain"

        return {
            'emotion': emotion_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'inference_time': inference_time,
            'face_detected': True,
            'is_confident': is_confident
        }


class RealTimeEmotionDetector:
    """Enhanced real-time emotion detection from camera feed"""

    def __init__(self, model_path: Optional[str] = None, camera_index: int = 0):
        self.inference_engine = EnhancedEmotionInference(model_path)
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False

        # Color coding per emotion
        self.emotion_colors = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 128, 128),
            'Fear': (128, 0, 128),
            'Happy': (0, 255, 0),
            'Neutral': (200, 200, 200),
            'Sad': (255, 100, 0),
            'Surprised': (0, 255, 255),
            'Uncertain': (128, 128, 128),
            'No Face Detected': (0, 0, 200),
            'Face Too Small': (0, 165, 255),
        }

    def start_camera(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logging.error(f"Cannot open camera {self.camera_index}")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.is_running = True
        logging.info(f"Camera {self.camera_index} started")
        return True

    def draw_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw enhanced emotion results on frame with probability bars"""
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]

        emotion = result.get('emotion', 'Unknown')
        confidence = result.get('confidence', 0.0)
        is_confident = result.get('is_confident', False)

        # Get emotion color
        text_color = self.emotion_colors.get(emotion, (255, 255, 255))

        if result['face_detected']:
            # Main emotion display
            text = f"{emotion}: {confidence*100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Background rectangle
            cv2.rectangle(frame_copy, (10, 10),
                         (text_size[0] + 30, text_size[1] + 30),
                         (0, 0, 0), -1)
            cv2.rectangle(frame_copy, (10, 10),
                         (text_size[0] + 30, text_size[1] + 30),
                         text_color, 2)

            cv2.putText(frame_copy, text, (20, 35),
                       font, font_scale, text_color, thickness)

            # Probability bars for all emotions
            if result.get('probabilities'):
                y_offset = 60
                sorted_probs = sorted(result['probabilities'].items(),
                                     key=lambda x: x[1], reverse=True)

                for emotion_name, prob in sorted_probs:
                    bar_width = int(prob * 180)
                    bar_color = self.emotion_colors.get(emotion_name, (100, 255, 100))

                    # Background bar
                    cv2.rectangle(frame_copy, (10, y_offset),
                                 (190, y_offset + 18), (40, 40, 40), -1)
                    # Probability bar
                    cv2.rectangle(frame_copy, (10, y_offset),
                                 (10 + bar_width, y_offset + 18),
                                 bar_color, -1)
                    # Label
                    cv2.putText(frame_copy, f"{emotion_name}: {prob*100:.1f}%",
                               (195, y_offset + 14),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                    y_offset += 22

            # Confidence indicator
            if not is_confident:
                cv2.putText(frame_copy, "Low Confidence", (w - 160, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            cv2.putText(frame_copy, emotion, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # FPS display
        if self.inference_engine.inference_times:
            avg_time = np.mean(self.inference_engine.inference_times[-30:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame_copy, f"FPS: {fps:.1f}", (w - 110, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame_copy

    def run(self):
        """Main detection loop with enhanced processing"""
        if not self.start_camera():
            return

        print("\n" + "="*50)
        print("VisageCNN Enhanced - Real-time Emotion Detection")
        print("="*50)
        print(f"Device: {Config.DEVICE}")
        print(f"Camera: {self.camera_index}")
        print(f"Features: Temporal Smoothing, Multi-frame Voting, Distance Handling")
        print("Press 'q' to quit")
        print("="*50 + "\n")

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process frame with enhanced inference
                result = self.inference_engine.predict_emotion(frame)
                annotated_frame = self.draw_results(frame, result)

                # Display frame
                cv2.imshow('VisageCNN Enhanced - Emotion Detection', annotated_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Performance stats every 100 frames
                if self.inference_engine.frame_count % 100 == 0 and self.inference_engine.frame_count > 0:
                    avg_time = np.mean(self.inference_engine.inference_times[-100:])
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    logging.info(f"Performance - FPS: {fps:.2f}, Frames: {self.inference_engine.frame_count}")

        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.stop_camera()

    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Enhanced VisageCNN emotion detection')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to enhanced model')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index')

    args = parser.parse_args()

    detector = RealTimeEmotionDetector(
        model_path=args.model_path,
        camera_index=args.camera_index
    )

    detector.run()

if __name__ == "__main__":
    main()
