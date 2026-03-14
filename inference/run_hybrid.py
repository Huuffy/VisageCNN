"""
Hybrid Model Inference - Real-time emotion recognition
using both facial landmarks AND face appearance
"""

# Suppress noisy warnings
import warnings
import os
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import mediapipe as mp
from pathlib import Path
from collections import deque
import pickle
import logging
import argparse
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from visage_er.config import Config
from visage_er.models.hybrid_model import HybridEmotionNet, create_hybrid_model


class HybridInference:
    """Real-time inference with hybrid CNN + Coordinate model"""

    FACE_CROP_SIZE = 112
    EMOTION_COLORS = {
        'Angry': (0, 0, 255),
        'Disgust': (0, 140, 255),
        'Fear': (180, 0, 180),
        'Happy': (0, 255, 0),
        'Neutral': (255, 200, 0),
        'Sad': (255, 0, 0),
        'Surprised': (0, 255, 255)
    }

    def __init__(self, model_path: str = None):
        self.device = Config.DEVICE

        # Load model
        if model_path is None:
            model_path = Config.MODELS_PATH / "weights" / "hybrid_best_model.pth"

        self.model = create_hybrid_model(pretrained_cnn=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        best_acc = checkpoint.get('val_acc', 'N/A')
        print(f"Hybrid model loaded | Best val acc: {best_acc}%")

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Scaler
        scaler_path = Config.MODELS_PATH / "scalers" / "hybrid_coordinate_scaler.pkl"
        self.scaler = None
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # ImageNet normalization
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Smoothing
        self.prob_history = deque(maxlen=8)
        self.ema_probs = None
        self.ema_alpha = 0.35

        self.use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()

    def process_frame(self, frame):
        """Process a single frame and return emotion prediction"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame, None, None

        landmarks = results.multi_face_landmarks[0]

        # Get face bounding box
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        face_w = x_max - x_min
        face_h = y_max - y_min

        if face_w < 30 or face_h < 30:
            return frame, None, None

        # Extract coordinates
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x * w, lm.y * h, lm.z * w])
        coords = np.array(coords, dtype=np.float32)

        coords_3d = coords.reshape(-1, 3)
        half_w, half_h = w / 2.0, h / 2.0
        coords_3d[:, 0] = (coords_3d[:, 0] - half_w) / half_w
        coords_3d[:, 1] = (coords_3d[:, 1] - half_h) / half_h
        coords_3d[:, 2] = coords_3d[:, 2] * 0.1
        coords = coords_3d.flatten()

        if self.scaler:
            try:
                coords = self.scaler.transform([coords])[0]
            except:
                pass

        if len(coords) < Config.COORDINATE_DIM:
            padded = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
            padded[:len(coords)] = coords
            coords = padded

        # Extract face crop
        pad_w = int(face_w * 0.2)
        pad_h = int(face_h * 0.2)
        crop_x1 = max(0, x_min - pad_w)
        crop_x2 = min(w, x_max + pad_w)
        crop_y1 = max(0, y_min - pad_h)
        crop_y2 = min(h, y_max + pad_h)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        face_crop = cv2.resize(face_crop, (self.FACE_CROP_SIZE, self.FACE_CROP_SIZE))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = face_crop.astype(np.float32) / 255.0
        face_crop = (face_crop - self.img_mean) / self.img_std
        face_crop = face_crop.transpose(2, 0, 1)

        # Inference
        coord_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        crop_tensor = torch.tensor(face_crop, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with autocast('cuda'):
                    output = self.model(coord_tensor, crop_tensor)
            else:
                output = self.model(coord_tensor, crop_tensor)

        probs = F.softmax(output, dim=1).cpu().numpy()[0]

        # Temporal smoothing (EMA + sliding window)
        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs

        self.prob_history.append(probs)
        avg_probs = np.mean(self.prob_history, axis=0)

        # Blend EMA and window average
        smoothed = 0.5 * self.ema_probs + 0.5 * avg_probs

        pred_idx = np.argmax(smoothed)
        confidence = smoothed[pred_idx]
        emotion = Config.EMOTION_CLASSES[pred_idx]

        # Draw visualization
        color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Face rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Emotion label
        label = f"{emotion} ({confidence*100:.0f}%)" if confidence > 0.35 else "Uncertain"
        label_color = color if confidence > 0.35 else (128, 128, 128)

        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        # Probability bars
        bar_x = 10
        bar_y = 30
        bar_w = 200
        bar_h = 18

        for i, em in enumerate(Config.EMOTION_CLASSES):
            p = smoothed[i]
            em_color = self.EMOTION_COLORS.get(em, (200, 200, 200))

            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y + i * 25),
                         (bar_x + bar_w, bar_y + i * 25 + bar_h), (50, 50, 50), -1)
            # Fill bar
            fill_w = int(bar_w * p)
            cv2.rectangle(frame, (bar_x, bar_y + i * 25),
                         (bar_x + fill_w, bar_y + i * 25 + bar_h), em_color, -1)
            # Label
            cv2.putText(frame, f"{em[:3]} {p*100:.0f}%",
                       (bar_x + bar_w + 5, bar_y + i * 25 + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame, emotion, confidence

    def run(self, camera_index: int = 0):
        """Run live inference"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Hybrid Emotion Recognition started. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, emotion, confidence = self.process_frame(frame)

            cv2.imshow('Hybrid Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()

    engine = HybridInference(model_path=args.model)
    engine.run(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
