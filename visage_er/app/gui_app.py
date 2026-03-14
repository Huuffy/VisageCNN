"""VisageCNN desktop application — real-time facial expression recognition with
session analytics, expression history logging, and CSV export.
"""

import warnings
import os

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from datetime import datetime
import pytz
import time
from collections import deque, Counter
import sys
import json
import csv
import pickle

sys.path.insert(0, str(__file__))

from ..config import Config
from ..models.hybrid_model import create_hybrid_model

ist = pytz.timezone('Asia/Kolkata')

prediction_history: deque = deque(maxlen=10)
fps_counter: deque = deque(maxlen=30)

camera = None
last_frame_time = 0.0
show_landmarks = False
dark_mode = False

session_stats = {
    'total_frames': 0,
    'emotions_detected': {emotion: 0 for emotion in Config.EMOTION_CLASSES},
}

inference_engine = None


class HybridInferenceEngine:
    """Inference engine that wraps HybridEmotionNet for single-frame prediction.

    Handles model loading, MediaPipe face mesh initialisation, coordinate
    extraction and normalisation, face crop preparation, and model inference.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = None
        self.face_mesh = None
        self.scaler = None
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.use_amp = Config.MIXED_PRECISION and torch.cuda.is_available()
        self._load_model()
        self._init_face_mesh()
        self._load_scaler()

    def _load_model(self):
        """Load HybridEmotionNet weights from the default checkpoint path."""
        model_path = Config.MODELS_PATH / "weights" / "hybrid_best_model.pth"
        try:
            self.model = create_hybrid_model(pretrained_cnn=False)
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception:
            self.model = None

    def _init_face_mesh(self):
        """Initialise the MediaPipe FaceMesh processor."""
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _load_scaler(self):
        """Load the RobustScaler fitted during training, if available."""
        scaler_path = Config.MODELS_PATH / "scalers" / "hybrid_coordinate_scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = None

    def predict(self, frame: np.ndarray):
        """Run hybrid emotion prediction on a single RGB frame.

        Args:
            frame: RGB image array (as produced by cv2.cvtColor BGR→RGB).

        Returns:
            Tuple of (top2, all_predictions) where top2 is a list of
            (emotion, confidence) pairs for the top-2 classes, and
            all_predictions is a float32 numpy array of length 7.
            Returns (None, None) when no face is detected or the model
            is not loaded.
        """
        if self.model is None:
            return None, None

        try:
            h, w = frame.shape[:2]
            results = self.face_mesh.process(frame)

            if not results.multi_face_landmarks:
                return None, None

            landmarks = results.multi_face_landmarks[0]

            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            if (x_max - x_min) < 30 or (y_max - y_min) < 30:
                return None, None

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

            if len(coords) < Config.COORDINATE_DIM:
                padded = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
                padded[:len(coords)] = coords
                coords = padded

            if self.scaler is not None:
                try:
                    coords = self.scaler.transform([coords])[0]
                except Exception:
                    pass

            face_w = x_max - x_min
            face_h = y_max - y_min
            pad_w = int(face_w * 0.2)
            pad_h = int(face_h * 0.2)
            crop_x1 = max(0, x_min - pad_w)
            crop_x2 = min(w, x_max + pad_w)
            crop_y1 = max(0, y_min - pad_h)
            crop_y2 = min(h, y_max + pad_h)

            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            face_crop = cv2.resize(face_crop, (Config.FACE_CROP_SIZE, Config.FACE_CROP_SIZE))
            face_crop = face_crop.astype(np.float32) / 255.0
            face_crop = (face_crop - self.img_mean) / self.img_std
            face_crop = face_crop.transpose(2, 0, 1)

            coord_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)
            crop_tensor = torch.tensor(face_crop, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(coord_tensor, crop_tensor)
                else:
                    output = self.model(coord_tensor, crop_tensor)

            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            top2_idx = np.argsort(probs)[-2:][::-1]
            top2 = [(Config.EMOTION_CLASSES[i], float(probs[i])) for i in top2_idx]

            return top2, probs

        except Exception:
            return None, None


def initialize_db():
    """Create the expressions and sessions tables if they do not exist."""
    conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expressions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expression TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_predictions TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            model_version TEXT,
            inference_time REAL,
            model_architecture TEXT,
            batch_size INTEGER,
            gpu_memory TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            start_time DATETIME,
            end_time DATETIME,
            total_frames INTEGER,
            dominant_emotion TEXT,
            avg_confidence REAL,
            avg_inference_time REAL,
            model_version TEXT,
            gpu_used TEXT
        )
    ''')

    conn.commit()
    conn.close()


def insert_expression(expression, confidence, all_predictions, session_id="current",
                      inference_time=0.0):
    """Insert a single prediction record into the expressions table.

    Args:
        expression: Predicted emotion label.
        confidence: Confidence score (0–1).
        all_predictions: Array of probabilities for all 7 classes.
        session_id: Identifier for the current session.
        inference_time: Wall-clock inference time in seconds.
    """
    conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
    cursor = conn.cursor()
    current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

    predictions_json = ','.join([
        f"{Config.EMOTION_CLASSES[i]}:{pred:.4f}"
        for i, pred in enumerate(all_predictions)
    ])

    gpu_info = "GPU" if torch.cuda.is_available() else "CPU"
    model_arch = "HybridEmotionNet (EfficientNet-B0 + MLP)"

    cursor.execute('''
        INSERT INTO expressions (expression, confidence, all_predictions, timestamp, session_id,
                               model_version, inference_time, model_architecture, batch_size, gpu_memory)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (expression, confidence, predictions_json, current_time, session_id,
          "hybrid_v3.0", inference_time, model_arch, Config.BATCH_SIZE, gpu_info))

    conn.commit()
    conn.close()


def view_data():
    """Open the analytics dashboard window."""
    conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM expressions ORDER BY timestamp DESC LIMIT 1000')
    rows = cursor.fetchall()
    conn.close()

    view_window = tk.Toplevel(window)
    view_window.title("VisageCNN Expression Analytics")
    view_window.geometry("1200x800")
    apply_theme(view_window)

    notebook = ttk.Notebook(view_window)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text="Recent Data")

    tree_frame = tk.Frame(data_frame)
    tree_frame.pack(expand=True, fill='both', padx=5, pady=5)

    columns = ('ID', 'Expression', 'Confidence', 'Timestamp', 'Model', 'Architecture',
               'Inference Time', 'GPU')
    tree = ttk.Treeview(tree_frame, columns=columns, show='headings')

    for col in columns:
        tree.heading(col, text=col)

    tree.column('ID', width=50, anchor='center')
    tree.column('Expression', width=100, anchor='center')
    tree.column('Confidence', width=100, anchor='center')
    tree.column('Timestamp', width=150, anchor='center')
    tree.column('Model', width=120, anchor='center')
    tree.column('Architecture', width=200, anchor='center')
    tree.column('Inference Time', width=100, anchor='center')
    tree.column('GPU', width=80, anchor='center')

    v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    tree.pack(expand=True, fill='both')

    for row in rows:
        display_row = list(row)
        if len(display_row) > 2:
            display_row[2] = f"{float(display_row[2]):.2%}"
        if len(display_row) > 7 and display_row[7]:
            display_row[7] = f"{float(display_row[7]) * 1000:.1f}ms"
        while len(display_row) < len(columns):
            display_row.append("N/A")
        tree.insert('', tk.END, values=display_row[:len(columns)])

    stats_frame = ttk.Frame(notebook)
    notebook.add(stats_frame, text="Analytics")
    if rows:
        create_statistics(stats_frame, rows)

    button_frame = tk.Frame(view_window)
    button_frame.pack(fill='x', padx=10, pady=5)

    ttk.Button(button_frame, text="Export CSV", command=export_data).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Trends", command=show_emotion_trends).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Model Info", command=show_model_info).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Clear Data",
               command=lambda: clear_data(view_window)).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Close", command=view_window.destroy).pack(side='right', padx=5)


def create_statistics(parent, rows):
    """Render analytics text into a scrollable text widget.

    Args:
        parent: Tkinter parent widget.
        rows: List of expression table rows.
    """
    text_frame = tk.Frame(parent)
    text_frame.pack(fill='both', expand=True, padx=10, pady=10)

    stats_text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=stats_text.yview)
    stats_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    stats_text.pack(fill='both', expand=True)

    emotions_count = {}
    confidence_by_emotion = {}
    inference_times = []

    for row in rows:
        emotion = row[1]
        confidence = float(row[2])

        if len(row) > 7 and row[7]:
            inference_times.append(float(row[7]))

        emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
        confidence_by_emotion.setdefault(emotion, []).append(confidence)

    content = "VISAGECNN ANALYTICS\n"
    content += "=" * 70 + "\n\n"
    content += f"Total Predictions:  {len(rows)}\n"
    content += f"Model:              HybridEmotionNet (EfficientNet-B0 + MLP)\n"
    content += f"Device:             {'GPU' if torch.cuda.is_available() else 'CPU'}\n\n"

    content += "EMOTION DISTRIBUTION\n" + "-" * 50 + "\n"
    for emotion, count in sorted(emotions_count.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(rows)) * 100
        avg_conf = np.mean(confidence_by_emotion[emotion]) * 100
        content += f"{emotion:<12} {count:4} ({pct:5.1f}%)  avg confidence: {avg_conf:.1f}%\n"

    content += "\nCONFIDENCE ANALYSIS\n" + "-" * 50 + "\n"
    all_confidences = [float(r[2]) for r in rows]
    content += f"Mean confidence:    {np.mean(all_confidences) * 100:.1f}%\n"
    content += f"Std deviation:      {np.std(all_confidences) * 100:.1f}%\n"
    content += f"High (>80%):        {sum(1 for c in all_confidences if c > 0.8)}/{len(all_confidences)}\n"

    if inference_times:
        content += "\nPERFORMANCE\n" + "-" * 50 + "\n"
        content += f"Avg inference:      {np.mean(inference_times) * 1000:.1f} ms\n"
        content += f"Min / Max:          {np.min(inference_times) * 1000:.1f} / {np.max(inference_times) * 1000:.1f} ms\n"
        avg_fps = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
        content += f"Average FPS:        {avg_fps:.1f}\n"

    content += "\nPER-CLASS CONFIDENCE\n" + "-" * 50 + "\n"
    for emotion, confs in sorted(confidence_by_emotion.items(),
                                  key=lambda x: np.mean(x[1]), reverse=True):
        avg = np.mean(confs)
        grade = "high" if avg > 0.8 else "medium" if avg > 0.6 else "low"
        content += f"[{grade}] {emotion:<12} {avg * 100:.1f}%\n"

    stats_text.insert(tk.END, content)
    stats_text.config(state=tk.DISABLED)


def show_model_info():
    """Open a window showing HybridEmotionNet architecture and system details."""
    info_window = tk.Toplevel(window)
    info_window.title("Model Information")
    info_window.geometry("600x600")
    apply_theme(info_window)

    text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
    text_widget.pack(fill='both', expand=True, padx=10, pady=10)

    device_info = Config.get_device_info()

    content = "VISAGECNN — HYBRID MODEL INFORMATION\n" + "=" * 60 + "\n\n"

    content += "ARCHITECTURE\n"
    content += "Model:              HybridEmotionNet\n"
    content += "CNN Branch:         EfficientNet-B0 (blocks 0-2 frozen)\n"
    content += "Coordinate Branch:  MLP (1434 → 512 → 384 → 256)\n"
    content += "Fusion:             Cross-attention + MLP (512 → 128 → 7)\n"
    content += f"Input Landmarks:    {Config.NUM_LANDMARKS} × 3D = {Config.COORDINATE_DIM} features\n"
    content += f"Face Crop Size:     {Config.FACE_CROP_SIZE} × {Config.FACE_CROP_SIZE}\n"
    content += f"Output Classes:     {Config.NUM_CLASSES}\n\n"

    content += "SYSTEM\n"
    if device_info['device'] == 'cuda':
        content += f"GPU:                {device_info['device_name']}\n"
        content += f"Total VRAM:         {device_info.get('memory_total', 'N/A')} GB\n"
    else:
        content += f"Device:             CPU ({device_info.get('cores', 'N/A')} cores)\n"
    content += f"Mixed Precision:    {Config.MIXED_PRECISION}\n"
    content += f"Batch Size:         {Config.BATCH_SIZE}\n\n"

    content += "MODEL STATUS\n"
    model_status = "Loaded" if inference_engine and inference_engine.model else "Not loaded"
    scaler_status = "Available" if inference_engine and inference_engine.scaler else "Not available"
    content += f"Model weights:      {model_status}\n"
    content += f"Coordinate scaler:  {scaler_status}\n"
    content += f"Device:             {Config.DEVICE}\n"

    if inference_engine and inference_engine.model:
        try:
            total = sum(p.numel() for p in inference_engine.model.parameters())
            trainable = sum(p.numel() for p in inference_engine.model.parameters()
                            if p.requires_grad)
            content += f"Total parameters:   {total:,}\n"
            content += f"Trainable params:   {trainable:,}\n"
        except Exception:
            pass

    text_widget.insert(tk.END, content)
    text_widget.config(state=tk.DISABLED)
    ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)


def export_data():
    """Export all expression records to a user-selected CSV file."""
    try:
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM expressions ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ID', 'Expression', 'Confidence', 'All_Predictions', 'Timestamp',
                'Session', 'Model_Version', 'Inference_Time', 'Architecture',
                'Batch_Size', 'GPU_Memory',
            ])
            writer.writerows(rows)

        messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")
    except Exception as e:
        messagebox.showerror("Export Failed", str(e))


def generate_report():
    """Open a comprehensive analytics window for all stored predictions."""
    try:
        conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM expressions ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            messagebox.showinfo("No Data", "No predictions recorded yet.")
            return

        report_window = tk.Toplevel(window)
        report_window.title("Analytics Report")
        report_window.geometry("900x700")
        apply_theme(report_window)
        create_statistics(report_window, rows)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def clear_data(parent_window):
    """Delete all stored expression and session records after user confirmation.

    Args:
        parent_window: Parent window for the confirmation dialog.
    """
    if not messagebox.askyesno(
        "Confirm", "Delete ALL stored data? This cannot be undone.",
        parent=parent_window,
    ):
        return
    try:
        conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
        cursor = conn.cursor()
        cursor.execute('DELETE FROM expressions')
        cursor.execute('DELETE FROM sessions')
        conn.commit()
        conn.close()
        messagebox.showinfo("Done", "All data cleared.", parent=parent_window)
        parent_window.destroy()
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=parent_window)


def predict_expression(image: np.ndarray):
    """Run inference and return smoothed top-2 predictions.

    Args:
        image: RGB frame array.

    Returns:
        Tuple of (top2 list, all_predictions array, inference_time seconds).
    """
    try:
        start = time.time()
        top2, all_predictions = inference_engine.predict(image)
        inference_time = time.time() - start

        if top2 is None:
            return [], [], inference_time

        if all_predictions is not None:
            all_predictions = smooth_predictions(all_predictions)
            top2_idx = np.argsort(all_predictions)[-2:][::-1]
            top2 = [(Config.EMOTION_CLASSES[i], float(all_predictions[i])) for i in top2_idx]

        threshold = 0.25
        filtered = [(e, c) for e, c in top2 if c >= threshold]
        if not filtered:
            filtered = [top2[0]] if top2 else []

        return filtered, all_predictions, inference_time

    except Exception:
        return [], [], 0.0


def smooth_predictions(predictions: np.ndarray) -> np.ndarray:
    """Apply exponential-weighted averaging over the recent prediction history.

    Args:
        predictions: Raw softmax probability array of length 7.

    Returns:
        Smoothed probability array.
    """
    prediction_history.append(predictions)

    if len(prediction_history) > 1:
        weights = np.exp(np.linspace(-2, 0, len(prediction_history)))
        weights /= weights.sum()
        smoothed = sum(p * w for p, w in zip(prediction_history, weights))
        return smoothed

    return predictions


def calculate_fps() -> float:
    """Compute current frames-per-second from the rolling timestamp buffer.

    Returns:
        Estimated FPS, or 0 if insufficient data.
    """
    fps_counter.append(time.time())
    if len(fps_counter) > 1:
        elapsed = fps_counter[-1] - fps_counter[0]
        if elapsed > 0:
            return (len(fps_counter) - 1) / elapsed
    return 0.0


def update_session_stats():
    """Refresh the session statistics label at the bottom of the camera view."""
    if hasattr(window, 'stats_label'):
        total = session_stats['total_frames']
        if total > 0:
            dominant = max(session_stats['emotions_detected'],
                           key=session_stats['emotions_detected'].get)
            pct = session_stats['emotions_detected'][dominant] / total * 100
            text = f"Frames: {total}  |  Dominant: {dominant} ({pct:.1f}%)"
        else:
            text = "Frames: 0  |  Dominant: –"
        window.stats_label.config(text=text)


def show_camera():
    """Read one webcam frame, run inference, render overlay, and schedule next call."""
    global camera, last_frame_time

    if camera is None or not camera.isOpened():
        messagebox.showerror("Error", "Camera is not open.")
        return

    now = time.time()
    if now - last_frame_time < 1.0 / 30:
        window.after(1, show_camera)
        return
    last_frame_time = now

    ret, frame = camera.read()
    if not ret:
        camera_label.after(33, show_camera)
        return

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if show_landmarks:
        frame_rgb = draw_landmarks(frame_rgb)

    top2, all_predictions, inference_time = predict_expression(frame_rgb)

    if top2:
        session_stats['total_frames'] += 1
        session_stats['emotions_detected'][top2[0][0]] += 1

        current_fps = calculate_fps()

        overlay = frame_rgb.copy()
        cv2.rectangle(overlay, (5, 5), (560, 185), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_rgb, 0.3, 0, frame_rgb)

        for i, (emotion, confidence) in enumerate(top2):
            color = (0, 255, 0) if i == 0 else (255, 255, 0)
            grade = (
                "Excellent" if confidence >= 0.9 else
                "Very Good" if confidence >= 0.8 else
                "Good" if confidence >= 0.6 else
                "Fair" if confidence >= 0.4 else
                "Poor"
            )
            cv2.putText(frame_rgb, f"{i + 1}. {emotion}: {confidence:.1%}",
                        (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame_rgb, f"   {grade}",
                        (10, 50 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame_rgb, "HybridEmotionNet",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_rgb, f"FPS: {current_fps:.1f}  |  {inference_time * 1000:.1f}ms",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, (_, confidence) in enumerate(top2):
            bar_w = int(confidence * 250)
            color = (0, 255, 0) if i == 0 else (255, 255, 0)
            cv2.rectangle(frame_rgb, (310, 30 + i * 35), (310 + bar_w, 45 + i * 35), color, -1)

    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    camera_label.config(image=img)
    camera_label.image = img

    update_session_stats()
    camera_label.after(33, show_camera)


def draw_landmarks(frame: np.ndarray) -> np.ndarray:
    """Draw MediaPipe facial landmark points onto the frame.

    Args:
        frame: RGB image array.

    Returns:
        Frame with landmark dots rendered.
    """
    try:
        results = inference_engine.face_mesh.process(frame)
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            for lm in results.multi_face_landmarks[0].landmark:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
    except Exception:
        pass
    return frame


def capture_image_frame():
    """Capture the current webcam frame, run inference, and open the analysis window."""
    global camera

    if camera is None or not camera.isOpened():
        messagebox.showerror("Error", "Camera not available.")
        return

    ret, frame = camera.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture frame.")
        return

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    top2, all_predictions, inference_time = predict_expression(frame_rgb)

    if top2:
        insert_expression(top2[0][0], top2[0][1], all_predictions, inference_time=inference_time)
        display_capture_analysis(frame_rgb, top2, all_predictions, inference_time)
    else:
        messagebox.showinfo("No Face", "No face detected in this frame.")


def display_capture_analysis(frame, top2, predictions, inference_time):
    """Open a detailed analysis window for a captured frame.

    Args:
        frame: RGB image array of the captured frame.
        top2: Top-2 (emotion, confidence) pairs.
        predictions: Full probability array.
        inference_time: Inference duration in seconds.
    """
    result_window = tk.Toplevel(window)
    result_window.title("Expression Analysis")
    result_window.geometry("900x1000")
    apply_theme(result_window)

    img = Image.fromarray(frame).resize((450, 350), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(tk.Frame(result_window), image=img_tk)
    img_label.image = img_tk
    img_label.pack(padx=10, pady=10)

    result_frame = tk.Frame(result_window)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    info_frame = tk.Frame(result_frame)
    info_frame.pack(fill='x', pady=5)
    tk.Label(info_frame, text="HybridEmotionNet", font=('Helvetica', 12, 'bold')).pack(side='left')
    tk.Label(info_frame, text="EfficientNet-B0 + MLP", font=('Helvetica', 10)).pack(side='left', padx=10)
    tk.Label(info_frame, text=f"{inference_time * 1000:.1f}ms", font=('Helvetica', 10)).pack(side='right')

    top_frame = tk.LabelFrame(result_frame, text="Top Predictions", padx=10, pady=10)
    top_frame.pack(fill='x', pady=5)

    for i, (emotion, confidence) in enumerate(top2):
        row = tk.Frame(top_frame)
        row.pack(fill='x', pady=3)
        tk.Label(row, text=f"#{i + 1}", font=('Helvetica', 14, 'bold'), width=3).pack(side='left')
        tk.Label(row, text=emotion, font=('Helvetica', 12, 'bold'), width=12, anchor='w').pack(side='left')
        color = 'green' if confidence > 0.8 else 'blue' if confidence > 0.6 else 'orange' if confidence > 0.4 else 'red'
        grade = "Excellent" if confidence > 0.8 else "Very Good" if confidence > 0.6 else "Good" if confidence > 0.4 else "Uncertain"
        tk.Label(row, text=f"{confidence:.2%}", font=('Helvetica', 12), fg=color, width=8).pack(side='left')
        tk.Label(row, text=grade, font=('Helvetica', 10)).pack(side='left', padx=10)

    all_frame = tk.LabelFrame(result_frame, text="All Emotions", padx=10, pady=10)
    all_frame.pack(fill='both', expand=True, pady=5)

    canvas = tk.Canvas(all_frame, height=350, bg='white')
    scrollbar = ttk.Scrollbar(all_frame, orient="vertical", command=canvas.yview)
    scrollable = ttk.Frame(canvas)
    scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for i, emotion in enumerate(Config.EMOTION_CLASSES):
        conf = float(predictions[i])
        row = tk.Frame(scrollable)
        row.pack(fill='x', pady=4, padx=5)
        tk.Label(row, text=emotion, width=15, anchor='w', font=('Helvetica', 11, 'bold')).pack(side='left')
        ttk.Progressbar(row, length=250, value=conf * 100).pack(side='left', padx=10)
        color = 'green' if conf > 0.8 else 'blue' if conf > 0.6 else 'orange' if conf > 0.4 else 'red'
        tk.Label(row, text=f"{conf:.2%}", width=8, fg=color, font=('Helvetica', 11, 'bold')).pack(side='left', padx=5)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = tk.Frame(result_window)
    btn_frame.pack(fill='x', padx=10, pady=10)
    ttk.Button(btn_frame, text="Save Analysis",
               command=lambda: save_analysis(predictions, inference_time)).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Trends", command=show_emotion_trends).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Model Info", command=show_model_info).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Close", command=result_window.destroy).pack(side='right', padx=5)


def save_analysis(predictions, inference_time):
    """Prompt the user to save the current prediction analysis as JSON.

    Args:
        predictions: Full probability array.
        inference_time: Inference duration in seconds.
    """
    try:
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        data = {
            'timestamp': datetime.now(ist).isoformat(),
            'model': 'HybridEmotionNet',
            'architecture': 'EfficientNet-B0 + MLP Coordinate Encoder + Cross-Attention Fusion',
            'performance': {
                'inference_time_ms': inference_time * 1000,
                'device': str(Config.DEVICE),
                'mixed_precision': Config.MIXED_PRECISION,
            },
            'predictions': {e: float(predictions[i]) for i, e in enumerate(Config.EMOTION_CLASSES)},
            'top_emotion': Config.EMOTION_CLASSES[int(np.argmax(predictions))],
            'confidence': float(np.max(predictions)),
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Saved", f"Analysis saved to:\n{filename}")
    except Exception as e:
        messagebox.showerror("Save Failed", str(e))


def show_emotion_trends():
    """Open a window showing emotion distribution from the last 20 predictions."""
    try:
        conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
        cursor = conn.cursor()
        cursor.execute(
            'SELECT expression, timestamp, confidence FROM expressions '
            'ORDER BY timestamp DESC LIMIT 100'
        )
        recent = cursor.fetchall()
        conn.close()

        if not recent:
            messagebox.showinfo("No Data", "No predictions recorded yet.")
            return

        trends_window = tk.Toplevel(window)
        trends_window.title("Emotion Trends")
        trends_window.geometry("700x500")
        apply_theme(trends_window)

        text_widget = tk.Text(trends_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)

        sample = recent[:20]
        emotions = [r[0] for r in sample]
        confidences = [float(r[2]) for r in sample]
        counts = Counter(emotions)

        content = "EMOTION TRENDS (Last 20 Predictions)\n"
        content += "=" * 60 + "\n\n"
        content += "DISTRIBUTION\n"
        for emotion, count in counts.most_common():
            pct = count / len(emotions) * 100
            avg = np.mean([float(r[2]) for r in sample if r[0] == emotion])
            bar = "#" * int(pct / 3)
            content += f"{emotion:<10} {bar:<20} {pct:5.1f}%  avg conf: {avg:.1%}\n"

        content += f"\nQUALITY\n"
        content += f"Avg confidence:  {np.mean(confidences):.1%}\n"
        content += f"High (>80%):     {sum(1 for c in confidences if c > 0.8)}/{len(confidences)}\n"
        content += f"Std deviation:   {np.std(confidences):.1%}\n"

        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def toggle_theme():
    """Toggle between light and dark UI themes."""
    global dark_mode
    dark_mode = not dark_mode
    apply_theme(window)
    theme_btn.config(text="Light Mode" if dark_mode else "Dark Mode")


def apply_theme(widget):
    """Recursively apply the current colour theme to a widget and its children.

    Args:
        widget: Tkinter widget to theme.
    """
    bg = '#2b2b2b' if dark_mode else '#e0f7fa'
    fg = '#ffffff' if dark_mode else '#000000'
    try:
        widget.configure(bg=bg)
        for child in widget.winfo_children():
            if isinstance(child, (tk.Label, tk.Frame, tk.Text)):
                try:
                    child.configure(bg=bg, fg=fg)
                except Exception:
                    pass
            apply_theme(child)
    except Exception:
        pass


def toggle_landmarks():
    """Toggle facial landmark overlay on the camera feed."""
    global show_landmarks
    show_landmarks = not show_landmarks
    landmarks_btn.config(text="Hide Landmarks" if show_landmarks else "Show Landmarks")


def show_help():
    """Open the help and usage guide window."""
    help_window = tk.Toplevel(window)
    help_window.title("Help Guide")
    help_window.geometry("800x650")
    apply_theme(help_window)

    text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
    text_widget.pack(fill='both', expand=True, padx=10, pady=10)

    content = (
        "VISAGECNN — HYBRID EMOTION RECOGNITION\n\n"
        "ARCHITECTURE\n"
        "HybridEmotionNet combines two parallel feature streams:\n"
        "  • EfficientNet-B0 CNN branch — extracts appearance cues from a 224×224 face crop\n"
        "  • MLP Coordinate branch — encodes 478 3D MediaPipe landmarks (1,434 features)\n"
        "Both streams are fused via cross-attention and a 3-layer MLP classifier.\n\n"
        "CONTROLS\n"
        "  Access Camera       Start live emotion detection\n"
        "  Capture & Analyse   Detailed single-frame analysis with all 7 classes\n"
        "  Show/Hide Landmarks Toggle MediaPipe landmark overlay\n"
        "  Analytics Dashboard View prediction history and confidence statistics\n"
        "  Model Info          Architecture and system details\n"
        "  Dark/Light Mode     Switch UI colour theme\n\n"
        "TIPS FOR BEST RESULTS\n"
        "  • Ensure even, front-facing lighting\n"
        "  • Keep face centred and at a comfortable distance\n"
        "  • Avoid rapid head movements for more stable predictions\n"
        "  • Check confidence scores — values below 35% are shown as Uncertain\n\n"
        "TECHNICAL DETAILS\n"
        f"  Backbone:           EfficientNet-B0 (ImageNet pre-trained, top-3 blocks frozen)\n"
        f"  Landmark input:     {Config.COORDINATE_DIM} features ({Config.NUM_LANDMARKS} landmarks × 3)\n"
        f"  Face crop size:     {Config.FACE_CROP_SIZE} × {Config.FACE_CROP_SIZE}\n"
        f"  Device:             {Config.DEVICE}\n"
        f"  Mixed precision:    {Config.MIXED_PRECISION}\n"
        "  Storage:            SQLite (logs/expressions.db)\n"
    )

    text_widget.insert(tk.END, content)
    text_widget.config(state=tk.DISABLED)
    ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)


def terminate_program():
    """Release the camera and close the application."""
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    window.destroy()


def switch_to_screen2():
    """Navigate to the camera view."""
    screen1_frame.grid_forget()
    screen2_frame.grid(row=0, column=0, sticky="nsew")
    update_session_stats()


def switch_to_screen1():
    """Navigate back to the main menu and release the camera."""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    screen2_frame.grid_forget()
    screen1_frame.grid(row=0, column=0, sticky="nsew")


def open_camera():
    """Try each camera index in turn and start the live feed on success."""
    global camera
    try:
        for index in [0, 1, 2]:
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                break
        else:
            raise ValueError("No camera found.")

        show_camera()
        switch_to_screen2()

    except Exception as e:
        messagebox.showerror("Camera Error", str(e))


def main():
    """Initialise the database, build the inference engine, and start the GUI."""
    global window, camera_label, theme_btn, landmarks_btn
    global screen1_frame, screen2_frame, inference_engine

    initialize_db()
    inference_engine = HybridInferenceEngine()

    window = tk.Tk()
    window.title("VisageCNN — Hybrid Emotion Recognition")
    window.geometry("1000x750")

    window.update_idletasks()
    x = (window.winfo_screenwidth() - 1000) // 2
    y = (window.winfo_screenheight() - 750) // 2
    window.geometry(f"1000x750+{x}+{y}")

    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 11), padding=8)
    style.configure('TLabel', font=('Helvetica', 11))

    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)

    screen1_frame = tk.Frame(window, bg='#e0f7fa')
    screen1_frame.grid(row=0, column=0, sticky="nsew")

    tk.Label(screen1_frame, text="VisageCNN",
             font=('Helvetica', 26, 'bold'), bg='#e0f7fa', pady=15).pack(pady=20)

    tk.Label(screen1_frame, text="Hybrid CNN + Landmark Emotion Recognition",
             font=('Helvetica', 14), bg='#e0f7fa').pack()

    tk.Label(screen1_frame,
             text="EfficientNet-B0 × MediaPipe FaceMesh (478 landmarks)",
             font=('Helvetica', 10, 'italic'), bg='#e0f7fa', fg='#666666').pack(pady=5)

    status_frame = tk.Frame(screen1_frame, bg='#e0f7fa')
    status_frame.pack(pady=10)

    model_ok = inference_engine.model is not None
    scaler_ok = inference_engine.scaler is not None

    tk.Label(status_frame,
             text="Model: Ready" if model_ok else "Model: Not Found",
             font=('Helvetica', 10),
             fg='green' if model_ok else 'red',
             bg='#e0f7fa').pack()

    tk.Label(status_frame,
             text="Scaler: Ready" if scaler_ok else "Scaler: Not Found",
             font=('Helvetica', 10),
             fg='green' if scaler_ok else 'orange',
             bg='#e0f7fa').pack()

    tk.Label(status_frame,
             text=f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}",
             font=('Helvetica', 10),
             fg='blue' if torch.cuda.is_available() else 'orange',
             bg='#e0f7fa').pack()

    button_frame = tk.Frame(screen1_frame, bg='#e0f7fa')
    button_frame.pack(pady=40)

    ttk.Button(button_frame, text="Access Camera", command=open_camera).pack(pady=15, ipadx=30)
    ttk.Button(button_frame, text="Analytics Dashboard", command=view_data).pack(pady=10, ipadx=30)

    secondary_frame = tk.Frame(screen1_frame, bg='#e0f7fa')
    secondary_frame.pack(pady=20)

    theme_btn = ttk.Button(secondary_frame, text="Dark Mode", command=toggle_theme)
    theme_btn.pack(side='left', padx=5)

    ttk.Button(secondary_frame, text="Model Info", command=show_model_info).pack(side='left', padx=5)
    ttk.Button(secondary_frame, text="Help Guide", command=show_help).pack(side='left', padx=5)

    ttk.Button(screen1_frame, text="Exit", command=terminate_program).pack(side=tk.BOTTOM, pady=20)

    screen2_frame = tk.Frame(window, bg='#e0f7fa')

    camera_label = tk.Label(screen2_frame, bg='#000000', text="Camera Loading…",
                             fg='white', font=('Helvetica', 16))
    camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    control_frame = tk.Frame(screen2_frame, bg='#e0f7fa')
    control_frame.pack(fill=tk.X, pady=5)

    left_controls = tk.Frame(control_frame, bg='#e0f7fa')
    left_controls.pack(side='left', padx=10)

    ttk.Button(left_controls, text="Capture & Analyse",
               command=capture_image_frame).pack(side='left', padx=5)

    landmarks_btn = ttk.Button(left_controls, text="Show Landmarks", command=toggle_landmarks)
    landmarks_btn.pack(side='left', padx=5)

    ttk.Button(left_controls, text="Model Info", command=show_model_info).pack(side='left', padx=5)

    right_controls = tk.Frame(control_frame, bg='#e0f7fa')
    right_controls.pack(side='right', padx=10)

    ttk.Button(right_controls, text="Main Menu", command=switch_to_screen1).pack(side='right', padx=5)

    status_bar = tk.Frame(screen2_frame, bg='#e0f7fa')
    status_bar.pack(fill='x', pady=5)

    window.stats_label = tk.Label(status_bar, text="Frames: 0  |  Dominant: –",
                                   bg='#e0f7fa', font=('Helvetica', 10))
    window.stats_label.pack()

    switch_to_screen1()
    apply_theme(window)
    window.mainloop()


if __name__ == "__main__":
    main()
