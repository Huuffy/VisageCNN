import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from datetime import datetime
import pytz
import os
import time
from collections import deque
import sys
import json
import csv
from tkinter import filedialog
from collections import Counter
import pickle

# Import from package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ..config import Config
from ..models.enhanced_model import EnhancedModelUtils, create_enhanced_model
from ..core.face_processor import EnhancedFaceMeshProcessor
from ..data.processor import AdvancedEmotionDataset

# Global variables
pytorch_model = None
inference_engine = None
ist = pytz.timezone('Asia/Kolkata')
camera = None
last_frame_time = 0
prediction_history = deque(maxlen=10)
show_landmarks = False
dark_mode = False
fps_counter = deque(maxlen=30)
session_stats = {
    'total_frames': 0,
    'emotions_detected': {emotion: 0 for emotion in Config.EMOTION_CLASSES}
}

class EnhancedPyTorchInference:
    """PyTorch inference engine for VisageCNN"""

    def __init__(self):
        self.pytorch_model = None
        self.face_processor = EnhancedFaceMeshProcessor()
        self.scaler = None
        self.device = Config.DEVICE
        self.load_model()
        self.load_scaler()

    def load_model(self):
        """Load PyTorch model"""
        try:
            model_path = Config.MODELS_PATH / "enhanced_best_model.pth"
            if model_path.exists():
                self.pytorch_model, checkpoint = EnhancedModelUtils.load_enhanced_model(
                    model_path, self.device
                )
                self.pytorch_model.eval()
            else:
                self.pytorch_model = create_enhanced_model()
                self.pytorch_model.eval()
        except Exception as e:
            try:
                self.pytorch_model = create_enhanced_model()
                self.pytorch_model.eval()
            except Exception as e2:
                self.pytorch_model = None

    def load_scaler(self):
        """Load coordinate scaler"""
        try:
            scaler_path = Config.MODELS_PATH / "enhanced_coordinate_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    if isinstance(scaler_data, dict):
                        self.scaler = scaler_data.get('scaler')
                    else:
                        self.scaler = scaler_data
            else:
                self.scaler = None
        except Exception as e:
            self.scaler = None

    def predict(self, frame):
        """Prediction method using PyTorch architecture"""
        if self.pytorch_model is None:
            return None, None

        try:
            coordinates = self.face_processor.extract_coordinates_from_frame_enhanced(frame)

            if coordinates is None:
                return None, None

            normalized_coords = self.face_processor.normalize_coordinates_enhanced(coordinates)

            if normalized_coords is None:
                return None, None

            # Ensure consistent dimensions (1434 features)
            if len(normalized_coords) != Config.COORDINATE_DIM:
                if len(normalized_coords) > Config.COORDINATE_DIM:
                    normalized_coords = normalized_coords[:Config.COORDINATE_DIM]
                else:
                    padded = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
                    padded[:len(normalized_coords)] = normalized_coords
                    normalized_coords = padded

            # Apply scaling if available
            if self.scaler is not None:
                try:
                    normalized_coords = self.scaler.transform([normalized_coords])[0]
                except Exception:
                    pass

            # Model prediction
            with torch.no_grad():
                tensor_coords = torch.tensor(normalized_coords, dtype=torch.float32).unsqueeze(0)
                tensor_coords = tensor_coords.to(self.device)

                if Config.MIXED_PRECISION:
                    with torch.amp.autocast('cuda'):
                        outputs = self.pytorch_model(tensor_coords)
                else:
                    outputs = self.pytorch_model(tensor_coords)

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions = probabilities.cpu().numpy()[0]

                # Get top 2 predictions
                top2_idx = np.argsort(predictions)[-2:][::-1]
                top2 = [(Config.EMOTION_CLASSES[i], float(predictions[i])) for i in top2_idx]

                return top2, predictions

        except Exception:
            return None, None

# Initialize PyTorch inference
pytorch_inference = EnhancedPyTorchInference()

def initialize_db():
    """Initialize SQLite database for storing predictions"""
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

def insert_expression(expression, confidence, all_predictions, session_id="current", inference_time=0.0):
    """Insert prediction into database"""
    conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
    cursor = conn.cursor()
    current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

    predictions_json = ','.join([f"{Config.EMOTION_CLASSES[i]}:{pred:.4f}"
                                for i, pred in enumerate(all_predictions)])

    gpu_info = f"GPU {Config.CUDA_MEMORY_FRACTION*100:.0f}%" if torch.cuda.is_available() else "CPU"
    model_arch = f"Model-{Config.HIDDEN_SIZE}D-{Config.NUM_LAYERS}L-{Config.NUM_HEADS}H"

    cursor.execute('''
        INSERT INTO expressions (expression, confidence, all_predictions, timestamp, session_id,
                               model_version, inference_time, model_architecture, batch_size, gpu_memory)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (expression, confidence, predictions_json, current_time, session_id,
          "pytorch_v2.0", inference_time, model_arch, Config.BATCH_SIZE, gpu_info))

    conn.commit()
    conn.close()

def view_data():
    """Display stored predictions with analytics"""
    conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM expressions ORDER BY timestamp DESC LIMIT 1000')
    rows = cursor.fetchall()
    conn.close()

    view_window = tk.Toplevel(window)
    view_window.title("VisageCNN Expression Data Analytics")
    view_window.geometry("1200x800")
    apply_theme(view_window)

    # Create notebook for tabs
    notebook = ttk.Notebook(view_window)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Data tab
    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text="Recent Data")

    # Create treeview
    tree_frame = tk.Frame(data_frame)
    tree_frame.pack(expand=True, fill='both', padx=5, pady=5)

    columns = ('ID', 'Expression', 'Confidence', 'Timestamp', 'Model', 'Architecture', 'Inference Time', 'GPU')
    tree = ttk.Treeview(tree_frame, columns=columns, show='headings')

    # Define headings and column widths
    tree.heading('ID', text='ID')
    tree.heading('Expression', text='Expression')
    tree.heading('Confidence', text='Confidence (%)')
    tree.heading('Timestamp', text='Timestamp')
    tree.heading('Model', text='Model Version')
    tree.heading('Architecture', text='Architecture')
    tree.heading('Inference Time', text='Inference (ms)')
    tree.heading('GPU', text='GPU Usage')

    tree.column('ID', width=50, anchor='center')
    tree.column('Expression', width=100, anchor='center')
    tree.column('Confidence', width=100, anchor='center')
    tree.column('Timestamp', width=150, anchor='center')
    tree.column('Model', width=120, anchor='center')
    tree.column('Architecture', width=150, anchor='center')
    tree.column('Inference Time', width=100, anchor='center')
    tree.column('GPU', width=100, anchor='center')

    # Add scrollbars
    v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    tree.pack(expand=True, fill='both')

    # Add data to treeview
    for row in rows:
        display_row = list(row)
        if len(display_row) > 2:
            display_row[2] = f"{float(display_row[2]):.2%}"
        if len(display_row) > 7:
            display_row[7] = f"{float(display_row[7])*1000:.1f}" if display_row[7] else "N/A"

        while len(display_row) < len(columns):
            display_row.append("N/A")

        tree.insert('', tk.END, values=display_row[:len(columns)])

    # Statistics tab
    stats_frame = ttk.Frame(notebook)
    notebook.add(stats_frame, text="Analytics")

    if rows:
        create_statistics(stats_frame, rows)

    # Button frame
    button_frame = tk.Frame(view_window)
    button_frame.pack(fill='x', padx=10, pady=5)

    ttk.Button(button_frame, text="Export CSV", command=lambda: export_data()).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Generate Report", command=lambda: generate_report()).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Model Info", command=lambda: show_model_info()).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Clear Data", command=lambda: clear_data(view_window)).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Close", command=view_window.destroy).pack(side='right', padx=5)

def create_statistics(parent, rows):
    """Create statistics display"""
    text_frame = tk.Frame(parent)
    text_frame.pack(fill='both', expand=True, padx=10, pady=10)

    stats_text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=stats_text.yview)
    stats_text.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    stats_text.pack(fill='both', expand=True)

    # Calculate statistics
    emotions_count = {}
    confidence_by_emotion = {}
    inference_times = []
    model_versions = {}
    architectures = {}

    for row in rows:
        emotion = row[1]
        confidence = float(row[2])

        if len(row) > 7 and row[7]:
            inference_times.append(float(row[7]))

        if len(row) > 6 and row[6]:
            model_versions[row[6]] = model_versions.get(row[6], 0) + 1

        if len(row) > 8 and row[8]:
            architectures[row[8]] = architectures.get(row[8], 0) + 1

        emotions_count[emotion] = emotions_count.get(emotion, 0) + 1

        if emotion not in confidence_by_emotion:
            confidence_by_emotion[emotion] = []
        confidence_by_emotion[emotion].append(confidence)

    # Generate report
    stats_content = f"VISAGECNN ANALYTICS REPORT\n"
    stats_content += "=" * 70 + "\n\n"

    stats_content += f"MODEL OVERVIEW\n"
    stats_content += f"Total Predictions: {len(rows)}\n"
    stats_content += f"Framework: PyTorch\n"
    stats_content += f"Architecture: {Config.HIDDEN_SIZE}D Hidden, {Config.NUM_LAYERS} Layers, {Config.NUM_HEADS} Heads\n"
    stats_content += f"Batch Size: {Config.BATCH_SIZE}\n"
    stats_content += f"Mixed Precision: {Config.MIXED_PRECISION}\n\n"

    stats_content += f"EMOTION DISTRIBUTION\n"
    stats_content += "-" * 50 + "\n"
    for emotion, count in sorted(emotions_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(rows)) * 100
        if emotion in confidence_by_emotion:
            avg_confidence = np.mean(confidence_by_emotion[emotion]) * 100
            stats_content += f"{emotion.capitalize():12} {count:4} ({percentage:5.1f}%) | Avg Conf: {avg_confidence:5.1f}%\n"

    stats_content += f"\nCONFIDENCE ANALYSIS\n"
    stats_content += "-" * 50 + "\n"
    all_confidences = [float(row[2]) for row in rows]
    stats_content += f"Overall Avg Confidence: {np.mean(all_confidences)*100:.1f}%\n"
    stats_content += f"Confidence Std Dev: {np.std(all_confidences)*100:.1f}%\n"
    stats_content += f"High Confidence (>80%): {sum(1 for c in all_confidences if c > 0.8)}/{len(all_confidences)}\n"
    stats_content += f"Excellent Confidence (>90%): {sum(1 for c in all_confidences if c > 0.9)}/{len(all_confidences)}\n"

    # Performance analysis
    if inference_times:
        stats_content += f"\nPERFORMANCE ANALYSIS\n"
        stats_content += "-" * 50 + "\n"
        stats_content += f"Avg Inference Time: {np.mean(inference_times)*1000:.1f}ms\n"
        stats_content += f"Min Inference Time: {np.min(inference_times)*1000:.1f}ms\n"
        stats_content += f"Max Inference Time: {np.max(inference_times)*1000:.1f}ms\n"
        stats_content += f"Std Dev: {np.std(inference_times)*1000:.1f}ms\n"

        avg_fps = 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
        stats_content += f"Average FPS: {avg_fps:.1f}\n"

    # Top performing emotions
    stats_content += f"\nTOP PERFORMING EMOTIONS\n"
    stats_content += "-" * 50 + "\n"
    emotion_performance = [(emotion, np.mean(confidences))
                          for emotion, confidences in confidence_by_emotion.items()]
    emotion_performance.sort(key=lambda x: x[1], reverse=True)

    for emotion, avg_conf in emotion_performance:
        grade = "[HIGH]" if avg_conf > 0.8 else "[MED]" if avg_conf > 0.6 else "[LOW]"
        stats_content += f"{grade} {emotion.capitalize():12} {avg_conf*100:5.1f}% average confidence\n"

    stats_text.insert(tk.END, stats_content)
    stats_text.config(state=tk.DISABLED)

def show_model_info():
    """Show detailed model information"""
    info_window = tk.Toplevel(window)
    info_window.title("Model Information")
    info_window.geometry("600x700")
    apply_theme(info_window)

    text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
    text_widget.pack(fill='both', expand=True, padx=10, pady=10)

    device_info = Config.get_device_info()

    info_content = f"VISAGECNN MODEL INFORMATION\n"
    info_content += "=" * 60 + "\n\n"

    info_content += f"MODEL ARCHITECTURE\n"
    info_content += f"Framework: PyTorch\n"
    info_content += f"Model Type: CoordinateEmotionNet\n"
    info_content += f"Input Features: {Config.FEATURE_SIZE} (1434 coordinates)\n"
    info_content += f"Hidden Size: {Config.HIDDEN_SIZE}\n"
    info_content += f"Number of Layers: {Config.NUM_LAYERS}\n"
    info_content += f"Attention Heads: {Config.NUM_HEADS}\n"
    info_content += f"Dropout Rate: {Config.DROPOUT_RATE}\n"
    info_content += f"Output Classes: {Config.NUM_CLASSES}\n\n"

    info_content += f"MODEL FEATURES\n"
    info_content += f"Multi-Head Attention: Yes\n"
    info_content += f"Geometric Features: Yes ({Config.GEOMETRIC_FEATURE_DIM}D)\n"
    info_content += f"Expert Networks: Yes ({Config.NUM_EXPERTS} experts)\n"
    info_content += f"Residual Connections: Yes\n"
    info_content += f"Layer Normalization: Yes\n"
    info_content += f"3D Coordinates: Yes\n\n"

    info_content += f"SYSTEM OPTIMIZATION\n"
    info_content += f"GPU: {device_info.get('device_name', 'N/A')}\n"
    info_content += f"Total Memory: {device_info.get('memory_total', 'N/A')} GB\n"
    info_content += f"Memory Fraction: {Config.CUDA_MEMORY_FRACTION*100:.0f}%\n"
    info_content += f"Mixed Precision: {Config.MIXED_PRECISION}\n"
    info_content += f"Batch Size: {Config.BATCH_SIZE}\n"
    info_content += f"Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}\n\n"

    info_content += f"TRAINING CONFIGURATION\n"
    info_content += f"Learning Rate: {Config.BASE_LEARNING_RATE}\n"
    info_content += f"Weight Decay: {Config.WEIGHT_DECAY}\n"
    info_content += f"Focal Loss Alpha: {Config.FOCAL_LOSS_ALPHA}\n"
    info_content += f"Focal Loss Gamma: {Config.FOCAL_LOSS_GAMMA}\n"
    info_content += f"Mixup Alpha: {Config.MIXUP_ALPHA}\n\n"

    info_content += f"FACE PROCESSING\n"
    info_content += f"MediaPipe Complexity: {Config.FACE_MESH_COMPLEXITY}\n"
    info_content += f"Landmarks: {Config.NUM_LANDMARKS}\n"
    info_content += f"Coordinate Features: {Config.COORDINATE_FEATURES}\n"
    info_content += f"Face Confidence Threshold: {Config.FACE_CONFIDENCE_THRESHOLD}\n"
    info_content += f"Coordinate Smoothing: {Config.USE_COORDINATE_SMOOTHING}\n\n"

    info_content += f"MODEL STATUS\n"
    model_status = "Loaded" if pytorch_inference.pytorch_model is not None else "Not Loaded"
    scaler_status = "Available" if pytorch_inference.scaler is not None else "Not Available"
    info_content += f"Model: {model_status}\n"
    info_content += f"Scaler: {scaler_status}\n"
    info_content += f"Device: {pytorch_inference.device}\n"

    if pytorch_inference.pytorch_model is not None:
        try:
            total_params = sum(p.numel() for p in pytorch_inference.pytorch_model.parameters())
            trainable_params = sum(p.numel() for p in pytorch_inference.pytorch_model.parameters() if p.requires_grad)
            info_content += f"Total Parameters: {total_params:,}\n"
            info_content += f"Trainable Parameters: {trainable_params:,}\n"
        except:
            info_content += f"Parameter count: Unable to calculate\n"

    text_widget.insert(tk.END, info_content)
    text_widget.config(state=tk.DISABLED)

    ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)

def export_data():
    """Export data to CSV"""
    try:
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM expressions ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            conn.close()

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ID', 'Expression', 'Confidence', 'All_Predictions', 'Timestamp',
                               'Session', 'Model_Version', 'Inference_Time', 'Architecture',
                               'Batch_Size', 'GPU_Memory'])
                writer.writerows(rows)

            messagebox.showinfo("Success", f"Data exported to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export data: {str(e)}")

def generate_report():
    """Generate comprehensive report"""
    try:
        conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM expressions ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            messagebox.showinfo("Info", "No data available for report generation")
            return

        report_window = tk.Toplevel(window)
        report_window.title("Comprehensive Analytics Report")
        report_window.geometry("900x700")
        apply_theme(report_window)

        create_statistics(report_window, rows)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

def clear_data(parent_window):
    """Clear all stored data"""
    if messagebox.askyesno("Confirm Data Deletion",
                          "Are you sure you want to delete ALL stored data?\n\nThis action cannot be undone!",
                          parent=parent_window):
        try:
            conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM expressions')
            cursor.execute('DELETE FROM sessions')
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "All data cleared successfully", parent=parent_window)
            parent_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear data: {str(e)}", parent=parent_window)

def predict_expression(image):
    """Expression prediction using PyTorch model"""
    try:
        start_time = time.time()
        top2, all_predictions = pytorch_inference.predict(image)
        inference_time = time.time() - start_time

        if top2 is None:
            return [], [], inference_time

        # Apply smoothing
        if all_predictions is not None:
            all_predictions = smooth_predictions(all_predictions)
            top2_idx = np.argsort(all_predictions)[-2:][::-1]
            top2 = [(Config.EMOTION_CLASSES[i], float(all_predictions[i])) for i in top2_idx]

        # Confidence filtering
        confidence_threshold = 0.25
        filtered_top2 = [(emotion, conf) for emotion, conf in top2 if conf >= confidence_threshold]

        if not filtered_top2:
            filtered_top2 = [top2[0]] if top2 else []

        return filtered_top2, all_predictions, inference_time

    except Exception:
        return [], [], 0.0

def smooth_predictions(predictions):
    """Prediction smoothing with stability"""
    prediction_history.append(predictions)

    if len(prediction_history) > 1:
        weights = np.exp(np.linspace(-2, 0, len(prediction_history)))
        weights = weights / weights.sum()

        smoothed = np.zeros_like(predictions)
        for i, pred in enumerate(prediction_history):
            smoothed += pred * weights[i]

        return smoothed

    return predictions

def calculate_fps():
    """Calculate FPS with accuracy"""
    current_time = time.time()
    fps_counter.append(current_time)

    if len(fps_counter) > 1:
        time_diff = fps_counter[-1] - fps_counter[0]
        if time_diff > 0:
            return (len(fps_counter) - 1) / time_diff

    return 0

def update_session_stats():
    """Update session statistics"""
    if hasattr(window, 'stats_label'):
        total_frames = session_stats['total_frames']
        if total_frames > 0:
            dominant_emotion = max(session_stats['emotions_detected'],
                                 key=session_stats['emotions_detected'].get)
            dominant_count = session_stats['emotions_detected'][dominant_emotion]
            dominant_percentage = (dominant_count / total_frames) * 100

            stats_text = (f"Frames: {total_frames} | PyTorch Model | "
                         f"GPU Available | Dominant: {dominant_emotion.capitalize()} ({dominant_percentage:.1f}%)")
        else:
            stats_text = f"Frames: 0 | PyTorch Model | GPU Available | Dominant: None"

        window.stats_label.config(text=stats_text)

def show_camera():
    """Camera display with predictions"""
    global camera, last_frame_time

    if camera is None or not camera.isOpened():
        messagebox.showerror("Error", "Camera is not open")
        return

    now = time.time()
    fps_limit = 30
    if now - last_frame_time < 1.0 / fps_limit:
        window.after(1, show_camera)
        return

    last_frame_time = now

    ret, frame = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show landmarks if enabled
        if show_landmarks:
            frame_rgb = draw_landmarks(frame_rgb)

        # Prediction
        top2, all_predictions, inference_time = predict_expression(frame_rgb)

        if top2:
            # Update session stats
            session_stats['total_frames'] += 1
            max_emotion = top2[0][0]
            session_stats['emotions_detected'][max_emotion] += 1

            # Calculate FPS
            current_fps = calculate_fps()

            # Overlay drawing
            overlay = frame_rgb.copy()
            cv2.rectangle(overlay, (5, 5), (550, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame_rgb, 0.3, 0, frame_rgb)

            # Draw predictions
            for i, (emotion, confidence) in enumerate(top2):
                color = (0, 255, 0) if i == 0 else (255, 255, 0)
                confidence_text = f"{emotion.capitalize()}: {confidence:.1%}"

                # Confidence indicator
                if confidence >= 0.9:
                    indicator = "Excellent"
                elif confidence >= 0.8:
                    indicator = "Very Good"
                elif confidence >= 0.6:
                    indicator = "Good"
                elif confidence >= 0.4:
                    indicator = "Fair"
                else:
                    indicator = "Poor"

                cv2.putText(frame_rgb, f"{i+1}. {confidence_text}",
                           (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame_rgb, f"   {indicator}",
                           (10, 50 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Framework indicator
            cv2.putText(frame_rgb, "PyTorch Model",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame_rgb, f"GPU | {Config.HIDDEN_SIZE}D-{Config.NUM_LAYERS}L",
                       (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Performance metrics
            cv2.putText(frame_rgb, f"FPS: {current_fps:.1f}", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_rgb, f"Inference: {inference_time*1000:.1f}ms", (150, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Confidence bars
            for i, (_, confidence) in enumerate(top2):
                bar_width = int(confidence * 250)
                color = (0, 255, 0) if i == 0 else (255, 255, 0)
                cv2.rectangle(frame_rgb, (300, 30 + i * 35), (300 + bar_width, 45 + i * 35), color, -1)
                cv2.putText(frame_rgb, f"{confidence:.1%}", (560, 42 + i * 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Convert and display
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)
        camera_label.config(image=img)
        camera_label.image = img

        update_session_stats()

    camera_label.after(33, show_camera)

def draw_landmarks(frame):
    """Draw facial landmarks on frame"""
    try:
        coordinates = pytorch_inference.face_processor.extract_coordinates_from_frame_enhanced(frame)
        if coordinates is not None:
            # Reshape coordinates to (num_landmarks, 3)
            coords_3d = coordinates.reshape(-1, 3)

            # Draw landmarks
            for i, (x, y, z) in enumerate(coords_3d):
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    except Exception:
        pass

    return frame

def capture_image_frame():
    """Capture frame with detailed analysis"""
    global camera

    if camera is not None and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            top2, all_predictions, inference_time = predict_expression(frame_rgb)

            if top2:
                insert_expression(top2[0][0], top2[0][1], all_predictions, inference_time=inference_time)
                display_capture_analysis(frame_rgb, top2, all_predictions, inference_time)
            else:
                messagebox.showinfo("Info", "No confident predictions available")
        else:
            messagebox.showerror("Error", "Failed to capture frame")
    else:
        messagebox.showerror("Error", "Camera not available")

def display_capture_analysis(frame, top2, predictions, inference_time):
    """Display capture analysis"""
    result_window = tk.Toplevel(window)
    result_window.title("Expression Analysis")
    result_window.geometry("900x1000")
    apply_theme(result_window)

    # Image frame
    img_frame = tk.Frame(result_window)
    img_frame.pack(fill=tk.X, padx=10, pady=10)

    # Convert and display image
    img = Image.fromarray(frame)
    img = img.resize((450, 350), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(img_frame, image=img)
    img_label.image = img
    img_label.pack()

    # Results frame
    result_frame = tk.Frame(result_window)
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Model and performance indicators
    info_frame = tk.Frame(result_frame)
    info_frame.pack(fill='x', pady=5)

    framework_label = tk.Label(info_frame, text="PyTorch Model",
                              font=('Helvetica', 12, 'bold'))
    framework_label.pack(side='left')

    model_info = tk.Label(info_frame, text=f"{Config.HIDDEN_SIZE}D-{Config.NUM_LAYERS}L-{Config.NUM_HEADS}H",
                         font=('Helvetica', 10))
    model_info.pack(side='left', padx=10)

    perf_label = tk.Label(info_frame, text=f"{inference_time*1000:.1f}ms | GPU",
                         font=('Helvetica', 10))
    perf_label.pack(side='right')

    # Top predictions
    top_frame = tk.LabelFrame(result_frame, text="Top Predictions", padx=10, pady=10)
    top_frame.pack(fill='x', pady=5)

    for i, (emotion, confidence) in enumerate(top2):
        pred_frame = tk.Frame(top_frame)
        pred_frame.pack(fill='x', pady=3)

        # Rank and emotion
        rank_label = tk.Label(pred_frame, text=f"#{i+1}", font=('Helvetica', 14, 'bold'), width=3)
        rank_label.pack(side='left')

        emotion_label = tk.Label(pred_frame, text=f"{emotion.capitalize()}",
                                font=('Helvetica', 12, 'bold'), width=12, anchor='w')
        emotion_label.pack(side='left')

        # Confidence with color coding
        if confidence > 0.8:
            conf_color = 'green'
            grade = "Excellent"
        elif confidence > 0.6:
            conf_color = 'blue'
            grade = "Very Good"
        elif confidence > 0.4:
            conf_color = 'orange'
            grade = "Good"
        else:
            conf_color = 'red'
            grade = "Needs improvement"

        conf_label = tk.Label(pred_frame, text=f"{confidence:.2%}",
                             font=('Helvetica', 12), fg=conf_color, width=8)
        conf_label.pack(side='left')

        grade_label = tk.Label(pred_frame, text=grade, font=('Helvetica', 10))
        grade_label.pack(side='left', padx=10)

    # All emotions analysis
    all_frame = tk.LabelFrame(result_frame, text="Complete Analysis (All 7 Emotions)",
                             padx=10, pady=10)
    all_frame.pack(fill='both', expand=True, pady=5)

    # Create canvas for visualization
    canvas_frame = tk.Frame(all_frame)
    canvas_frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(canvas_frame, height=350, bg='white')
    scrollbar_canvas = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_canvas.set)

    # Emotion analysis
    emotion_emojis = {
        'happy': 'Happy', 'sad': 'Sad', 'angry': 'Angry', 'fear': 'Fear',
        'surprised': 'Surprised', 'disgust': 'Disgust', 'neutral': 'Neutral'
    }

    for i, emotion in enumerate(Config.EMOTION_CLASSES):
        confidence = float(predictions[i])
        emotion_frame = tk.Frame(scrollable_frame)
        emotion_frame.pack(fill='x', pady=4, padx=5)

        # Emotion name
        label_text = emotion_emojis.get(emotion.lower(), emotion.capitalize())
        tk.Label(emotion_frame, text=label_text,
                width=15, anchor='w', font=('Helvetica', 11, 'bold')).pack(side='left')

        # Progress bar
        progress_frame = tk.Frame(emotion_frame)
        progress_frame.pack(side='left', padx=10)
        progress = ttk.Progressbar(progress_frame, length=250, value=confidence*100)
        progress.pack(side='top')

        # Percentage with color coding
        if confidence > 0.8:
            conf_color = 'green'
        elif confidence > 0.6:
            conf_color = 'blue'
        elif confidence > 0.4:
            conf_color = 'orange'
        else:
            conf_color = 'red'

        tk.Label(emotion_frame, text=f"{confidence:.2%}",
                width=8, fg=conf_color, font=('Helvetica', 11, 'bold')).pack(side='left', padx=5)

        # Confidence level
        if confidence >= 0.9:
            level = "Excellent"
        elif confidence >= 0.8:
            level = "Very High"
        elif confidence >= 0.6:
            level = "High"
        elif confidence >= 0.4:
            level = "Medium"
        elif confidence >= 0.2:
            level = "Low"
        else:
            level = "Very Low"

        tk.Label(emotion_frame, text=level, width=12,
                font=('Helvetica', 10), fg='gray').pack(side='left')

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar_canvas.pack(side="right", fill="y")

    # Button frame
    btn_frame = tk.Frame(result_window)
    btn_frame.pack(fill='x', padx=10, pady=10)

    ttk.Button(btn_frame, text="Save Analysis",
              command=lambda: save_analysis(frame, predictions, inference_time)).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="View Trends",
              command=lambda: show_emotion_trends()).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Model Details",
              command=lambda: show_model_info()).pack(side='left', padx=5)
    ttk.Button(btn_frame, text="Close",
              command=result_window.destroy).pack(side='right', padx=5)

def save_analysis(frame, predictions, inference_time):
    """Save analysis to file"""
    try:
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            analysis_data = {
                'timestamp': datetime.now(ist).isoformat(),
                'framework': 'PyTorch',
                'model_version': 'pytorch_v2.0',
                'model_architecture': {
                    'hidden_size': Config.HIDDEN_SIZE,
                    'num_layers': Config.NUM_LAYERS,
                    'num_heads': Config.NUM_HEADS,
                    'dropout_rate': Config.DROPOUT_RATE,
                    'geometric_feature_dim': Config.GEOMETRIC_FEATURE_DIM,
                    'expert_hidden_size': Config.EXPERT_HIDDEN_SIZE
                },
                'performance_metrics': {
                    'inference_time_ms': inference_time * 1000,
                    'batch_size': Config.BATCH_SIZE,
                    'mixed_precision': Config.MIXED_PRECISION,
                    'gpu_memory_fraction': Config.CUDA_MEMORY_FRACTION
                },
                'predictions': {emotion: float(predictions[i])
                              for i, emotion in enumerate(Config.EMOTION_CLASSES)},
                'top_emotion': Config.EMOTION_CLASSES[np.argmax(predictions)],
                'confidence': float(np.max(predictions)),
                'quality_metrics': {
                    'entropy': float(-np.sum(predictions * np.log(predictions + 1e-10))),
                    'avg_confidence': float(np.mean(predictions)),
                    'max_confidence': float(np.max(predictions)),
                    'confidence_spread': float(np.std(predictions)),
                    'high_confidence_count': int(np.sum(predictions > 0.8))
                }
            }

            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=2)

            messagebox.showinfo("Success", f"Analysis saved to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save analysis: {str(e)}")

def show_emotion_trends():
    """Show emotion trends analysis"""
    try:
        conn = sqlite3.connect(str(Config.LOGS_PATH / "expressions.db"))
        cursor = conn.cursor()
        cursor.execute('SELECT expression, timestamp, confidence FROM expressions ORDER BY timestamp DESC LIMIT 100')
        recent_data = cursor.fetchall()
        conn.close()

        if not recent_data:
            messagebox.showinfo("Info", "No data available for trend analysis")
            return

        trends_window = tk.Toplevel(window)
        trends_window.title("Emotion Trends")
        trends_window.geometry("700x500")
        apply_theme(trends_window)

        text_widget = tk.Text(trends_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)

        # Calculate trends
        recent_emotions = [row[0] for row in recent_data[:20]]
        recent_confidences = [float(row[2]) for row in recent_data[:20]]
        emotion_counts = Counter(recent_emotions)

        trends_text = "EMOTION TRENDS (Last 20 Predictions)\n"
        trends_text += f"Framework: PyTorch | Model: {Config.HIDDEN_SIZE}D-{Config.NUM_LAYERS}L-{Config.NUM_HEADS}H\n"
        trends_text += "=" * 60 + "\n\n"

        trends_text += "EMOTION DISTRIBUTION\n"
        for emotion, count in emotion_counts.most_common():
            percentage = (count / len(recent_emotions)) * 100
            emotion_confidences = [float(row[2]) for row in recent_data[:20] if row[0] == emotion]
            avg_conf = np.mean(emotion_confidences) if emotion_confidences else 0

            bar = "#" * int(percentage / 3)
            trends_text += f"{emotion.capitalize():10} {bar:20} {percentage:5.1f}% (Avg: {avg_conf:.1%})\n"

        trends_text += f"\nQUALITY METRICS\n"
        trends_text += f"Average Confidence: {np.mean(recent_confidences):.1%}\n"
        trends_text += f"High Confidence (>80%): {sum(1 for c in recent_confidences if c > 0.8)}/{len(recent_confidences)}\n"
        trends_text += f"Confidence Std Dev: {np.std(recent_confidences):.1%}\n"

        text_widget.insert(tk.END, trends_text)
        text_widget.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to show trends: {str(e)}")

# Theme functions - FIXED
def toggle_theme():
    """Toggle between light and dark themes - FIXED"""
    global dark_mode
    dark_mode = not dark_mode
    apply_theme(window)
    theme_btn.config(text="Light Mode" if dark_mode else "Dark Mode")

def apply_theme(widget):
    """Apply current theme to widget - FIXED"""
    if dark_mode:
        bg_color = '#2b2b2b'
        fg_color = '#ffffff'
    else:
        bg_color = '#e0f7fa'
        fg_color = '#000000'

    try:
        widget.configure(bg=bg_color)
        for child in widget.winfo_children():
            if isinstance(child, (tk.Label, tk.Frame, tk.Text)):
                try:
                    child.configure(bg=bg_color, fg=fg_color)
                except:
                    pass
            apply_theme(child)
    except:
        pass

# Landmarks function - FIXED
def toggle_landmarks():
    """Toggle landmark display - FIXED"""
    global show_landmarks
    show_landmarks = not show_landmarks
    landmarks_btn.config(text="Hide Landmarks" if show_landmarks else "Show Landmarks")

def show_help():
    """Show help dialog"""
    help_text = """
VISAGECNN - PYTORCH EMOTION RECOGNITION

FEATURES
- PyTorch implementation with coordinate-based learning
- Multi-head attention mechanism for better feature relationships
- Geometric feature extraction from facial landmarks
- Emotion-specific expert networks for specialized processing
- 478 facial landmark analysis using MediaPipe (highest complexity)
- Real-time prediction smoothing and confidence filtering
- Session statistics with dominant emotion tracking
- Comprehensive analytics dashboard with performance metrics

CONTROLS
- Access Camera: Start live emotion detection
- Capture & Analyze: Detailed frame analysis with all 7 emotions
- Show/Hide Landmarks: Toggle facial landmarks display
- Analytics Dashboard: View comprehensive data analytics
- Model Info: Detailed information about the architecture
- Dark/Light Mode: Theme switching

ANALYTICS
- Real-time confidence scoring with quality indicators
- Session statistics tracking with performance metrics
- Emotion trend analysis over time with confidence tracking
- Comprehensive reporting with inference time tracking
- Data export capabilities (CSV format)
- Model architecture performance analysis

TIPS FOR BEST RESULTS
- Ensure good lighting for better face detection
- Keep face centered and at comfortable distance
- Avoid rapid movements for stable predictions
- Check confidence indicators for prediction quality
- Use analytics to track emotion patterns and model performance

TECHNICAL INFO
- Framework: PyTorch Architecture
- Model: CoordinateEmotionNet
- Input Features: 1,434 (478 landmarks x 3 coordinates)
- Hidden Dimensions: """ + str(Config.HIDDEN_SIZE) + """D
- Transformer Layers: """ + str(Config.NUM_LAYERS) + """
- Attention Heads: """ + str(Config.NUM_HEADS) + """
- Real-time processing with prediction smoothing
- MediaPipe integration with highest complexity (Level 2)
- SQLite database for comprehensive data logging
- GPU acceleration support

PERFORMANCE
- Target: 30 FPS real-time processing
- Batch Size: """ + str(Config.BATCH_SIZE) + """
- Mixed Precision: Enabled for efficiency
- Memory Usage: """ + str(Config.CUDA_MEMORY_FRACTION*100) + """% GPU utilization
- Inference time tracking and optimization
- Memory efficient coordinate processing
"""

    help_window = tk.Toplevel(window)
    help_window.title("Help Guide - PyTorch VisageCNN")
    help_window.geometry("800x700")
    apply_theme(help_window)

    text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
    text_widget.pack(fill='both', expand=True, padx=10, pady=10)
    text_widget.insert(tk.END, help_text)
    text_widget.config(state=tk.DISABLED)

    ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)

# Camera functions
def terminate_program():
    """Cleanup and shutdown"""
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    window.destroy()

def switch_to_screen2():
    """Switch to camera screen"""
    screen1_frame.grid_forget()
    screen2_frame.grid(row=0, column=0, sticky="nsew")
    update_session_stats()

def switch_to_screen1():
    """Switch to main menu"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    screen2_frame.grid_forget()
    screen1_frame.grid(row=0, column=0, sticky="nsew")

def open_camera():
    """Open camera"""
    global camera
    try:
        for camera_index in [1, 2]:
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                break

        if not camera.isOpened():
            raise ValueError("No camera found")

        show_camera()
        switch_to_screen2()

    except Exception as e:
        messagebox.showerror("Error", f"Camera error: {str(e)}")

def main():
    """Main application function"""
    global window, camera_label, theme_btn, landmarks_btn
    global screen1_frame, screen2_frame

    # Initialize database
    initialize_db()

    # Create main window
    window = tk.Tk()
    window.title("VisageCNN - PyTorch Expression Recognition")
    window.geometry("1000x750")

    # Center window
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - 1000) // 2
    y = (screen_height - 750) // 2
    window.geometry(f"1000x750+{x}+{y}")

    # Configure styles
    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 11), padding=8)
    style.configure('TLabel', font=('Helvetica', 11))

    # Configure grid
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)

    # Screen 1: Main Menu
    screen1_frame = tk.Frame(window, bg='#e0f7fa')
    screen1_frame.grid(row=0, column=0, sticky="nsew")

    # Title
    title_label = tk.Label(
        screen1_frame,
        text="VisageCNN",
        font=('Helvetica', 26, 'bold'),
        bg='#e0f7fa',
        pady=15
    )
    title_label.pack(pady=20)

    subtitle_label = tk.Label(
        screen1_frame,
        text="PyTorch Coordinate-Based Expression Recognition",
        font=('Helvetica', 14),
        bg='#e0f7fa'
    )
    subtitle_label.pack()

    # Model architecture info
    arch_label = tk.Label(
        screen1_frame,
        text=f"Architecture: {Config.HIDDEN_SIZE}D Hidden | {Config.NUM_LAYERS} Layers | {Config.NUM_HEADS} Attention Heads",
        font=('Helvetica', 10, 'italic'),
        bg='#e0f7fa',
        fg='#666666'
    )
    arch_label.pack(pady=5)

    # Model status
    model_status = "PyTorch Model Ready" if pytorch_inference.pytorch_model is not None else "PyTorch Model Not Found"
    scaler_status = "Scaler Ready" if pytorch_inference.scaler is not None else "Scaler Not Found"

    status_frame = tk.Frame(screen1_frame, bg='#e0f7fa')
    status_frame.pack(pady=10)

    model_label = tk.Label(
        status_frame,
        text=model_status,
        font=('Helvetica', 10),
        fg="green" if pytorch_inference.pytorch_model is not None else "red",
        bg='#e0f7fa'
    )
    model_label.pack()

    scaler_label = tk.Label(
        status_frame,
        text=scaler_status,
        font=('Helvetica', 10),
        fg="green" if pytorch_inference.scaler is not None else "orange",
        bg='#e0f7fa'
    )
    scaler_label.pack()

    # GPU status
    gpu_status = f"GPU Ready ({Config.CUDA_MEMORY_FRACTION*100:.0f}% utilization)" if torch.cuda.is_available() else "CPU Mode"
    gpu_label = tk.Label(
        status_frame,
        text=gpu_status,
        font=('Helvetica', 10),
        fg="blue" if torch.cuda.is_available() else "orange",
        bg='#e0f7fa'
    )
    gpu_label.pack()

    # Main buttons
    button_frame = tk.Frame(screen1_frame, bg='#e0f7fa')
    button_frame.pack(pady=40)

    access_camera_button = ttk.Button(button_frame, text="Access Camera", command=open_camera)
    access_camera_button.pack(pady=15, ipadx=30)

    view_data_button = ttk.Button(button_frame, text="Analytics Dashboard", command=view_data)
    view_data_button.pack(pady=10, ipadx=30)

    # Secondary buttons
    secondary_frame = tk.Frame(screen1_frame, bg='#e0f7fa')
    secondary_frame.pack(pady=20)

    theme_btn = ttk.Button(secondary_frame, text="Dark Mode", command=toggle_theme)
    theme_btn.pack(side='left', padx=5)

    model_info_btn = ttk.Button(secondary_frame, text="Model Info", command=show_model_info)
    model_info_btn.pack(side='left', padx=5)

    help_btn = ttk.Button(secondary_frame, text="Help Guide", command=show_help)
    help_btn.pack(side='left', padx=5)

    # Terminate button
    terminate_button_screen1 = ttk.Button(screen1_frame, text="Exit Application", command=terminate_program)
    terminate_button_screen1.pack(side=tk.BOTTOM, pady=20)

    # Screen 2: Camera View
    screen2_frame = tk.Frame(window, bg='#e0f7fa')

    # Camera display
    camera_label = tk.Label(screen2_frame, bg='#000000', text="Camera Loading...",
                           fg='white', font=('Helvetica', 16))
    camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Control buttons
    control_frame = tk.Frame(screen2_frame, bg='#e0f7fa')
    control_frame.pack(fill=tk.X, pady=5)

    # Left side controls
    left_controls = tk.Frame(control_frame, bg='#e0f7fa')
    left_controls.pack(side='left', padx=10)

    capture_image_button = ttk.Button(left_controls, text="Capture & Analyze", command=capture_image_frame)
    capture_image_button.pack(side='left', padx=5)

    landmarks_btn = ttk.Button(left_controls, text="Show Landmarks", command=toggle_landmarks)
    landmarks_btn.pack(side='left', padx=5)

    model_info_btn2 = ttk.Button(left_controls, text="Model Info", command=show_model_info)
    model_info_btn2.pack(side='left', padx=5)

    # Right side controls
    right_controls = tk.Frame(control_frame, bg='#e0f7fa')
    right_controls.pack(side='right', padx=10)

    back_button = ttk.Button(right_controls, text="Main Menu", command=switch_to_screen1)
    back_button.pack(side='right', padx=5)

    # Status bar
    status_frame = tk.Frame(screen2_frame, bg='#e0f7fa')
    status_frame.pack(fill='x', pady=5)

    window.stats_label = tk.Label(status_frame,
                                 text="Session Stats: Frames: 0 | PyTorch Model | GPU Available | Dominant: None",
                                 bg='#e0f7fa', font=('Helvetica', 10))
    window.stats_label.pack()

    # Start with Screen 1
    switch_to_screen1()

    # Apply initial theme
    apply_theme(window)

    # Start main loop
    window.mainloop()

if __name__ == "__main__":
    main()
