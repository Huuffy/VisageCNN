"""VisageCNN — Professional Real-Time Emotion Recognition.

Three-screen experience
-----------------------
  0. Loading     — hardware check, model / scaler status with tick indicators
  1. Camera pick — live thumbnails for every detected camera; press 0-3 to choose
  2. Live HUD    — 1280×720 split display: webcam left, stats panel right

Controls (Screen 2)
-------------------
  Q   quit
  C   save annotated capture to captures/<timestamp>.png
  L   toggle MediaPipe landmark dots

Usage
-----
  python inference/run_hybrid.py
  python inference/run_hybrid.py --ensemble
  python inference/run_hybrid.py --camera-index 0 --max-fps 30
"""

import warnings
import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime

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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from visage_er.config import Config
from visage_er.models.hybrid_model import create_hybrid_model

# ── Canvas geometry ───────────────────────────────────────────────────────────
CW, CH   = 1280, 720          # full canvas
CAM_W    = 640                # webcam region width
CAM_H    = 480                # webcam capture height
CAM_Y    = (CH - CAM_H) // 2  # = 120 — centre webcam vertically
PX       = CAM_W              # right panel origin-x = 640
PW       = CW - PX            # right panel width    = 640

# ── Colour palette — BGR ──────────────────────────────────────────────────────
C_BG      = (28,  22,  38)
C_PANEL   = (38,  32,  52)
C_BORDER  = (90,  72,  45)
C_TEXT1   = (245, 240, 255)
C_TEXT2   = (140, 130, 160)
C_DIV     = (65,  55,  80)
C_OK      = (80,  200,  80)
C_WARN    = (30,  170, 255)
C_ERR     = (60,   60, 220)

EMOTION_BGR: dict = {
    "Angry":     (50,  50,  220),
    "Disgust":   (30,  140, 255),
    "Fear":      (180,  30, 180),
    "Happy":     (30,  210,  80),
    "Neutral":   (180, 170,  80),
    "Sad":       (220,  80,  30),
    "Surprised": (30,  220, 220),
}
# PIL needs RGB
EMOTION_RGB = {k: (v[2], v[1], v[0]) for k, v in EMOTION_BGR.items()}

# ── Font candidates ───────────────────────────────────────────────────────────
_FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


# ─────────────────────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────────────────────

class HybridInference:
    """Stateful inference engine — MediaPipe + HybridEmotionNet + temporal smoothing."""

    FACE_CROP_SIZE = Config.FACE_CROP_SIZE

    def __init__(self, model_path=None, swa_path=None):
        self.device = Config.DEVICE

        model_path = Path(model_path or Config.MODELS_PATH / "weights" / "hybrid_best_model.pth")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = create_hybrid_model(pretrained_cnn=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.val_acc = ckpt.get("val_acc", "N/A")

        self.swa_model = None
        if swa_path:
            sp = Path(swa_path)
            if sp.exists():
                self.swa_model = create_hybrid_model(pretrained_cnn=False)
                sd = torch.load(sp, map_location=self.device, weights_only=False)
                if isinstance(sd, dict) and "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                if any(k.startswith("module.") for k in sd):
                    sd = {k[7:]: v for k, v in sd.items()}
                self.swa_model.load_state_dict(sd)
                self.swa_model.eval()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        scaler_path = Config.MODELS_PATH / "scalers" / "hybrid_coordinate_scaler.pkl"
        self.scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.prob_history = deque(maxlen=8)
        self.ema_probs    = None
        self.ema_alpha    = 0.35
        self.use_amp      = Config.MIXED_PRECISION and torch.cuda.is_available()

    def infer(self, frame: np.ndarray) -> dict:
        """Run inference on one BGR frame.  Returns a result dict — no drawing."""
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        empty = dict(emotion=None, confidence=None, probs=np.zeros(7),
                     bbox=None, face_too_small=False, landmarks=None)

        if not res.multi_face_landmarks:
            return empty

        lms  = res.multi_face_landmarks[0]
        xs   = [lm.x * w for lm in lms.landmark]
        ys   = [lm.y * h for lm in lms.landmark]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        fw, fh = x2 - x1, y2 - y1

        if fw < 30 or fh < 30:
            return {**empty, "face_too_small": True}

        # Coordinates
        coords = np.array([v for lm in lms.landmark
                           for v in (lm.x * w, lm.y * h, lm.z * w)], dtype=np.float32)
        c3 = coords.reshape(-1, 3)
        c3[:, 0] = (c3[:, 0] - w / 2) / (w / 2)
        c3[:, 1] = (c3[:, 1] - h / 2) / (h / 2)
        c3[:, 2] *= 0.1
        coords = c3.flatten()
        if self.scaler:
            try:
                coords = self.scaler.transform([coords])[0]
            except Exception:
                pass
        if len(coords) < Config.COORDINATE_DIM:
            pad = np.zeros(Config.COORDINATE_DIM, dtype=np.float32)
            pad[:len(coords)] = coords
            coords = pad

        # Face crop
        pr = 0.4 if fw < 80 else (0.3 if fw < 140 else 0.2)
        cx1, cx2 = max(0, x1 - int(fw * pr)), min(w, x2 + int(fw * pr))
        cy1, cy2 = max(0, y1 - int(fh * pr)), min(h, y2 + int(fh * pr))
        crop = frame[cy1:cy2, cx1:cx2]
        crop = cv2.resize(crop, (self.FACE_CROP_SIZE, self.FACE_CROP_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = ((crop - self.img_mean) / self.img_std).transpose(2, 0, 1)

        ct = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        it = torch.tensor(crop,   dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            ctx = autocast("cuda") if self.use_amp else torch.no_grad()
            with (autocast("cuda") if self.use_amp else _Noop()):
                out = self.model(ct, it)
                if self.swa_model is not None:
                    out = (out + self.swa_model(ct, it)) / 2.0

        probs = F.softmax(out, dim=1).cpu().numpy()[0]

        self.ema_probs = (probs if self.ema_probs is None
                          else self.ema_alpha * probs + (1 - self.ema_alpha) * self.ema_probs)
        self.prob_history.append(probs)
        smoothed   = 0.5 * self.ema_probs + 0.5 * np.mean(self.prob_history, axis=0)
        idx        = int(np.argmax(smoothed))
        confidence = float(smoothed[idx])
        emotion    = Config.EMOTION_CLASSES[idx]

        return dict(emotion=emotion, confidence=confidence, probs=smoothed,
                    bbox=(x1, y1, x2, y2), face_too_small=(fw < 60 or fh < 60),
                    landmarks=lms)


class _Noop:
    """No-op context manager used as CPU autocast fallback."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

class VisageCNNApp:
    """Three-screen OpenCV application (loading → camera pick → live HUD)."""

    WIN  = "VisageCNN"
    WIN2 = "VisageCNN — Select Camera"

    def __init__(self):
        self._font_cache: dict = {}
        self._history: deque  = deque(maxlen=40)
        self._flash_msg: str  = ""
        self._flash_until: float = 0.0
        self._show_landmarks: bool = False
        self._fps_buf: deque  = deque(maxlen=30)

    # ── Font helpers ──────────────────────────────────────────────────────────

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            for p in _FONT_PATHS:
                if Path(p).exists():
                    try:
                        self._font_cache[size] = ImageFont.truetype(p, size)
                        break
                    except Exception:
                        pass
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _text_pil(self, canvas: np.ndarray, text: str, x: int, y: int,
                  size: int, color_rgb: tuple, anchor: str = "la") -> np.ndarray:
        """Render text via Pillow and alpha-blend onto an existing BGR canvas."""
        font = self._font(size)
        dummy = Image.new("RGBA", (1, 1))
        bb = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)

        # Draw text at (DRAW, DRAW) inside the layer.
        # Layer must reach (DRAW + bb[2], DRAW + bb[3]) — NOT (bb[2]-bb[0], bb[3]-bb[1])
        # because subtracting bb[1] underestimates height when bb[1] > 0, clipping descenders.
        DRAW = 2
        PAD  = 6
        lw = max(1, DRAW + bb[2] + PAD)
        lh = max(1, DRAW + bb[3] + PAD)

        layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((DRAW, DRAW), text, font=font, fill=(*color_rgb, 255))
        arr = np.array(layer)   # H×W×4 RGBA

        # Clip to canvas bounds
        cx1, cy1 = max(0, x), max(0, y)
        cx2, cy2 = min(canvas.shape[1], x + lw), min(canvas.shape[0], y + lh)
        ax1, ay1 = cx1 - x, cy1 - y
        ax2, ay2 = ax1 + (cx2 - cx1), ay1 + (cy2 - cy1)
        if cx2 <= cx1 or cy2 <= cy1:
            return canvas

        alpha  = arr[ay1:ay2, ax1:ax2, 3:4].astype(np.float32) / 255.0
        rgb_fg = arr[ay1:ay2, ax1:ax2, :3][:, :, ::-1]   # RGBA → BGR

        canvas[cy1:cy2, cx1:cx2] = (
            canvas[cy1:cy2, cx1:cx2] * (1 - alpha) + rgb_fg * alpha
        ).astype(np.uint8)
        return canvas

    # ── Drawing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rect(canvas, x1, y1, x2, y2, color, alpha=1.0, thickness=-1):
        if alpha < 1.0:
            roi = canvas[y1:y2, x1:x2]
            cv2.rectangle(roi, (0, 0), (x2-x1, y2-y1), color, thickness)
            canvas[y1:y2, x1:x2] = cv2.addWeighted(roi, alpha,
                                                    canvas[y1:y2, x1:x2], 1-alpha, 0)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def _divider(canvas, y, x1=PX+20, x2=PX+PW-20):
        cv2.line(canvas, (x1, y), (x2, y), C_DIV, 1)

    @staticmethod
    def _corner_markers(frame, x1, y1, x2, y2, color, arm=24, t=3):
        for (ax, ay, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                  (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (ax, ay), (ax + dx*arm, ay), color, t)
            cv2.line(frame, (ax, ay), (ax, ay + dy*arm), color, t)

    # ── Screen 0: Loading ─────────────────────────────────────────────────────

    def show_loading(self, args) -> "HybridInference | None":
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, CW, CH)

        steps: list = []

        def refresh():
            c = np.full((CH, CW, 3), C_BG, dtype=np.uint8)
            # Left accent strip
            cv2.rectangle(c, (0, 0), (6, CH), (100, 80, 50), -1)
            c = self._text_pil(c, "VISAGECNN", 40, 60, 52,
                               (200, 180, 255))
            c = self._text_pil(c, "Hybrid Emotion Recognition  ·  Loading",
                               40, 122, 20, (140, 130, 160))
            cv2.line(c, (40, 160), (CW - 40, 160), C_DIV, 1)
            for i, (msg, status) in enumerate(steps):
                y = 190 + i * 42
                if   status == "ok":      col = C_OK;   icon = "  OK "
                elif status == "warn":    col = C_WARN; icon = "WARN "
                elif status == "error":   col = C_ERR;  icon = " ERR "
                elif status == "loading": col = (180, 180, 180); icon = " ...  "
                else:                     col = C_TEXT2; icon = "     "
                cv2.rectangle(c, (36, y - 2), (40, y + 26), col, -1)
                c = self._text_pil(c, icon, 46, y, 18,
                                   (col[2], col[1], col[0]))
                c = self._text_pil(c, msg,  130, y, 18,
                                   (C_TEXT1[2], C_TEXT1[1], C_TEXT1[0]))
            cv2.imshow(self.WIN, c)
            cv2.waitKey(1)

        # ── Step 1: Hardware ──
        steps.append(("Checking hardware...", "loading")); refresh()
        time.sleep(0.2)
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            steps[-1] = (f"CUDA  ·  {name}  ({vram} GB VRAM)", "ok")
        else:
            steps[-1] = ("No CUDA — running on CPU (inference will be slow)", "warn")
        refresh()

        # ── Step 2: Model weights ──
        model_path = Path(args.model) if args.model else (
            Config.MODELS_PATH / "weights" / "hybrid_best_model.pth")
        steps.append(("Loading model weights...", "loading")); refresh()
        if not model_path.exists():
            steps[-1] = (f"Model not found:  {model_path}", "error")
            steps.append(("  →  huggingface-cli download Huuffy/VisageCNN "
                          "models/weights/hybrid_best_model.pth --local-dir .", "info"))
            refresh(); cv2.waitKey(0)
            return None

        swa_path = None
        if args.ensemble:
            swa_path = args.swa_model or str(Config.MODELS_PATH / "weights" / "hybrid_swa_final.pth")

        try:
            engine = HybridInference(model_path=str(model_path), swa_path=swa_path)
        except Exception as exc:
            steps[-1] = (f"Failed to load model:  {exc}", "error")
            refresh(); cv2.waitKey(0)
            return None

        mb = model_path.stat().st_size // (1024 * 1024)
        steps[-1] = (f"Model loaded  ·  {model_path.name}  ({mb} MB)  "
                     f"·  best val {engine.val_acc}%", "ok")
        refresh()

        # ── Step 3: SWA ──
        if args.ensemble:
            if engine.swa_model is not None:
                steps.append(("SWA ensemble model loaded — averaging both checkpoints", "ok"))
            else:
                steps.append(("SWA model not found — running single-model mode", "warn"))
            refresh()

        # ── Step 4: Scaler ──
        if engine.scaler is not None:
            steps.append(("Coordinate scaler loaded  ·  hybrid_coordinate_scaler.pkl", "ok"))
        else:
            steps.append(("Scaler not found — coordinates unscaled (accuracy may drop)", "warn"))
        refresh()

        steps.append(("Ready!  Opening camera selector...", "ok"))
        refresh()
        time.sleep(0.8)
        return engine

    # ── Screen 1: Camera selector ─────────────────────────────────────────────

    def select_camera(self, force_index: int = None) -> int:
        if force_index is not None:
            return force_index

        # Probe cameras 0-3
        available: dict = {}
        for idx in range(4):
            kw = {"index": idx}
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available[idx] = {"cap": cap, "frame": frame}
                    continue
            cap.release()

        if not available:
            print("[VisageCNN] No cameras detected — defaulting to index 0.")
            return 0

        if len(available) == 1:
            idx = list(available.keys())[0]
            available[idx]["cap"].release()
            return idx

        # Build selector window
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, CW, CH)

        selected = None
        while selected is None:
            # Refresh frames
            for info in available.values():
                ret, f = info["cap"].read()
                if ret:
                    info["frame"] = f

            canvas = self._build_selector_canvas(available)
            cv2.imshow(self.WIN, canvas)

            key = cv2.waitKey(40) & 0xFF
            for i in available:
                if key == ord(str(i)):
                    selected = i
            if key == 27:
                selected = list(available.keys())[0]

        for info in available.values():
            info["cap"].release()

        return selected

    def _build_selector_canvas(self, available: dict) -> np.ndarray:
        c = np.full((CH, CW, 3), C_BG, dtype=np.uint8)
        cv2.rectangle(c, (0, 0), (6, CH), C_BORDER, -1)

        c = self._text_pil(c, "VISAGECNN", 40, 60,  52, (200, 180, 255))
        c = self._text_pil(c, "Select a camera — press the number key",
                           40, 122, 22, (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))
        cv2.line(c, (40, 165), (CW - 40, 165), C_DIV, 1)

        n       = max(len(available), 1)
        thumb_w = min(280, (CW - 80) // n - 30)
        thumb_h = int(thumb_w * 3 / 4)
        total_w = n * thumb_w + (n - 1) * 30
        start_x = (CW - total_w) // 2
        thumb_y = (CH - thumb_h) // 2 - 20

        for slot, idx in enumerate(sorted(available.keys())):
            tx = start_x + slot * (thumb_w + 30)
            frame = available[idx]["frame"]
            thumb = cv2.resize(frame, (thumb_w, thumb_h))

            # Border
            cv2.rectangle(c, (tx - 3, thumb_y - 3),
                          (tx + thumb_w + 3, thumb_y + thumb_h + 3),
                          C_BORDER, 2)
            c[thumb_y:thumb_y + thumb_h, tx:tx + thumb_w] = thumb

            label = f"Camera {idx}  [press {idx}]"
            c = self._text_pil(c, label,
                               tx, thumb_y + thumb_h + 14, 18,
                               (C_TEXT1[2], C_TEXT1[1], C_TEXT1[0]))

        return c

    # ── Screen 2: Live HUD ────────────────────────────────────────────────────

    def run(self, engine: "HybridInference", camera_index: int, max_fps: int = 0):
        cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY,
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, CW, CH)

        frame_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
        infer_ms = 0.0

        while True:
            t0  = time.perf_counter()
            ret, raw = cap.read()
            if not ret:
                break

            t_inf = time.perf_counter()
            data  = engine.infer(raw)
            infer_ms = (time.perf_counter() - t_inf) * 1000

            if data["emotion"]:
                self._history.append(data["emotion"])

            # FPS
            now = time.perf_counter()
            self._fps_buf.append(now)
            fps = (len(self._fps_buf) - 1) / max(self._fps_buf[-1] - self._fps_buf[0], 1e-6) \
                  if len(self._fps_buf) > 1 else 0.0

            canvas = self._build_canvas(raw, data, fps, infer_ms)
            cv2.imshow(self.WIN, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key in (ord("c"), ord("C")):
                Path("captures").mkdir(exist_ok=True)
                fname = f"captures/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, canvas)
                self._flash(f"Saved  {fname}")
            elif key in (ord("l"), ord("L")):
                self._show_landmarks = not self._show_landmarks

            # Throttle
            elapsed = time.perf_counter() - t0
            if frame_interval and elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        cap.release()
        cv2.destroyAllWindows()

    def _flash(self, msg: str, duration: float = 2.5):
        self._flash_msg   = msg
        self._flash_until = time.perf_counter() + duration

    # ── Canvas assembly ───────────────────────────────────────────────────────

    def _build_canvas(self, raw: np.ndarray, data: dict,
                      fps: float, infer_ms: float) -> np.ndarray:
        canvas = np.full((CH, CW, 3), C_BG, dtype=np.uint8)

        # ── Left: webcam area ──
        raw_h, raw_w = raw.shape[:2]
        frame = cv2.resize(raw, (CAM_W, CAM_H))

        # Scale bbox from raw capture resolution → display resolution
        draw_data = data
        if data["bbox"] is not None and (raw_w != CAM_W or raw_h != CAM_H):
            sx, sy = CAM_W / raw_w, CAM_H / raw_h
            x1, y1, x2, y2 = data["bbox"]
            draw_data = {**data, "bbox": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))}

        self._draw_face_overlay(frame, draw_data)
        if self._show_landmarks and data["landmarks"]:
            h, w = frame.shape[:2]
            for lm in data["landmarks"].landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 200, 180), -1)
        canvas[CAM_Y:CAM_Y + CAM_H, 0:CAM_W] = frame

        # Keyboard hint strip below webcam
        canvas = self._text_pil(canvas,
                                "C  capture    L  landmarks    Q  quit",
                                20, CAM_Y + CAM_H + 12, 15,
                                (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))

        # ── Divider line between halves ──
        cv2.line(canvas, (PX, 0), (PX, CH), C_BORDER, 2)

        # ── Right panel ──
        self._draw_panel(canvas, data, fps, infer_ms)

        # ── Flash message ──
        if time.perf_counter() < self._flash_until:
            canvas = self._text_pil(canvas, self._flash_msg,
                                    CAM_W // 2 - 200, CAM_Y + 20, 20,
                                    (80, 220, 80))

        return canvas

    # ── Face overlay ──────────────────────────────────────────────────────────

    def _draw_face_overlay(self, frame: np.ndarray, data: dict):
        if data["bbox"] is None:
            msg = "Move closer" if data["face_too_small"] else "No face detected"
            h, w = frame.shape[:2]
            cv2.putText(frame, msg, (w // 2 - 90, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
            return

        x1, y1, x2, y2 = data["bbox"]
        em   = data["emotion"]
        conf = data["confidence"]
        color = EMOTION_BGR.get(em, (200, 200, 200))

        # Corner markers
        self._corner_markers(frame, x1, y1, x2, y2, color, arm=22, t=3)

        if conf >= 0.35:
            # Emotion chip above face
            label = f"{em.upper()}  {conf * 100:.0f}%"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.65, 2)
            pad = 7
            bx1, by1 = x1, max(0, y1 - th - 2 * pad - 6)
            bx2, by2 = bx1 + tw + 2 * pad, bx1 + th + 2 * pad   # height
            by2 = by1 + th + 2 * pad
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            lighter = tuple(min(v + 60, 255) for v in color)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), lighter, 1)
            cv2.putText(frame, label, (bx1 + pad, by2 - pad),
                        font, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Uncertain", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (130, 130, 130), 2)

        if data["face_too_small"]:
            cv2.putText(frame, "Move closer", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 170, 255), 1)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _draw_panel(self, canvas: np.ndarray, data: dict,
                    fps: float, infer_ms: float):
        # Panel background slightly lighter
        canvas[0:CH, PX:CW] = C_PANEL

        em   = data["emotion"] or "—"
        conf = data["confidence"] or 0.0
        probs = data["probs"]

        # ── Header ──
        canvas = self._text_pil(canvas, "VISAGECNN", PX + 22, 22, 26,
                                (200, 180, 255))
        fps_str = f"FPS  {fps:.1f}"
        canvas = self._text_pil(canvas, fps_str, PX + PW - 120, 24, 20,
                                (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))
        self._divider(canvas, 68, PX + 16, CW - 16)

        # ── Dominant emotion block ──
        color_bgr = EMOTION_BGR.get(em, (200, 200, 200))
        color_rgb = EMOTION_RGB.get(em, (200, 200, 200))

        # Subtle glow bar on left edge
        cv2.rectangle(canvas, (PX, 72), (PX + 5, 225), color_bgr, -1)

        if em != "—":
            canvas = self._text_pil(canvas, em.upper(), PX + 28, 80, 48, color_rgb)
            pct_str = f"{conf * 100:.1f}%"
            canvas = self._text_pil(canvas, pct_str, PX + 28, 138, 34,
                                    (C_TEXT1[2], C_TEXT1[1], C_TEXT1[0]))
            grade = ("Excellent" if conf >= 0.90 else
                     "Very Good" if conf >= 0.75 else
                     "Good"      if conf >= 0.55 else
                     "Fair"      if conf >= 0.40 else "Uncertain")
            canvas = self._text_pil(canvas, grade, PX + 30, 186, 18,
                                    (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))
        else:
            canvas = self._text_pil(canvas, "No face", PX + 28, 100, 36,
                                    (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))

        self._divider(canvas, 230, PX + 16, CW - 16)

        # ── Probability bars ──
        bar_x  = PX + 110
        bar_w  = PW - 160
        bar_h  = 20
        gap    = 37
        label_x = PX + 20

        canvas = self._text_pil(canvas, "CONFIDENCE", PX + 22, 238, 13,
                                (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))

        for i, emotion in enumerate(Config.EMOTION_CLASSES):
            y = 258 + i * gap
            p = float(probs[i])
            is_top = (emotion == em)
            c_bgr  = EMOTION_BGR.get(emotion, (160, 160, 160))
            c_rgb  = EMOTION_RGB.get(emotion, (160, 160, 160))

            # Track
            cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + bar_h),
                          (55, 48, 70), -1)
            # Fill
            fw = max(2, int(bar_w * p)) if p > 0.01 else 0
            if fw:
                alpha = 1.0 if is_top else 0.55
                fill  = c_bgr if is_top else tuple(int(v * 0.6) for v in c_bgr)
                cv2.rectangle(canvas, (bar_x, y), (bar_x + fw, y + bar_h), fill, -1)
                if fw > 4:
                    lighter = tuple(min(v + 70, 255) for v in c_bgr)
                    cv2.rectangle(canvas, (bar_x + fw - 3, y),
                                  (bar_x + fw, y + bar_h), lighter, -1)

            # Label
            lc = c_rgb if is_top else (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0])
            canvas = self._text_pil(canvas, emotion[:7], label_x, y + 2, 14, lc)
            # Percentage right
            pct = f"{p * 100:.0f}%"
            canvas = self._text_pil(canvas, pct, bar_x + bar_w + 8, y + 2, 14,
                                    (C_TEXT1[2], C_TEXT1[1], C_TEXT1[0])
                                    if is_top else
                                    (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))

        self._divider(canvas, 524, PX + 16, CW - 16)

        # ── History strip ──
        canvas = self._text_pil(canvas, "HISTORY", PX + 22, 530, 13,
                                (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))
        sq = 14
        gp = 3
        hx = PX + 22
        hy = 548
        for i, past_em in enumerate(self._history):
            col = EMOTION_BGR.get(past_em, (100, 100, 100))
            cv2.rectangle(canvas, (hx + i*(sq+gp), hy),
                          (hx + i*(sq+gp) + sq, hy + sq), col, -1)

        self._divider(canvas, 580, PX + 16, CW - 16)

        # ── Status bar ──
        device_str = f"{'GPU' if torch.cuda.is_available() else 'CPU'}  ·  {infer_ms:.1f} ms"
        canvas = self._text_pil(canvas, device_str, PX + 22, 590, 15,
                                (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))
        canvas = self._text_pil(canvas, "HybridEmotionNet  ·  EfficientNet-B2 + 478 landmarks",
                                PX + 22, 614, 13, (C_TEXT2[2], C_TEXT2[1], C_TEXT2[0]))

        # Accent border line (left edge of panel)
        cv2.rectangle(canvas, (PX, 0), (PX + 4, CH), C_BORDER, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VisageCNN — Real-Time Emotion Recognition")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="Skip camera selector and use this index directly")
    parser.add_argument("--model",     type=str,  default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--ensemble",  action="store_true",
                        help="Average predictions from best + SWA model")
    parser.add_argument("--swa-model", type=str,  default=None,
                        help="Path to SWA checkpoint (default: models/weights/hybrid_swa_final.pth)")
    parser.add_argument("--max-fps",   type=int,  default=0,
                        help="Cap frame rate to N FPS (0 = uncapped)")
    args = parser.parse_args()

    app = VisageCNNApp()

    # Screen 0 — loading
    engine = app.show_loading(args)
    if engine is None:
        sys.exit(1)

    # Screen 1 — camera selection
    cam_idx = app.select_camera(force_index=args.camera_index)

    # Screen 2 — live HUD
    app.run(engine, camera_index=cam_idx, max_fps=args.max_fps)


if __name__ == "__main__":
    main()
