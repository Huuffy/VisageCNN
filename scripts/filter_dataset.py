"""Confidence-based dataset quality filter for VisageCNN.

Two-stage pipeline:
  Stage 1 — MediaPipe FaceMesh: images with no detected face are moved to
             dataset/irrelevant/<emotion>/ immediately (zero-coordinate vectors
             during training make them useless examples).
  Stage 2 — HuggingFace ViT (dima806/facial_emotions_image_detection): images
             where the model predicts a DIFFERENT emotion with high confidence
             (default >= 0.65) are quarantined to dataset/irrelevant/<emotion>/.

Nothing is permanently deleted — irrelevant/ acts as a quarantine so the filter
is fully reversible via --restore.

Confusion pairs (Fear↔Surprised, Angry↔Disgust, Sad↔Neutral, Fear↔Angry) use
a boosted threshold (+0.07 = 0.72) because these are legitimately ambiguous.

Usage
-----
    python scripts/filter_dataset.py --dry-run
    python scripts/filter_dataset.py
    python scripts/filter_dataset.py --threshold 0.55
    python scripts/filter_dataset.py --restore
"""

import os
import sys
import logging
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from visage_er.config import Config

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

LABEL_NORMALISE = {
    'angry': 'Angry', 'anger': 'Angry',
    'disgust': 'Disgust', 'disgusted': 'Disgust',
    'fear': 'Fear', 'fearful': 'Fear',
    'happy': 'Happy', 'happiness': 'Happy',
    'neutral': 'Neutral',
    'sad': 'Sad', 'sadness': 'Sad',
    'surprise': 'Surprised', 'surprised': 'Surprised',
}

CONFUSION_PAIRS = {
    frozenset({'Fear', 'Surprised'}),
    frozenset({'Angry', 'Disgust'}),
    frozenset({'Sad', 'Neutral'}),
    frozenset({'Fear', 'Angry'}),
}

CONFUSION_THRESHOLD_BOOST = 0.07

# ── Per-class asymmetric quarantine thresholds ───────────────────────────────
# For classes the ViT frequently confuses, we use per-prediction thresholds
# instead of a flat value.  This lets us aggressively catch obvious mislabels
# (e.g. Happy in a Disgust folder) while protecting ambiguous but valid images.
#
# Key: ViT prediction when folder_label == X → minimum confidence to quarantine.
# Higher = harder to quarantine (more protection for that label).

# Disgust: ViT constantly confuses with Angry/Fear (curled lip, furrowed brow).
DISGUST_QUARANTINE_THRESHOLDS: Dict[str, float] = {
    'Angry':     0.90,  # Almost never quarantine — heavy feature overlap
    'Fear':      0.85,  # Wrinkled nose/brow overlap
    'Sad':       0.80,  # Subtle disgust can look sad
    'Neutral':   0.70,  # Mild disgust can look neutral
    'Happy':     0.50,  # Clearly wrong — catch aggressively
    'Surprised': 0.50,  # Clearly wrong — catch aggressively
}

# Fear: ViT confuses most strongly with Surprised (wide eyes, raised brows)
# and to a lesser extent Angry (tension, open mouth) and Sad (downturned).
FEAR_QUARANTINE_THRESHOLDS: Dict[str, float] = {
    'Surprised': 0.90,  # Almost never quarantine — widened eyes/raised brows overlap heavily
    'Angry':     0.85,  # Brow tension and open mouth can look fearful
    'Sad':       0.80,  # Downturned, tense expression overlaps with fear
    'Disgust':   0.75,  # Wrinkled brow overlap
    'Neutral':   0.65,  # Subtle fear can read as neutral; be cautious
    'Happy':     0.45,  # Clearly wrong — catch aggressively
}

# Lazy-initialised MediaPipe FaceMesh instances (normal + lenient)
_face_mesh_normal = None
_face_mesh_lenient = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def _ensure_meshes():
    global _face_mesh_normal, _face_mesh_lenient
    if _face_mesh_normal is None:
        import mediapipe as mp
        _face_mesh_normal = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=0.5,
        )
        _face_mesh_lenient = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=0.2,  # lower threshold for extreme expressions
        )


def _upscale_if_small(img_bgr) -> 'np.ndarray':
    """Upscale images whose shortest side is under 112px (MediaPipe minimum)."""
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    if min_dim < 112:
        scale = 256.0 / min_dim
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
    return img_bgr


def has_face(img_bgr) -> bool:
    """Return True if MediaPipe detects a face using normal OR lenient threshold.

    Also upscales small images (FER2013 48x48) before detection — same logic
    as the training pipeline.  Tries normal confidence first; if that fails,
    tries the lenient mesh to handle extreme expressions (e.g. Disgust open mouth).
    """
    import numpy as np
    _ensure_meshes()
    img_bgr = _upscale_if_small(img_bgr)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if _face_mesh_normal.process(rgb).multi_face_landmarks:
        return True
    # Second attempt with lower confidence for extreme expressions
    return bool(_face_mesh_lenient.process(rgb).multi_face_landmarks)


def load_model(model_id: str):
    """Load a HuggingFace image classification pipeline.

    Returns:
        Tuple of (pipeline, label_to_class_map).
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise RuntimeError(
            "'transformers' not installed. Run:  pip install transformers"
        )

    import torch
    device = 0 if torch.cuda.is_available() else -1

    logging.getLogger(__name__).info(f"Loading model {model_id} ...")
    pipe = pipeline('image-classification', model=model_id, device=device)

    label_map: Dict[str, Optional[str]] = {}
    if hasattr(pipe.model, 'config') and hasattr(pipe.model.config, 'id2label'):
        for label_str in pipe.model.config.id2label.values():
            normalised = LABEL_NORMALISE.get(label_str.lower().strip())
            label_map[label_str] = normalised

    return pipe, label_map


def run_filter(
    dataset_base: Path,
    irrelevant_base: Path,
    pipe,
    label_map: Dict[str, Optional[str]],
    threshold: float,
    dry_run: bool,
    batch_size: int,
    logger: logging.Logger,
    relabel: bool = False,
    min_quality: float = 0.15,
    only_class: Optional[str] = None,
) -> Dict[str, Dict]:
    """Run the two-stage filter over all images in dataset_base/train/.

    Stage 1: MediaPipe face detection  — no face → irrelevant/
    Stage 2: ViT low-quality check    — top-1 confidence < min_quality → irrelevant/
               catches occluded faces (hands, extreme angles) where ViT is uncertain
    Stage 3: ViT confidence check      — wrong label with high confidence:
               relabel=False → irrelevant/ (quarantine)
               relabel=True  → move to predicted class folder (free data)

    Returns:
        Dict {emotion: {'kept': n, 'no_face': n, 'quarantined': n, 'relabeled': n}}.
    """
    from PIL import Image

    stats: Dict[str, Dict] = defaultdict(lambda: {'kept': 0, 'no_face': 0, 'quarantined': 0, 'relabeled': 0, 'low_quality': 0})

    train_dir = dataset_base / 'train'
    if not train_dir.is_dir():
        logger.error(f"train/ directory not found at {train_dir}")
        return stats

    for emotion_dir in sorted(train_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        folder_label = emotion_dir.name
        if folder_label not in set(Config.EMOTION_CLASSES):
            continue
        if only_class and folder_label != only_class:
            continue

        images = [f for f in emotion_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            continue

        logger.info(f"  Filtering train/{folder_label} ({len(images)} images)...")

        # --- Stage 1: MediaPipe face detection ---
        # Images with no detected face are NOT immediately quarantined.
        # They are tagged and still passed to Stage 2 (ViT check).
        # Only unreadable/corrupt images are skipped entirely.
        # Rationale: extreme Disgust expressions (square/open mouth) can fail
        # MediaPipe face detection even when a valid face is present.
        stage2_paths = []    # (path, face_detected: bool)
        for img_path in images:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                # Truly unreadable — skip without quarantining
                stats[folder_label]['no_face'] += 1
                continue
            face_found = has_face(img_bgr)
            stage2_paths.append((img_path, face_found))

        # --- Stage 2: ViT confidence check ---
        # Images that failed MediaPipe face detection get a LOWER ViT threshold
        # to keep them (they need stronger ViT evidence to be quarantined).
        NO_FACE_THRESHOLD_PENALTY = 0.10  # raise quarantine bar by 10% for no-face images

        for i in range(0, len(stage2_paths), batch_size):
            batch_chunk = stage2_paths[i:i + batch_size]
            pil_images = []
            valid_paths = []

            for p, face_found in batch_chunk:
                try:
                    img = Image.open(p).convert('RGB')
                    pil_images.append(img)
                    valid_paths.append((p, face_found))
                except Exception:
                    stats[folder_label]['kept'] += 1

            batch_paths = valid_paths  # rename for rest of loop

            if not pil_images:
                continue

            try:
                results = pipe(pil_images, top_k=1)
                if isinstance(results[0], dict):
                    results = [[r] for r in results]
            except Exception as exc:
                logger.warning(f"Inference failed on batch: {exc}")
                stats[folder_label]['kept'] += len(pil_images)
                continue

            for (img_path, face_found), top_result in zip(batch_paths, results):
                predicted_raw = top_result[0]['label']
                confidence = top_result[0]['score']

                predicted = label_map.get(predicted_raw) or LABEL_NORMALISE.get(predicted_raw.lower().strip())

                # ── Stage 2: Low-quality / occlusion check ──────────────────
                # If the ViT can't confidently predict ANY emotion (e.g. hands on
                # face, extreme blur, non-face object), quarantine the image.
                if confidence < min_quality:
                    dest_dir = irrelevant_base / folder_label
                    if not dry_run:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(img_path), str(dest_dir / img_path.name))
                    stats[folder_label]['low_quality'] += 1
                    continue

                if predicted is None or predicted == folder_label:
                    stats[folder_label]['kept'] += 1
                    continue

                # ── Compute effective quarantine threshold ──
                if folder_label == 'Disgust' and predicted in DISGUST_QUARANTINE_THRESHOLDS:
                    effective_threshold = DISGUST_QUARANTINE_THRESHOLDS[predicted]
                elif folder_label == 'Fear' and predicted in FEAR_QUARANTINE_THRESHOLDS:
                    effective_threshold = FEAR_QUARANTINE_THRESHOLDS[predicted]
                else:
                    # Standard threshold + confusion pair boost
                    effective_threshold = threshold
                    pair = frozenset({predicted, folder_label})
                    if pair in CONFUSION_PAIRS:
                        effective_threshold += CONFUSION_THRESHOLD_BOOST

                # Images that failed MediaPipe need even higher ViT confidence to quarantine
                if not face_found:
                    effective_threshold += NO_FACE_THRESHOLD_PENALTY

                if confidence < effective_threshold:
                    stats[folder_label]['kept'] += 1
                    continue

                dest_dir = irrelevant_base / folder_label
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest_dir / img_path.name))
                stats[folder_label]['quarantined'] += 1

    return stats


def move_val_to_train(dataset_base: Path, only_class: Optional[str], logger: logging.Logger):
    """Move images from dataset/val/<emotion>/ to dataset/train/<emotion>/.

    Used before re-filtering a class so the filter sees the full data pool.
    Filename conflicts resolved with _v<n> suffix.
    """
    val_dir = dataset_base / 'val'
    if not val_dir.exists():
        logger.info("No val/ directory found — nothing to move.")
        return

    moved = 0
    for emotion_dir in sorted(val_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name
        if only_class and emotion != only_class:
            continue
        if emotion not in set(Config.EMOTION_CLASSES):
            continue

        dest_dir = dataset_base / 'train' / emotion
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_file in sorted(emotion_dir.iterdir()):
            if not img_file.is_file():
                continue
            dst = dest_dir / img_file.name
            counter = 1
            while dst.exists():
                dst = dest_dir / f"{img_file.stem}_v{counter}{img_file.suffix}"
                counter += 1
            shutil.move(str(img_file), str(dst))
            moved += 1

    logger.info(f"Moved {moved} val images to train/.")


def restore_irrelevant(dataset_base: Path, irrelevant_base: Path, logger: logging.Logger,
                       only_class: Optional[str] = None):
    """Move quarantined images back to dataset/train/<emotion>/.

    Args:
        only_class: If set, only restore this emotion class. Otherwise restore all.
    """
    if not irrelevant_base.exists():
        logger.info("No irrelevant/ directory found -- nothing to restore.")
        return

    restored = 0
    for emotion_dir in irrelevant_base.iterdir():
        if not emotion_dir.is_dir():
            continue
        if only_class and emotion_dir.name != only_class:
            continue
        dest_dir = dataset_base / 'train' / emotion_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_file in emotion_dir.iterdir():
            if img_file.is_file():
                shutil.move(str(img_file), str(dest_dir / img_file.name))
                restored += 1

        # Remove the emotion sub-dir if empty (full restore only removes root)
        try:
            emotion_dir.rmdir()
        except OSError:
            pass

    if not only_class:
        shutil.rmtree(irrelevant_base, ignore_errors=True)
    logger.info(f"Restored {restored} images to dataset/train/" +
                (f" ({only_class} only)" if only_class else "") + ".")


def print_stats(stats: Dict, dry_run: bool, dataset_base: Path):
    """Print per-class summary of filter results."""
    print("\n" + "=" * 78)
    print("FILTER RESULTS" + (" [DRY RUN]" if dry_run else ""))
    print("=" * 78)
    print(f"  {'Class':<12} {'Kept':>8} {'No Face':>9} {'LowQual':>9} {'Mislabel':>10} {'% removed':>10}")
    print("  " + "-" * 62)

    total_kept = 0
    total_no_face = 0
    total_quarantined = 0
    total_low_quality = 0

    for emotion in Config.EMOTION_CLASSES:
        s = stats.get(emotion, {'kept': 0, 'no_face': 0, 'quarantined': 0, 'relabeled': 0, 'low_quality': 0})
        k, nf, q, lq = s['kept'], s['no_face'], s['quarantined'], s.get('low_quality', 0)
        total = k + nf + q + lq
        pct = ((nf + q + lq) / total * 100) if total > 0 else 0.0
        total_kept += k
        total_no_face += nf
        total_quarantined += q
        total_low_quality += lq
        print(f"  {emotion:<12} {k:>8} {nf:>9} {lq:>9} {q:>10} {pct:>9.1f}%")

    total = total_kept + total_no_face + total_quarantined + total_low_quality
    pct = ((total_no_face + total_quarantined + total_low_quality) / total * 100) if total > 0 else 0.0
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<12} {total_kept:>8} {total_no_face:>9} {total_low_quality:>9} {total_quarantined:>10} {pct:>9.1f}%")
    print("=" * 78)

    if not dry_run:
        print(f"\nMoved images saved to: {dataset_base}/irrelevant/")
        print("To restore them: python scripts/filter_dataset.py --restore")
        print("Next step:       python scripts/prepare_data.py")


def main():
    parser = argparse.ArgumentParser(description="Two-stage dataset noise filter (MediaPipe + ViT)")
    parser.add_argument(
        '--model', type=str,
        default='dima806/facial_emotions_image_detection',
        help="HuggingFace model ID for the pre-trained emotion classifier",
    )
    parser.add_argument(
        '--threshold', type=float, default=0.65,
        help="Min ViT confidence for a DIFFERENT class to trigger quarantine (default: 0.65)",
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help="Number of images per ViT inference batch (default: 32)",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Count what would be moved without touching any files",
    )
    parser.add_argument(
        '--relabel', action='store_true',
        help="Move mislabeled images to the predicted class folder instead of irrelevant/",
    )
    parser.add_argument(
        '--restore', action='store_true',
        help="Move all previously quarantined images back to dataset/train/",
    )
    parser.add_argument(
        '--only-class', type=str, default=None,
        metavar='EMOTION',
        help="Process only this emotion class (e.g. Fear). Works with --restore and normal filter.",
    )
    parser.add_argument(
        '--move-val-to-train', action='store_true',
        help="Move val/<class> images to train/<class> before filtering. "
             "Use with --only-class to move a single class (e.g. Fear val → train).",
    )
    parser.add_argument(
        '--min-quality', type=float, default=0.15,
        help="Min ViT top-1 confidence to keep an image at all — images below this "
             "are quarantined as occluded/unclear (hands on face, blur, etc.). "
             "Default: 0.15. Raise to 0.20-0.25 for stricter occlusion filtering.",
    )
    args = parser.parse_args()

    logger = setup_logging()

    dataset_base = Config.DATASET_PATH
    irrelevant_base = dataset_base / 'irrelevant'

    only_class = args.only_class
    if only_class and only_class not in Config.EMOTION_CLASSES:
        logger.error(f"Unknown class '{only_class}'. Valid: {Config.EMOTION_CLASSES}")
        return

    if args.restore:
        restore_irrelevant(dataset_base, irrelevant_base, logger, only_class=only_class)
        return

    if args.move_val_to_train:
        move_val_to_train(dataset_base, only_class=only_class, logger=logger)
        if args.restore or not any([args.threshold != 0.65]):
            # If user only wanted to move, stop here (no --threshold given alongside)
            pass
        # Continue to filter after moving

    pipe, label_map = load_model(args.model)

    logger.info(f"Model label map: {label_map}")
    if only_class:
        logger.info(f"Only-class mode: processing {only_class} only")
    logger.info(f"Threshold: {args.threshold}  (confusion pairs: {args.threshold + CONFUSION_THRESHOLD_BOOST:.2f})")
    logger.info(f"Min quality (occlusion filter): {args.min_quality}")
    logger.info(f"Dry run: {args.dry_run}")

    if args.relabel:
        logger.info("Relabel mode: mislabeled images will be moved to predicted class folder")

    stats = run_filter(
        dataset_base=dataset_base,
        irrelevant_base=irrelevant_base,
        pipe=pipe,
        label_map=label_map,
        threshold=args.threshold,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        logger=logger,
        relabel=args.relabel,
        min_quality=args.min_quality,
        only_class=only_class,
    )

    print_stats(stats, args.dry_run, dataset_base)


if __name__ == "__main__":
    main()
