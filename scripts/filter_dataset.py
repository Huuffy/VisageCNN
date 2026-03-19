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

# Lazy-initialised MediaPipe FaceMesh (static image mode)
_face_mesh = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def has_face(img_bgr) -> bool:
    """Return True if MediaPipe FaceMesh detects at least one face."""
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return bool(_face_mesh.process(rgb).multi_face_landmarks)


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
) -> Dict[str, Dict]:
    """Run the two-stage filter over all images in dataset_base/train/.

    Stage 1: MediaPipe face detection  — no face → irrelevant/
    Stage 2: ViT confidence check      — wrong label with high confidence → irrelevant/

    Returns:
        Dict {emotion: {'kept': n, 'no_face': n, 'quarantined': n}}.
    """
    from PIL import Image

    stats: Dict[str, Dict] = defaultdict(lambda: {'kept': 0, 'no_face': 0, 'quarantined': 0})

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

        images = [f for f in emotion_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            continue

        logger.info(f"  Filtering train/{folder_label} ({len(images)} images)...")

        # --- Stage 1: MediaPipe face detection ---
        stage2_paths = []
        for img_path in images:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None or not has_face(img_bgr):
                dest_dir = irrelevant_base / folder_label
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest_dir / img_path.name))
                stats[folder_label]['no_face'] += 1
            else:
                stage2_paths.append(img_path)

        # --- Stage 2: ViT confidence check ---
        for i in range(0, len(stage2_paths), batch_size):
            batch_paths = stage2_paths[i:i + batch_size]
            pil_images = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    pil_images.append(img)
                    valid_paths.append(p)
                except Exception:
                    stats[folder_label]['kept'] += 1

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

            for img_path, top_result in zip(valid_paths, results):
                predicted_raw = top_result[0]['label']
                confidence = top_result[0]['score']

                predicted = label_map.get(predicted_raw) or LABEL_NORMALISE.get(predicted_raw.lower().strip())

                if predicted is None or predicted == folder_label:
                    stats[folder_label]['kept'] += 1
                    continue

                effective_threshold = threshold
                pair = frozenset({predicted, folder_label})
                if pair in CONFUSION_PAIRS:
                    effective_threshold += CONFUSION_THRESHOLD_BOOST

                if confidence < effective_threshold:
                    stats[folder_label]['kept'] += 1
                    continue

                dest_dir = irrelevant_base / folder_label
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest_dir / img_path.name))

                stats[folder_label]['quarantined'] += 1

    return stats


def restore_irrelevant(dataset_base: Path, irrelevant_base: Path, logger: logging.Logger):
    """Move all quarantined images back to dataset/train/<emotion>/."""
    if not irrelevant_base.exists():
        logger.info("No irrelevant/ directory found -- nothing to restore.")
        return

    restored = 0
    for emotion_dir in irrelevant_base.iterdir():
        if not emotion_dir.is_dir():
            continue
        dest_dir = dataset_base / 'train' / emotion_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_file in emotion_dir.iterdir():
            if img_file.is_file():
                shutil.move(str(img_file), str(dest_dir / img_file.name))
                restored += 1

    shutil.rmtree(irrelevant_base, ignore_errors=True)
    logger.info(f"Restored {restored} images to dataset/train/.")


def print_stats(stats: Dict, dry_run: bool, dataset_base: Path):
    """Print per-class summary of filter results."""
    print("\n" + "=" * 68)
    print("FILTER RESULTS" + (" [DRY RUN]" if dry_run else ""))
    print("=" * 68)
    print(f"  {'Class':<12} {'Kept':>8} {'No Face':>9} {'Mislabeled':>11} {'% removed':>10}")
    print("  " + "-" * 52)

    total_kept = 0
    total_no_face = 0
    total_quarantined = 0

    for emotion in Config.EMOTION_CLASSES:
        s = stats.get(emotion, {'kept': 0, 'no_face': 0, 'quarantined': 0})
        k, nf, q = s['kept'], s['no_face'], s['quarantined']
        total = k + nf + q
        pct = ((nf + q) / total * 100) if total > 0 else 0.0
        total_kept += k
        total_no_face += nf
        total_quarantined += q
        print(f"  {emotion:<12} {k:>8} {nf:>9} {q:>11} {pct:>9.1f}%")

    total = total_kept + total_no_face + total_quarantined
    pct = ((total_no_face + total_quarantined) / total * 100) if total > 0 else 0.0
    print("  " + "-" * 52)
    print(f"  {'TOTAL':<12} {total_kept:>8} {total_no_face:>9} {total_quarantined:>11} {pct:>9.1f}%")
    print("=" * 68)

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
        '--restore', action='store_true',
        help="Move all previously quarantined images back to dataset/train/",
    )
    args = parser.parse_args()

    logger = setup_logging()

    dataset_base = Config.DATASET_PATH
    irrelevant_base = dataset_base / 'irrelevant'

    if args.restore:
        restore_irrelevant(dataset_base, irrelevant_base, logger)
        return

    pipe, label_map = load_model(args.model)

    logger.info(f"Model label map: {label_map}")
    logger.info(f"Threshold: {args.threshold}  (confusion pairs: {args.threshold + CONFUSION_THRESHOLD_BOOST:.2f})")
    logger.info(f"Dry run: {args.dry_run}")

    stats = run_filter(
        dataset_base=dataset_base,
        irrelevant_base=irrelevant_base,
        pipe=pipe,
        label_map=label_map,
        threshold=args.threshold,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        logger=logger,
    )

    print_stats(stats, args.dry_run, dataset_base)


if __name__ == "__main__":
    main()
