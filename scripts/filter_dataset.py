"""Confidence-based dataset quality filter for VisageCNN.

Loads a pre-trained HuggingFace emotion ViT model and runs every image in the
dataset through it.  Images where the model predicts a DIFFERENT emotion with
high confidence (default >= 0.70) are quarantined to  dataset/rejected/<emotion>/
rather than deleted, so nothing is permanently lost.

The filtering is intentionally conservative:
- Borderline / ambiguous images are kept (model uncertain means it could go either way)
- Only clear, high-confidence disagreements are quarantined
- Plausible confusions (Fear <-> Surprised, Angry <-> Disgust) use a higher threshold

Usage
-----
    python scripts/filter_dataset.py
    python scripts/filter_dataset.py --threshold 0.75
    python scripts/filter_dataset.py --dry-run
    python scripts/filter_dataset.py --restore        # move rejected/ back to dataset/
"""

import os
import sys
import logging
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_model(model_id: str):
    """Load a HuggingFace image classification pipeline.

    Args:
        model_id: HuggingFace model identifier string.

    Returns:
        Tuple of (pipeline, label_to_class_map) where label_to_class_map maps
        model output label strings to TARGET_CLASSES members.
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
    rejected_base: Path,
    pipe,
    label_map: Dict[str, Optional[str]],
    threshold: float,
    dry_run: bool,
    batch_size: int,
    logger: logging.Logger,
) -> Dict[str, Dict[str, int]]:
    """Run the confidence filter over all images in dataset_base.

    Args:
        dataset_base: Root dataset directory containing train/ and val/ splits.
        rejected_base: Directory to move quarantined images into.
        pipe: HuggingFace image-classification pipeline.
        label_map: Mapping from model label strings to TARGET_CLASSES members.
        threshold: Minimum model confidence to trigger quarantine.
        dry_run: When True, count actions without moving any files.
        batch_size: Number of images passed to the pipeline at once.
        logger: Logger instance.

    Returns:
        Nested dict {split: {emotion: {'kept': n, 'quarantined': n}}}.
    """
    from PIL import Image

    stats: Dict[str, Dict] = defaultdict(lambda: defaultdict(lambda: {'kept': 0, 'quarantined': 0}))

    for split_dir in sorted(dataset_base.iterdir()):
        if not split_dir.is_dir() or split_dir.name in ('rejected', 'excess'):
            continue
        split = split_dir.name

        for emotion_dir in sorted(split_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            folder_label = emotion_dir.name

            images = [f for f in emotion_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

            if not images:
                continue

            logger.info(f"  Filtering {split}/{folder_label} ({len(images)} images)...")

            for i in range(0, len(images), batch_size):
                batch_paths = images[i:i + batch_size]
                pil_images = []
                valid_paths = []

                for p in batch_paths:
                    try:
                        img = Image.open(p).convert('RGB')
                        pil_images.append(img)
                        valid_paths.append(p)
                    except Exception:
                        stats[split][folder_label]['kept'] += 1

                if not pil_images:
                    continue

                try:
                    results = pipe(pil_images, top_k=1)
                    if isinstance(results[0], dict):
                        results = [[r] for r in results]
                except Exception as exc:
                    logger.warning(f"Inference failed on batch: {exc}")
                    stats[split][folder_label]['kept'] += len(pil_images)
                    continue

                for img_path, top_result in zip(valid_paths, results):
                    predicted_raw = top_result[0]['label']
                    confidence = top_result[0]['score']

                    predicted = label_map.get(predicted_raw) or LABEL_NORMALISE.get(predicted_raw.lower().strip())

                    if predicted is None or predicted == folder_label:
                        stats[split][folder_label]['kept'] += 1
                        continue

                    effective_threshold = threshold
                    pair = frozenset({predicted, folder_label})
                    if pair in CONFUSION_PAIRS:
                        effective_threshold += CONFUSION_THRESHOLD_BOOST

                    if confidence < effective_threshold:
                        stats[split][folder_label]['kept'] += 1
                        continue

                    reject_dir = rejected_base / split / folder_label
                    if not dry_run:
                        reject_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(img_path), str(reject_dir / img_path.name))

                    stats[split][folder_label]['quarantined'] += 1

    return stats


def restore_rejected(dataset_base: Path, rejected_base: Path, logger: logging.Logger):
    """Move all quarantined images back to their original dataset locations.

    Args:
        dataset_base: Root dataset directory.
        rejected_base: Directory where quarantined images were moved.
        logger: Logger instance.
    """
    if not rejected_base.exists():
        logger.info("No rejected directory found -- nothing to restore.")
        return

    restored = 0
    for split_dir in rejected_base.iterdir():
        if not split_dir.is_dir():
            continue
        for emotion_dir in split_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            dest_dir = dataset_base / split_dir.name / emotion_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_file in emotion_dir.iterdir():
                if img_file.is_file():
                    shutil.move(str(img_file), str(dest_dir / img_file.name))
                    restored += 1

    shutil.rmtree(rejected_base, ignore_errors=True)
    logger.info(f"Restored {restored} images from rejected/.")


def print_stats(stats: Dict, dry_run: bool):
    """Print a per-split, per-class summary of filter results."""
    print("\n" + "=" * 60)
    print("FILTER RESULTS" + (" [DRY RUN]" if dry_run else ""))
    print("=" * 60)

    for split in sorted(stats.keys()):
        print(f"\n  {split.upper()}")
        print(f"  {'Class':<12} {'Kept':>8} {'Quarantined':>12} {'% removed':>10}")
        print("  " + "-" * 44)

        total_kept = 0
        total_quarantined = 0

        for emotion in Config.EMOTION_CLASSES:
            s = stats[split].get(emotion, {'kept': 0, 'quarantined': 0})
            k, q = s['kept'], s['quarantined']
            total = k + q
            pct = (q / total * 100) if total > 0 else 0.0
            total_kept += k
            total_quarantined += q
            print(f"  {emotion:<12} {k:>8} {q:>12} {pct:>9.1f}%")

        total = total_kept + total_quarantined
        pct = (total_quarantined / total * 100) if total > 0 else 0.0
        print("  " + "-" * 44)
        print(f"  {'TOTAL':<12} {total_kept:>8} {total_quarantined:>12} {pct:>9.1f}%")

    print("=" * 60)

    if not dry_run:
        print(f"\nQuarantined images saved to: {Config.DATASET_PATH}/rejected/")
        print("To restore them: python scripts/filter_dataset.py --restore")
        print("Next step: python scripts/balance_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Confidence-based dataset quality filter")
    parser.add_argument(
        '--model', type=str,
        default='dima806/facial_emotions_image_detection',
        help="HuggingFace model ID for the pre-trained emotion classifier",
    )
    parser.add_argument(
        '--threshold', type=float, default=0.70,
        help="Min model confidence for a DIFFERENT class to trigger quarantine (default: 0.70)",
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help="Number of images per inference batch (default: 32)",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Count what would be quarantined without moving any files",
    )
    parser.add_argument(
        '--restore', action='store_true',
        help="Move all previously quarantined images back to the dataset",
    )
    args = parser.parse_args()

    logger = setup_logging()

    dataset_base = Config.DATASET_PATH
    rejected_base = dataset_base / 'rejected'

    if args.restore:
        restore_rejected(dataset_base, rejected_base, logger)
        return

    pipe, label_map = load_model(args.model)

    logger.info(f"Model label map: {label_map}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Dry run: {args.dry_run}")

    stats = run_filter(
        dataset_base=dataset_base,
        rejected_base=rejected_base,
        pipe=pipe,
        label_map=label_map,
        threshold=args.threshold,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        logger=logger,
    )

    print_stats(stats, args.dry_run)


if __name__ == "__main__":
    main()
