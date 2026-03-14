"""Class balance checker and rebalancer for VisageCNN.

Analyses the current per-class image counts and optionally:
  - Caps majority classes (moves excess to dataset/excess/<emotion>/) so no
    single class dominates training more than MAX_RATIO x the minority class.
  - Reports the imbalance ratio and recommended action.

Nothing is deleted.  Capped images go to  dataset/excess/  and can be restored
with  --restore.

Usage
-----
    python scripts/balance_dataset.py          # just report counts
    python scripts/balance_dataset.py --cap    # cap + report
    python scripts/balance_dataset.py --restore
    python scripts/balance_dataset.py --cap --max-ratio 4.0
"""

import os
import sys
import random
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from visage_er.config import Config

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
DEFAULT_MAX_RATIO = 4.0


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def count_images(dataset_base: Path) -> Dict[str, Dict[str, int]]:
    """Count images per split and per class.

    Args:
        dataset_base: Root dataset directory containing train/ and val/ splits.

    Returns:
        Nested dict {split: {emotion: count}}.
    """
    counts: Dict[str, Dict[str, int]] = {}

    for split_dir in sorted(dataset_base.iterdir()):
        if not split_dir.is_dir() or split_dir.name in ('rejected', 'excess'):
            continue
        split = split_dir.name
        counts[split] = {}

        for emotion in Config.EMOTION_CLASSES:
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                counts[split][emotion] = 0
                continue
            counts[split][emotion] = sum(
                1 for f in emotion_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )

    return counts


def print_report(counts: Dict[str, Dict[str, int]], cap_ceiling: Dict[str, int] = None):
    """Print a per-split class count table with imbalance analysis.

    Args:
        counts: Nested dict from count_images().
        cap_ceiling: Optional dict {emotion: ceiling} to annotate the table.
    """
    print("\n" + "=" * 65)
    print("DATASET CLASS BALANCE REPORT")
    print("=" * 65)

    for split in sorted(counts.keys()):
        split_counts = counts[split]
        values = [v for v in split_counts.values() if v > 0]

        if not values:
            continue

        min_count = min(values)
        max_count = max(values)
        ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"\n  {split.upper()}  (total={sum(values):,}  ratio={ratio:.1f}x)")
        print(f"  {'Class':<12} {'Count':>8} {'vs min':>8}" +
              ("  {'Cap':<8}" if cap_ceiling else ""))
        print("  " + "-" * 40)

        for emotion in Config.EMOTION_CLASSES:
            count = split_counts.get(emotion, 0)
            vs_min = f"{count / min_count:.1f}x" if min_count > 0 and count > 0 else "-"
            cap_str = ""
            if cap_ceiling and emotion in cap_ceiling:
                c = cap_ceiling[emotion]
                if c < count:
                    cap_str = f"  -> {c}"
            print(f"  {emotion:<12} {count:>8} {vs_min:>8}{cap_str}")

    print("\n" + "=" * 65)


def compute_ceiling(counts: Dict[str, int], max_ratio: float) -> Dict[str, int]:
    """Compute the per-class image cap based on max_ratio * minority class count.

    Args:
        counts: {emotion: count} for the training split.
        max_ratio: Maximum allowed ratio between largest and smallest class.

    Returns:
        Dict {emotion: ceiling} — classes at or below ceiling are unchanged.
    """
    active = {e: c for e, c in counts.items() if c > 0}
    if not active:
        return {}

    minority = min(active.values())
    ceiling = int(minority * max_ratio)

    return {emotion: ceiling for emotion, count in active.items() if count > ceiling}


def cap_classes(
    dataset_base: Path,
    excess_base: Path,
    ceilings: Dict[str, int],
    seed: int,
    dry_run: bool,
    logger: logging.Logger,
):
    """Move excess images to excess_base to cap majority classes.

    Randomly samples which images to cap so the selection is reproducible.

    Args:
        dataset_base: Root dataset directory.
        excess_base: Destination for excess images (dataset/excess/).
        ceilings: Dict {emotion: max_count} for the train split only.
        seed: Random seed.
        dry_run: When True, reports without moving files.
        logger: Logger instance.
    """
    rng = random.Random(seed)
    train_dir = dataset_base / 'train'

    for emotion, ceiling in ceilings.items():
        emotion_dir = train_dir / emotion
        if not emotion_dir.exists():
            continue

        images = [
            f for f in emotion_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) <= ceiling:
            continue

        rng.shuffle(images)
        to_move = images[ceiling:]
        dest_dir = excess_base / 'train' / emotion

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in to_move:
                shutil.move(str(img), str(dest_dir / img.name))

        logger.info(
            f"  {emotion:<12} capped at {ceiling}  "
            f"(moved {len(to_move)} to excess/)"
            + (" [DRY RUN]" if dry_run else "")
        )


def restore_excess(dataset_base: Path, excess_base: Path, logger: logging.Logger):
    """Move all capped images back to their original dataset locations.

    Args:
        dataset_base: Root dataset directory.
        excess_base: Directory where capped images were moved.
        logger: Logger instance.
    """
    if not excess_base.exists():
        logger.info("No excess directory found -- nothing to restore.")
        return

    restored = 0
    for split_dir in excess_base.iterdir():
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

    shutil.rmtree(excess_base, ignore_errors=True)
    logger.info(f"Restored {restored} excess images.")


def main():
    parser = argparse.ArgumentParser(description="VisageCNN dataset class balance tool")
    parser.add_argument(
        '--cap', action='store_true',
        help="Cap majority classes to MAX_RATIO * minority class count",
    )
    parser.add_argument(
        '--max-ratio', type=float, default=DEFAULT_MAX_RATIO,
        help=f"Max allowed ratio between largest and smallest class (default: {DEFAULT_MAX_RATIO})",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for selecting which excess images to move (default: 42)",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Show what would be capped without moving any files",
    )
    parser.add_argument(
        '--restore', action='store_true',
        help="Restore all previously capped images back to the dataset",
    )
    args = parser.parse_args()

    logger = setup_logging()

    dataset_base = Config.DATASET_PATH
    excess_base = dataset_base / 'excess'

    if args.restore:
        restore_excess(dataset_base, excess_base, logger)
        counts = count_images(dataset_base)
        print_report(counts)
        return

    counts = count_images(dataset_base)

    if 'train' not in counts or not any(counts['train'].values()):
        print("No training images found. Run download_data.py first.")
        return

    train_counts = counts['train']
    active = {e: c for e, c in train_counts.items() if c > 0}
    minority = min(active.values())
    majority = max(active.values())
    ratio = majority / minority if minority > 0 else float('inf')

    ceilings = compute_ceiling(train_counts, args.max_ratio)
    print_report(counts, cap_ceiling=ceilings if args.cap or args.dry_run else None)

    print(f"\n  Imbalance ratio  : {ratio:.1f}x  (minority={minority:,}, majority={majority:,})")
    print(f"  Max ratio target : {args.max_ratio}x")

    if not ceilings:
        print(f"  Status           : BALANCED -- no classes exceed {args.max_ratio}x minority")
        return

    minority_class = min(active, key=active.get)
    majority_class = max(active, key=active.get)
    print(f"  Minority class   : {minority_class} ({minority:,})")
    print(f"  Majority class   : {majority_class} ({majority:,})")
    print(f"\n  Classes to cap   : {list(ceilings.keys())}")

    if not args.cap and not args.dry_run:
        print("\n  Run with --cap to apply capping.")
        print("  Run with --dry-run to preview without changes.")
        return

    print()
    cap_classes(
        dataset_base=dataset_base,
        excess_base=excess_base,
        ceilings=ceilings,
        seed=args.seed,
        dry_run=args.dry_run,
        logger=logger,
    )

    if not args.dry_run:
        print("\n  Final counts after capping:")
        counts_after = count_images(dataset_base)
        print_report(counts_after)
        print(f"\n  Excess images saved to: {excess_base}")
        print("  To restore them: python scripts/balance_dataset.py --restore")
        print("  Next step: delete models/cache/ then run  python scripts/train_hybrid.py")


if __name__ == "__main__":
    main()
