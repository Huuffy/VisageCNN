"""Import CK+ images from the extracted CK+48 folder into the dataset layout.

Copies images from  CK+48/<emotion>/  into  dataset/train/<emotion>/  and
dataset/val/<emotion>/  at an 80/20 split, skipping the 'contempt' class which
has no equivalent in the 7-class VisageCNN schema.

CK+ images are grayscale 640x490. This script converts them to RGB before saving
so they are consistent with the rest of the dataset.

Usage
-----
    python scripts/import_ckplus.py
    python scripts/import_ckplus.py --src path/to/CK+48
    python scripts/import_ckplus.py --val-ratio 0.2 --dry-run
"""

import os
import sys
import random
import shutil
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from visage_er.config import Config

FOLDER_TO_CLASS = {
    'anger':    'Angry',
    'disgust':  'Disgust',
    'fear':     'Fear',
    'happy':    'Happy',
    'sadness':  'Sad',
    'surprise': 'Surprised',
    'contempt': None,
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def import_ckplus(src: Path, output_base: Path, val_ratio: float, seed: int, dry_run: bool):
    """Copy CK+48 images into the dataset directory.

    Args:
        src: Root of the extracted CK+48 folder (contains emotion subfolders).
        output_base: Root dataset directory (Config.DATASET_PATH).
        val_ratio: Fraction of images per class assigned to val split.
        seed: Random seed for reproducible splitting.
        dry_run: When True, prints counts without copying any files.
    """
    logger = logging.getLogger(__name__)
    rng = random.Random(seed)

    total_copied = 0
    total_skipped = 0

    print("\n" + "=" * 55)
    print("CK+ IMPORT")
    print("=" * 55)

    for folder_name, emotion in FOLDER_TO_CLASS.items():
        src_dir = src / folder_name

        if not src_dir.exists():
            logger.warning(f"Folder not found, skipping: {src_dir}")
            continue

        if emotion is None:
            files = list(src_dir.glob('*'))
            logger.info(f"  contempt: skipped ({len([f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS])} images dropped)")
            continue

        images = [f for f in src_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(images)

        n_val = max(1, int(len(images) * val_ratio))
        splits = {
            'val': images[:n_val],
            'train': images[n_val:],
        }

        for split_name, batch in splits.items():
            dest_dir = output_base / split_name / emotion

            for i, img_path in enumerate(batch):
                dest_path = dest_dir / f"ckplus_{emotion}_{split_name}_{i:04d}.jpg"

                if dry_run:
                    total_copied += 1
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)

                img = cv2.imread(str(img_path))
                if img is None:
                    total_skipped += 1
                    continue

                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                total_copied += 1

        logger.info(
            f"  {emotion:<12} {len(images):>4} images  "
            f"(train={len(splits['train'])}, val={len(splits['val'])})"
            + (" [DRY RUN]" if dry_run else "")
        )

    print("-" * 55)
    print(f"  Total copied : {total_copied}")
    if total_skipped:
        print(f"  Total skipped: {total_skipped}")
    if dry_run:
        print("  [DRY RUN -- no files written]")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Import CK+ dataset into VisageCNN layout")
    parser.add_argument(
        '--src', type=str,
        default=str(Path(__file__).parent.parent / 'CK+48'),
        help="Path to the root CK+48 folder (contains anger/, disgust/, etc.)",
    )
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    setup_logging()
    Config.create_directories()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: Source folder not found: {src}")
        print("Pass the correct path with  --src path/to/CK+48")
        sys.exit(1)

    import_ckplus(
        src=src,
        output_base=Config.DATASET_PATH,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\nNext: delete models/cache/ then run  python scripts/train_hybrid.py")


if __name__ == "__main__":
    main()
