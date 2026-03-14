"""Multi-source facial emotion dataset downloader for VisageCNN.

Downloads images from verified public sources, normalises label conventions,
converts grayscale to RGB, upscales images too small for MediaPipe, and saves
everything into  dataset/train/<emotion>/  and  dataset/val/<emotion>/.

Datasets
--------
  fer2013      AutumnQiu/fer2013              HuggingFace  35k  48x48 grayscale
  rafdb        deanngkl/raf-db-7emotions      HuggingFace  20k  real-world color
  affectnet    Piro17/affectnethq             HuggingFace  28k  high-res color
  expw         Mengyuh/ExpW_preprocessed      HuggingFace  2.8k wild faces (subset)
  ckplus       shawon10/ckplus                Kaggle       ~1k  lab poses

Prerequisites
-------------
  pip install datasets          # for HuggingFace sources
  pip install kaggle            # for ckplus only
  # Kaggle API key: https://www.kaggle.com/settings -> API -> Create New Token
  # Save downloaded kaggle.json to  ~/.kaggle/kaggle.json  (or %USERPROFILE%/.kaggle/kaggle.json)

Usage
-----
  python scripts/download_data.py
  python scripts/download_data.py --sources fer2013 rafdb affectnet
  python scripts/download_data.py --sources ckplus
  python scripts/download_data.py --val-ratio 0.15 --dry-run
"""

import os
import sys
import logging
import random
import argparse
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

sys.path.insert(0, str(Path(__file__).parent.parent))

from visage_er.config import Config

TARGET_CLASSES = Config.EMOTION_CLASSES
MIN_FACE_DIM = 96
UPSCALE_TARGET = 256

LABEL_MAP: Dict[str, Optional[str]] = {
    'angry': 'Angry',
    'anger': 'Angry',
    'disgust': 'Disgust',
    'disgusted': 'Disgust',
    'fear': 'Fear',
    'fearful': 'Fear',
    'afraid': 'Fear',
    'happy': 'Happy',
    'happiness': 'Happy',
    'joy': 'Happy',
    'neutral': 'Neutral',
    'calm': 'Neutral',
    'sad': 'Sad',
    'sadness': 'Sad',
    'surprise': 'Surprised',
    'surprised': 'Surprised',
    'contempt': None,
    'bored': None,
    'excited': None,
    'other': None,
    'uncertain': None,
}

HF_SOURCES: Dict[str, dict] = {
    'fer2013': {
        'hf_path': 'AutumnQiu/fer2013',
        'splits': ['train', 'valid', 'test'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'description': 'FER2013 48x48 greyscale (AutumnQiu/fer2013)',
    },
    'rafdb': {
        'hf_path': 'deanngkl/raf-db-7emotions',
        'splits': ['train'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        'description': 'RAF-DB real-world color (deanngkl/raf-db-7emotions)',
    },
    'affectnet': {
        'hf_path': 'Piro17/affectnethq',
        'splits': ['train'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'description': 'AffectNet-HQ high-res color (Piro17/affectnethq)',
    },
    'expw': {
        'hf_path': 'Mengyuh/ExpW_preprocessed',
        'splits': ['train'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'description': 'ExpW wild faces 224x224 subset (Mengyuh/ExpW_preprocessed)',
    },
    'disgust_extra': {
        'hf_path': 'Mauregato/affectnet_short',
        'splits': ['train', 'val'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['anger', 'surprise', 'contempt', 'happy', 'neutral', 'fear', 'sad', 'disgust'],
        'class_filter': ['Disgust'],
        'description': 'AffectNet-Short -- Disgust class only (Mauregato/affectnet_short)',
    },
    'fear_extra': {
        'hf_path': 'Mauregato/affectnet_short',
        'splits': ['train', 'val'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['anger', 'surprise', 'contempt', 'happy', 'neutral', 'fear', 'sad', 'disgust'],
        'class_filter': ['Fear'],
        'description': 'AffectNet-Short -- Fear class only (Mauregato/affectnet_short)',
    },
    'sad_extra': {
        'hf_path': 'Mauregato/affectnet_short',
        'splits': ['train', 'val'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['anger', 'surprise', 'contempt', 'happy', 'neutral', 'fear', 'sad', 'disgust'],
        'class_filter': ['Sad'],
        'description': 'AffectNet-Short -- Sad class only (Mauregato/affectnet_short)',
    },
    'angry_extra': {
        'hf_path': 'Mauregato/affectnet_short',
        'splits': ['train', 'val'],
        'image_key': 'image',
        'label_key': 'label',
        'int_labels': ['anger', 'surprise', 'contempt', 'happy', 'neutral', 'fear', 'sad', 'disgust'],
        'class_filter': ['Angry'],
        'description': 'AffectNet-Short -- Angry class only (Mauregato/affectnet_short)',
    },
}

KAGGLE_SOURCES: Dict[str, dict] = {
    'ckplus': {
        'kaggle_id': 'shawon10/ckplus',
        'description': 'CK+ lab poses CC0 (shawon10/ckplus)',
        'label_mode': 'folders',
        'folder_label_map': {
            'anger': 'Angry', 'angry': 'Angry',
            'disgust': 'Disgust',
            'fear': 'Fear',
            'happiness': 'Happy', 'happy': 'Happy',
            'neutral': 'Neutral',
            'sadness': 'Sad', 'sad': 'Sad',
            'surprise': 'Surprised', 'surprised': 'Surprised',
            'contempt': None,
        },
    },
}

ALL_SOURCES = list(HF_SOURCES.keys()) + list(KAGGLE_SOURCES.keys())
HF_SOURCES_LIST = list(HF_SOURCES.keys())


def setup_logging() -> logging.Logger:
    """Configure logging to stdout and a log file using ASCII-safe formatting."""
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_data.log', encoding='utf-8'),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def normalise_label(raw, label_names: List[str]) -> Optional[str]:
    """Convert a raw dataset label (int index or string) to a TARGET_CLASSES member.

    Args:
        raw: Integer index or string label from the source dataset.
        label_names: Ordered string names for integer labels (used when raw is int).

    Returns:
        A member of TARGET_CLASSES, or None if the label should be dropped.
    """
    if isinstance(raw, int):
        if 0 <= raw < len(label_names):
            raw = label_names[raw]
        else:
            return None

    return LABEL_MAP.get(str(raw).lower().strip(), None)


def prepare_image(pil_image) -> Optional[np.ndarray]:
    """Convert a PIL image to a uint8 RGB numpy array, upscaling tiny images.

    Greyscale and RGBA images are converted to 3-channel RGB. Images whose
    shortest side is below MIN_FACE_DIM are bicubic-upscaled so that MediaPipe
    can reliably detect landmarks during the cache-build phase.

    Args:
        pil_image: A PIL.Image object from HuggingFace datasets.

    Returns:
        HxWx3 uint8 RGB array, or None on failure.
    """
    import cv2

    try:
        img = np.array(pil_image)
    except Exception:
        return None

    if img is None or img.size == 0:
        return None

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)

    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    h, w = img.shape[:2]
    if min(h, w) < MIN_FACE_DIM:
        scale = UPSCALE_TARGET / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return img


def save_image(img: np.ndarray, dest_path: Path) -> bool:
    """Write a uint8 RGB image array to disk as JPEG.

    Args:
        img: HxWx3 uint8 RGB array.
        dest_path: Full destination path including filename.

    Returns:
        True on success, False on failure.
    """
    import cv2

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as exc:
        logging.warning(f"Could not save {dest_path}: {exc}")
        return False


def download_hf_source(
    source_key: str,
    cfg: dict,
    output_base: Path,
    val_ratio: float,
    seed: int,
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Download one HuggingFace dataset and write images to the output directory.

    Args:
        source_key: Short identifier used in output filenames.
        cfg: Entry from HF_SOURCES.
        output_base: Root dataset directory (Config.DATASET_PATH).
        val_ratio: Fraction of images assigned to the val split.
        seed: Random seed for reproducible splitting.
        dry_run: When True, counts images without writing any files.
        logger: Logger instance.

    Returns:
        Dict mapping emotion class name to total images saved across both splits.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' not installed. Run:  pip install datasets")
        return {}

    logger.info(f"Loading {cfg['description']} ...")
    try:
        dataset = load_dataset(cfg['hf_path'])
    except Exception as exc:
        logger.warning(f"Could not load {cfg['hf_path']}: {exc}")
        return {}

    rng = random.Random(seed)
    per_class: Dict[str, list] = defaultdict(list)

    for split_name in cfg['splits']:
        if split_name not in dataset:
            continue

        split = dataset[split_name]

        label_names: List[str] = cfg['int_labels']
        try:
            feature = split.features[cfg['label_key']]
            if hasattr(feature, 'names'):
                label_names = feature.names
        except Exception:
            pass

        logger.info(f"  Processing split '{split_name}' ({len(split)} samples)...")

        class_filter = cfg.get('class_filter')

        for row in split:
            emotion = normalise_label(row[cfg['label_key']], label_names)
            if emotion is None or emotion not in TARGET_CLASSES:
                continue
            if class_filter and emotion not in class_filter:
                continue
            per_class[emotion].append(row[cfg['image_key']])

    saved: Dict[str, int] = defaultdict(int)

    for emotion, images in per_class.items():
        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio))

        for split_name, batch in [('val', images[:n_val]), ('train', images[n_val:])]:
            dest_dir = output_base / split_name / emotion
            for i, pil_img in enumerate(batch):
                img = prepare_image(pil_img)
                if img is None:
                    continue
                filename = f"{source_key}_{emotion}_{split_name}_{i:06d}.jpg"
                if dry_run or save_image(img, dest_dir / filename):
                    saved[emotion] += 1

        logger.info(
            f"  {emotion}: {len(images)} total  "
            f"(train={len(images) - n_val}, val={n_val})"
            + (" [DRY RUN]" if dry_run else "")
        )

    return dict(saved)


def download_kaggle_source(
    source_key: str,
    cfg: dict,
    output_base: Path,
    val_ratio: float,
    seed: int,
    dry_run: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Download one Kaggle dataset, walk its folder tree, and write images.

    Requires the kaggle package (pip install kaggle) and a valid API token at
    ~/.kaggle/kaggle.json.

    Args:
        source_key: Short identifier used in output filenames.
        cfg: Entry from KAGGLE_SOURCES.
        output_base: Root dataset directory.
        val_ratio: Fraction of images assigned to the val split.
        seed: Random seed.
        dry_run: When True, counts images without writing any files.
        logger: Logger instance.

    Returns:
        Dict mapping emotion class name to total images saved across both splits.
    """
    try:
        import kaggle
        kaggle.api.authenticate()
    except ImportError:
        logger.error(
            "'kaggle' not installed. Run:  pip install kaggle\n"
            "Then get your API token from https://www.kaggle.com/settings -> API\n"
            "and save it to  ~/.kaggle/kaggle.json"
        )
        return {}
    except Exception as exc:
        logger.error(
            f"Kaggle authentication failed: {exc}\n"
            "Download your kaggle.json from https://www.kaggle.com/settings -> API\n"
            "and save it to  ~/.kaggle/kaggle.json  (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)"
        )
        return {}

    dataset_id = cfg['kaggle_id']
    folder_label_map: Dict[str, Optional[str]] = cfg.get('folder_label_map', {})

    logger.info(f"Loading {cfg['description']} ...")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    per_class: Dict[str, list] = defaultdict(list)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            kaggle.api.dataset_download_files(dataset_id, path=tmp_dir, unzip=True, quiet=False)
        except Exception as exc:
            logger.warning(f"Could not download {dataset_id}: {exc}")
            return {}

        for img_path in Path(tmp_dir).rglob('*'):
            if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
                continue

            folder_name = img_path.parent.name.lower().strip()
            emotion = folder_label_map.get(folder_name)

            if emotion is None:
                emotion = LABEL_MAP.get(folder_name)

            if emotion is None or emotion not in TARGET_CLASSES:
                continue

            per_class[emotion].append(img_path)

    rng = random.Random(seed)
    saved: Dict[str, int] = defaultdict(int)

    for emotion, paths in per_class.items():
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))

        for split_name, batch in [('val', paths[:n_val]), ('train', paths[n_val:])]:
            dest_dir = output_base / split_name / emotion
            for i, src_path in enumerate(batch):
                import cv2
                img_bgr = cv2.imread(str(src_path))
                if img_bgr is None:
                    continue
                h, w = img_bgr.shape[:2]
                if min(h, w) < MIN_FACE_DIM:
                    scale = UPSCALE_TARGET / min(h, w)
                    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                filename = f"{source_key}_{emotion}_{split_name}_{i:06d}.jpg"
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest_dir / filename), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved[emotion] += 1

        logger.info(
            f"  {emotion}: {len(paths)} total  "
            f"(train={len(paths) - n_val}, val={n_val})"
            + (" [DRY RUN]" if dry_run else "")
        )

    return dict(saved)


def print_summary(results: Dict[str, Dict[str, int]], output_base: Path, dry_run: bool):
    """Print a formatted per-source and per-class download summary."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("DOWNLOAD SUMMARY" + (" [DRY RUN -- no files written]" if dry_run else ""))
    print(sep)

    header = f"{'Source':<20} " + "  ".join(f"{e[:4]:>6}" for e in TARGET_CLASSES)
    print(f"\n{header}")
    print("-" * 65)

    total_per_class: Dict[str, int] = defaultdict(int)
    for src_key, per_class in results.items():
        row = f"{src_key:<20} "
        for emotion in TARGET_CLASSES:
            count = per_class.get(emotion, 0)
            total_per_class[emotion] += count
            row += f"  {count:>6}"
        print(row)

    print("-" * 65)
    total_row = f"{'TOTAL':<20} "
    grand_total = sum(total_per_class[e] for e in TARGET_CLASSES)
    for emotion in TARGET_CLASSES:
        total_row += f"  {total_per_class[emotion]:>6}"
    print(total_row)
    print(f"\nGrand total: {grand_total:,} images")

    if not dry_run and grand_total > 0:
        print(f"\nImages written to: {output_base}")
        print("Next: delete models/cache/ then run  python scripts/train_hybrid.py")

    print(sep)


def main():
    """Entry point -- parse arguments and run the downloader."""
    parser = argparse.ArgumentParser(
        description="VisageCNN multi-source dataset downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "HuggingFace sources (pip install datasets):\n"
            "  fer2013     AutumnQiu/fer2013          35k 48x48 grayscale FER2013\n"
            "  rafdb       deanngkl/raf-db-7emotions  20k real-world color RAF-DB\n"
            "  affectnet   Piro17/affectnethq         28k high-res color AffectNet\n"
            "  expw        Mengyuh/ExpW_preprocessed  2.8k wild faces subset\n\n"
            "Kaggle sources (pip install kaggle + kaggle.json API token):\n"
            "  ckplus      shawon10/ckplus            ~1k lab poses CK+\n"
        ),
    )
    parser.add_argument(
        '--sources', nargs='+', default=ALL_SOURCES,
        choices=ALL_SOURCES, metavar='SOURCE',
        help=f"Datasets to download. Options: {ALL_SOURCES}",
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help="Fraction of each source assigned to val split (default: 0.2)",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for train/val splitting (default: 42)",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Count images and print the summary without writing any files",
    )
    args = parser.parse_args()

    logger = setup_logging()

    Config.create_directories()
    output_base = Config.DATASET_PATH

    logger.info(f"Output directory: {output_base}")
    logger.info(f"Sources: {args.sources}")
    logger.info(f"Val ratio: {args.val_ratio}")
    logger.info(f"Dry run: {args.dry_run}")

    results: Dict[str, Dict[str, int]] = {}

    for source_key in args.sources:
        print(f"\n{'=' * 50}")
        if source_key in HF_SOURCES:
            logger.info(f"Source: {source_key} -- {HF_SOURCES[source_key]['description']}")
            print(f"{'=' * 50}")
            results[source_key] = download_hf_source(
                source_key=source_key,
                cfg=HF_SOURCES[source_key],
                output_base=output_base,
                val_ratio=args.val_ratio,
                seed=args.seed,
                dry_run=args.dry_run,
                logger=logger,
            )
        elif source_key in KAGGLE_SOURCES:
            logger.info(f"Source: {source_key} -- {KAGGLE_SOURCES[source_key]['description']}")
            print(f"{'=' * 50}")
            results[source_key] = download_kaggle_source(
                source_key=source_key,
                cfg=KAGGLE_SOURCES[source_key],
                output_base=output_base,
                val_ratio=args.val_ratio,
                seed=args.seed,
                dry_run=args.dry_run,
                logger=logger,
            )

    print_summary(results, output_base, args.dry_run)


if __name__ == "__main__":
    main()
