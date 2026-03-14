"""Dataset preparation utility for VisageCNN.

Collects all images from the train and validation directories, shuffles them,
and redistributes them with an 80/20 train/val split per emotion class.
"""

import os
import shutil
import random
from pathlib import Path
import logging
from collections import defaultdict
import argparse
import sys

DATASET_TRAIN_PATH = "dataset/train"
DATASET_VAL_PATH = "dataset/val"
TRAIN_RATIO = 0.8
SEED = 42

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def setup_logging() -> logging.Logger:
    """Configure logging to file and stdout.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_preparation.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def count_images(directory: str) -> tuple:
    """Count images in each emotion subdirectory.

    Args:
        directory: Path to a directory containing one subdirectory per emotion class.

    Returns:
        Tuple of (total_count, per_emotion_dict).
    """
    counts = {}
    total = 0

    if not os.path.exists(directory):
        return 0, counts

    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(directory, emotion)
        if os.path.exists(emotion_path) and os.path.isdir(emotion_path):
            try:
                files = [f for f in os.listdir(emotion_path) if f.lower().endswith(SUPPORTED_FORMATS)]
                counts[emotion] = len(files)
                total += len(files)
            except Exception as e:
                logging.error(f"Error counting images in {emotion_path}: {e}")
                counts[emotion] = 0
        else:
            counts[emotion] = 0

    return total, counts


def collect_all_images() -> defaultdict:
    """Gather all image paths from both train and val directories.

    Returns:
        DefaultDict mapping emotion name to list of absolute image paths.
    """
    all_images = defaultdict(list)

    for base_dir in [DATASET_TRAIN_PATH, DATASET_VAL_PATH]:
        if not os.path.exists(base_dir):
            continue
        for emotion in EMOTION_CLASSES:
            emotion_path = os.path.join(base_dir, emotion)
            if not os.path.exists(emotion_path):
                continue
            try:
                for img_file in os.listdir(emotion_path):
                    if img_file.lower().endswith(SUPPORTED_FORMATS):
                        full_path = os.path.join(emotion_path, img_file)
                        if os.path.isfile(full_path):
                            all_images[emotion].append(full_path)
            except Exception as e:
                logging.error(f"Error collecting from {emotion_path}: {e}")

    return all_images


def safe_move(src_path: str, dst_dir: str, new_filename: str) -> bool:
    """Move a file to a destination directory, resolving filename conflicts.

    Args:
        src_path: Absolute path to the source file.
        dst_dir: Destination directory path.
        new_filename: Desired filename at the destination.

    Returns:
        True on success, False on failure.
    """
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, new_filename)

        base, ext = os.path.splitext(new_filename)
        counter = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(dst_dir, f"{base}_{counter}{ext}")
            counter += 1

        shutil.move(src_path, dst_path)
        return True
    except Exception as e:
        logging.error(f"Error moving {src_path}: {e}")
        return False


def balance_datasets() -> bool:
    """Redistribute all images into train/val splits at the configured ratio.

    Collects all images from both existing splits, shuffles them per class,
    then moves them through a temporary staging directory before overwriting
    the original split directories.

    Returns:
        True on success, False on failure.
    """
    random.seed(SEED)

    logging.info("Collecting images from both splits…")
    all_images = collect_all_images()

    total = sum(len(v) for v in all_images.values())
    if total == 0:
        logging.error("No images found.")
        return False

    logging.info(f"Total images collected: {total}")

    temp_dir = "temp_balanced"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for emotion in EMOTION_CLASSES:
            images = all_images.get(emotion, [])
            if not images:
                logging.warning(f"No images for class: {emotion}")
                continue

            random.shuffle(images)
            train_count = int(len(images) * TRAIN_RATIO)

            temp_train = os.path.join(temp_dir, "train", emotion)
            temp_val = os.path.join(temp_dir, "val", emotion)
            os.makedirs(temp_train, exist_ok=True)
            os.makedirs(temp_val, exist_ok=True)

            for i, path in enumerate(images[:train_count]):
                safe_move(path, temp_train, f"train_{emotion}_{i:04d}{Path(path).suffix}")

            for i, path in enumerate(images[train_count:]):
                safe_move(path, temp_val, f"val_{emotion}_{i:04d}{Path(path).suffix}")

            logging.info(f"{emotion}: {train_count} train, {len(images) - train_count} val")

        for old_dir in [DATASET_TRAIN_PATH, DATASET_VAL_PATH]:
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)

        for split in ["train", "val"]:
            src = os.path.join(temp_dir, split)
            dst = DATASET_TRAIN_PATH if split == "train" else DATASET_VAL_PATH
            if os.path.exists(src):
                shutil.move(src, dst)

        shutil.rmtree(temp_dir)
        logging.info("Dataset preparation complete.")
        return True

    except Exception as e:
        logging.error(f"Error during preparation: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def print_report(before_train, before_val, after_train, after_val):
    """Print a before/after comparison of per-class sample counts.

    Args:
        before_train: Per-emotion image counts before balancing (train split).
        before_val: Per-emotion image counts before balancing (val split).
        after_train: Per-emotion image counts after balancing (train split).
        after_val: Per-emotion image counts after balancing (val split).
    """
    print("\n" + "=" * 60)
    print("DATASET PREPARATION REPORT")
    print("=" * 60)

    print(f"\n{'BEFORE':}")
    print(f"{'Class':<12} {'Train':<8} {'Val':<8} {'Total':<8}")
    print("-" * 40)
    for emotion in EMOTION_CLASSES:
        t = before_train.get(emotion, 0)
        v = before_val.get(emotion, 0)
        print(f"{emotion:<12} {t:<8} {v:<8} {t + v:<8}")

    print(f"\n{'AFTER':}")
    print(f"{'Class':<12} {'Train':<8} {'Val':<8} {'Total':<8} {'Train%':<8}")
    print("-" * 50)
    for emotion in EMOTION_CLASSES:
        t = after_train.get(emotion, 0)
        v = after_val.get(emotion, 0)
        total = t + v
        pct = (t / total * 100) if total > 0 else 0
        print(f"{emotion:<12} {t:<8} {v:<8} {total:<8} {pct:<7.1f}%")

    print("=" * 60)


def main():
    """Entry point — parse arguments and run dataset preparation."""
    parser = argparse.ArgumentParser(description='VisageCNN Dataset Preparation')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    global TRAIN_RATIO, SEED
    TRAIN_RATIO = args.train_ratio
    SEED = args.seed

    logger = setup_logging()
    logger.info(f"Starting dataset preparation (train ratio: {TRAIN_RATIO:.0%})")

    _, before_train = count_images(DATASET_TRAIN_PATH)
    _, before_val = count_images(DATASET_VAL_PATH)

    if not balance_datasets():
        logger.error("Dataset preparation failed.")
        return

    _, after_train = count_images(DATASET_TRAIN_PATH)
    _, after_val = count_images(DATASET_VAL_PATH)

    print_report(before_train, before_val, after_train, after_val)
    logger.info("Done.")


if __name__ == "__main__":
    main()
