#!/usr/bin/env python3
"""
Fixed Dataset Balancer for VisageCNN
Properly handles 80/20 split without data loss
"""

import os
import shutil
import random
from pathlib import Path
import logging
from collections import defaultdict
import argparse
import sys

# Configuration
DATASET_IMAGES_PATH = "dataset/train"
DATASET_VAL_PATH = "dataset/val"
TRAIN_RATIO = 0.8
SEED = 42

# Emotion classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_balancing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def count_images(directory):
    """Count images in each emotion directory"""
    emotion_counts = {}
    total_count = 0

    if not os.path.exists(directory):
        return 0, emotion_counts

    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(directory, emotion)
        if os.path.exists(emotion_path) and os.path.isdir(emotion_path):
            try:
                image_files = [f for f in os.listdir(emotion_path)
                             if f.lower().endswith(SUPPORTED_FORMATS)]
                emotion_counts[emotion] = len(image_files)
                total_count += len(image_files)
            except Exception as e:
                logging.error(f"Error counting images in {emotion_path}: {e}")
                emotion_counts[emotion] = 0
        else:
            emotion_counts[emotion] = 0

    return total_count, emotion_counts

def collect_all_images():
    """Collect all images from both train and val directories"""
    logger = logging.getLogger(__name__)
    all_images = defaultdict(list)

    # Collect from training directory
    if os.path.exists(DATASET_IMAGES_PATH):
        for emotion in EMOTION_CLASSES:
            emotion_path = os.path.join(DATASET_IMAGES_PATH, emotion)
            if os.path.exists(emotion_path):
                try:
                    for img_file in os.listdir(emotion_path):
                        if img_file.lower().endswith(SUPPORTED_FORMATS):
                            full_path = os.path.join(emotion_path, img_file)
                            if os.path.isfile(full_path):
                                all_images[emotion].append(full_path)
                except Exception as e:
                    logger.error(f"Error collecting from {emotion_path}: {e}")

    # Collect from validation directory
    if os.path.exists(DATASET_VAL_PATH):
        for emotion in EMOTION_CLASSES:
            emotion_path = os.path.join(DATASET_VAL_PATH, emotion)
            if os.path.exists(emotion_path):
                try:
                    for img_file in os.listdir(emotion_path):
                        if img_file.lower().endswith(SUPPORTED_FORMATS):
                            full_path = os.path.join(emotion_path, img_file)
                            if os.path.isfile(full_path):
                                all_images[emotion].append(full_path)
                except Exception as e:
                    logger.error(f"Error collecting from {emotion_path}: {e}")

    return all_images

def safe_move_file(src_path, dst_dir, new_filename):
    """Safely move file with proper error handling"""
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, new_filename)

        # Handle filename conflicts
        counter = 1
        base_name, ext = os.path.splitext(new_filename)
        while os.path.exists(dst_path):
            dst_path = os.path.join(dst_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        shutil.move(src_path, dst_path)
        return True
    except Exception as e:
        logging.error(f"Error moving {src_path} to {dst_dir}: {e}")
        return False

def balance_datasets_fixed():
    """Fixed dataset balancing that preserves data"""
    logger = logging.getLogger(__name__)
    random.seed(SEED)

    # Step 1: Collect ALL images from both directories
    logger.info("Collecting all images from both directories...")
    all_images = collect_all_images()

    # Check if we have any images
    total_collected = sum(len(images) for images in all_images.values())
    if total_collected == 0:
        logger.error("No images found in either directory!")
        return False

    logger.info(f"Collected {total_collected} total images")

    # Step 2: Create temporary directory for staging
    temp_dir = "temp_balanced"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Step 3: Process each emotion
        for emotion in EMOTION_CLASSES:
            emotion_images = all_images.get(emotion, [])

            if len(emotion_images) == 0:
                logger.warning(f"No images found for {emotion}")
                continue

            # Calculate 80/20 split
            total_emotion_images = len(emotion_images)
            train_count = int(total_emotion_images * TRAIN_RATIO)
            val_count = total_emotion_images - train_count

            # Shuffle for random distribution
            random.shuffle(emotion_images)

            # Create temp directories
            temp_train_dir = os.path.join(temp_dir, "images", emotion)
            temp_val_dir = os.path.join(temp_dir, "val", emotion)
            os.makedirs(temp_train_dir, exist_ok=True)
            os.makedirs(temp_val_dir, exist_ok=True)

            # Move training images to temp
            for i, img_path in enumerate(emotion_images[:train_count]):
                filename = f"train_{emotion}_{i:04d}{Path(img_path).suffix}"
                safe_move_file(img_path, temp_train_dir, filename)

            # Move validation images to temp
            for i, img_path in enumerate(emotion_images[train_count:]):
                filename = f"val_{emotion}_{i:04d}{Path(img_path).suffix}"
                safe_move_file(img_path, temp_val_dir, filename)

            logger.info(f"{emotion}: {train_count} train, {val_count} val")

        # Step 4: Clear original directories and move from temp
        logger.info("Updating original directories...")

        # Remove old directories
        if os.path.exists(DATASET_IMAGES_PATH):
            shutil.rmtree(DATASET_IMAGES_PATH)
        if os.path.exists(DATASET_VAL_PATH):
            shutil.rmtree(DATASET_VAL_PATH)

        # Move from temp to final locations
        if os.path.exists(os.path.join(temp_dir, "images")):
            shutil.move(os.path.join(temp_dir, "images"), DATASET_IMAGES_PATH)
        if os.path.exists(os.path.join(temp_dir, "val")):
            shutil.move(os.path.join(temp_dir, "val"), DATASET_VAL_PATH)

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        logger.info("Dataset balancing completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during balancing: {e}")
        # Clean up temp directory on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def print_report(before_train, before_val, after_train, after_val):
    """Print detailed balancing report"""
    print("\n" + "="*60)
    print("FIXED DATASET BALANCING REPORT")
    print("="*60)

    print(f"\nBEFORE BALANCING:")
    print(f"{'Emotion':<12} {'Train':<8} {'Val':<8} {'Total':<8}")
    print("-" * 40)

    total_before_train = 0
    total_before_val = 0

    for emotion in EMOTION_CLASSES:
        train_count = before_train.get(emotion, 0)
        val_count = before_val.get(emotion, 0)
        total_count = train_count + val_count

        print(f"{emotion:<12} {train_count:<8} {val_count:<8} {total_count:<8}")
        total_before_train += train_count
        total_before_val += val_count

    print("-" * 40)
    print(f"{'TOTAL':<12} {total_before_train:<8} {total_before_val:<8} {total_before_train + total_before_val:<8}")

    print(f"\nAFTER BALANCING:")
    print(f"{'Emotion':<12} {'Train':<8} {'Val':<8} {'Total':<8} {'Train%':<8}")
    print("-" * 50)

    total_after_train = 0
    total_after_val = 0

    for emotion in EMOTION_CLASSES:
        train_count = after_train.get(emotion, 0)
        val_count = after_val.get(emotion, 0)
        total_count = train_count + val_count
        train_percent = (train_count / total_count * 100) if total_count > 0 else 0

        print(f"{emotion:<12} {train_count:<8} {val_count:<8} {total_count:<8} {train_percent:<7.1f}%")
        total_after_train += train_count
        total_after_val += val_count

    print("-" * 50)
    total_after_all = total_after_train + total_after_val
    overall_train_percent = (total_after_train / total_after_all * 100) if total_after_all > 0 else 0
    print(f"{'TOTAL':<12} {total_after_train:<8} {total_after_val:<8} {total_after_all:<8} {overall_train_percent:<7.1f}%")

    print("\n" + "="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fixed Dataset Balancer for VisageCNN')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    global TRAIN_RATIO, SEED
    TRAIN_RATIO = args.train_ratio
    SEED = args.seed

    logger = setup_logging()

    logger.info("Starting FIXED dataset balancing for VisageCNN")
    logger.info(f"Target train ratio: {TRAIN_RATIO:.1%}")

    # Count images before balancing
    images_total_before, images_per_emotion_before = count_images(DATASET_IMAGES_PATH)
    val_total_before, val_per_emotion_before = count_images(DATASET_VAL_PATH)

    # Perform balancing
    success = balance_datasets_fixed()

    if not success:
        logger.error("Dataset balancing failed!")
        return

    # Count images after balancing
    images_total_after, images_per_emotion_after = count_images(DATASET_IMAGES_PATH)
    val_total_after, val_per_emotion_after = count_images(DATASET_VAL_PATH)

    # Print report
    print_report(
        images_per_emotion_before, val_per_emotion_before,
        images_per_emotion_after, val_per_emotion_after
    )

    logger.info("Fixed dataset balancing completed successfully!")

if __name__ == "__main__":
    main()
