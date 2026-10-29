"""
Prepare the dataset from the already-downloaded Kaggle Chest X-Ray Pneumonia dataset.
Creates a unified dataset/ directory with 80/20 train/val split.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = BASE_DIR / "raw_datasets" / "kaggle_pneumonia"
RANDOM_SEED = 42


def find_data_root(kaggle_dir):
    """Find the actual chest_xray root directory."""
    for candidate in [
        kaggle_dir / "chest_xray",
        kaggle_dir / "chest_xray" / "chest_xray",
        kaggle_dir,
    ]:
        if (candidate / "train").exists():
            return candidate

    for dirpath, dirnames, _ in os.walk(kaggle_dir):
        if "train" in dirnames:
            test_dir = Path(dirpath) / "train"
            if (test_dir / "NORMAL").exists() or (test_dir / "PNEUMONIA").exists():
                return Path(dirpath)
    return None


def collect_images(data_root):
    """Collect all images from all splits (train/test/val)."""
    images = []
    for split in ["train", "test", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for label_name, label_val in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for img_file in label_dir.iterdir():
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    images.append((str(img_file), label_val))
    return images


def main():
    print("=" * 60)
    print("Pneumonia Detection -- Dataset Preparation")
    print("=" * 60)

    # Check if dataset already exists
    if (DATASET_DIR / "train" / "NORMAL").exists() and (DATASET_DIR / "train" / "PNEUMONIA").exists():
        train_normal = len(list((DATASET_DIR / "train" / "NORMAL").glob("*")))
        train_pneumonia = len(list((DATASET_DIR / "train" / "PNEUMONIA").glob("*")))
        val_normal = len(list((DATASET_DIR / "val" / "NORMAL").glob("*")))
        val_pneumonia = len(list((DATASET_DIR / "val" / "PNEUMONIA").glob("*")))
        total = train_normal + train_pneumonia + val_normal + val_pneumonia
        if total > 100:
            print(f"Dataset already exists with {total} images!")
            print(f"  train/NORMAL: {train_normal}")
            print(f"  train/PNEUMONIA: {train_pneumonia}")
            print(f"  val/NORMAL: {val_normal}")
            print(f"  val/PNEUMONIA: {val_pneumonia}")
            print("Skipping preparation. Delete dataset/ folder to re-prepare.")
            return

    # Find data root
    data_root = find_data_root(RAW_DIR)
    if data_root is None:
        print("[ERROR] Could not find chest_xray dataset structure in raw_datasets/kaggle_pneumonia/")
        print("Please run: download_datasets.py first")
        sys.exit(1)

    print(f"Found dataset at: {data_root}")

    # Collect all images
    print("\nCollecting images from all splits...")
    all_images = collect_images(data_root)

    normal_images = [p for p, l in all_images if l == 0]
    pneumonia_images = [p for p, l in all_images if l == 1]

    print(f"  Normal images: {len(normal_images)}")
    print(f"  Pneumonia images: {len(pneumonia_images)}")
    print(f"  Total: {len(all_images)}")

    # Shuffle and split 80/20
    random.seed(RANDOM_SEED)
    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)

    normal_train_count = int(len(normal_images) * 0.8)
    pneumonia_train_count = int(len(pneumonia_images) * 0.8)

    splits = {
        "train/NORMAL": normal_images[:normal_train_count],
        "val/NORMAL": normal_images[normal_train_count:],
        "train/PNEUMONIA": pneumonia_images[:pneumonia_train_count],
        "val/PNEUMONIA": pneumonia_images[pneumonia_train_count:],
    }

    # Create directories and copy files
    for split_name, file_list in splits.items():
        dest_dir = DATASET_DIR / split_name
        os.makedirs(dest_dir, exist_ok=True)

        print(f"\n  Copying {len(file_list)} files to {split_name}...")
        for i, src_path in enumerate(tqdm(file_list, desc=f"  {split_name}", unit="img")):
            ext = Path(src_path).suffix
            dest_path = dest_dir / f"{split_name.split('/')[1].lower()}_{i:06d}{ext}"
            try:
                shutil.copy2(src_path, dest_path)
            except Exception:
                pass

    # Summary
    print(f"\n{'='*60}")
    print("Dataset Summary:")
    print(f"{'='*60}")
    for split_name in ["train/NORMAL", "train/PNEUMONIA", "val/NORMAL", "val/PNEUMONIA"]:
        count = len(list((DATASET_DIR / split_name).glob("*")))
        print(f"  {split_name}: {count} images")
    total = sum(len(list((DATASET_DIR / s).glob("*"))) for s in
                ["train/NORMAL", "train/PNEUMONIA", "val/NORMAL", "val/PNEUMONIA"])
    print(f"  TOTAL: {total} images")
    print(f"{'='*60}")
    print("\n[DONE] Dataset preparation complete!")


if __name__ == "__main__":
    main()
