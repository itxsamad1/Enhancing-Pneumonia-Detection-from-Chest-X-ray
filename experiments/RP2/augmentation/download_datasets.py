"""
Download and prepare pneumonia detection datasets from Kaggle.

Datasets:
1. Chest X-Ray Images (Pneumonia) - paultimothymooney/chest-xray-pneumonia
2. NIH ChestX-ray14 - nih-chest-xrays/data
3. CheXpert - ashery/chexpert

Filters pneumonia-only data from multi-label datasets and merges everything
into a unified dataset/ directory with 80/20 train/val split.
"""

import os
import sys
import shutil
import random
import csv
from pathlib import Path
from tqdm import tqdm

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Dataset directory
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = BASE_DIR / "raw_datasets"

RANDOM_SEED = 42


def download_kaggle_dataset(dataset_slug, dest_dir):
    """Download a dataset from Kaggle using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_slug}")
        print(f"Destination: {dest_dir}")
        print(f"{'='*60}")

        os.makedirs(dest_dir, exist_ok=True)
        api.dataset_download_files(dataset_slug, path=dest_dir, unzip=True)
        print(f"[OK] Download complete: {dataset_slug}")
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading {dataset_slug}: {e}")
        return False


def process_kaggle_pneumonia(raw_dir):
    """
    Process the Kaggle Chest X-Ray Pneumonia dataset.
    Already organized as train/test/val with NORMAL and PNEUMONIA folders.
    Returns lists of (source_path, label) tuples.
    """
    print("\n[DATASET] Processing Kaggle Chest X-Ray Pneumonia dataset...")
    images = []

    kaggle_dir = Path(raw_dir) / "kaggle_pneumonia"

    # Find the actual data directory (may be nested)
    data_root = None
    for root_candidate in [
        kaggle_dir / "chest_xray",
        kaggle_dir / "chest_xray" / "chest_xray",
        kaggle_dir,
    ]:
        if (root_candidate / "train").exists():
            data_root = root_candidate
            break

    if data_root is None:
        for dirpath, dirnames, filenames in os.walk(kaggle_dir):
            if "train" in dirnames:
                candidate = Path(dirpath) / "train"
                if (candidate / "NORMAL").exists() or (candidate / "PNEUMONIA").exists():
                    data_root = Path(dirpath)
                    break

    if data_root is None:
        print("  [WARN] Could not find Kaggle pneumonia dataset structure")
        return images

    # Collect from train, test, and val splits
    for split in ["train", "test", "val"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        for label_name in ["NORMAL", "PNEUMONIA"]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            label = 0 if label_name == "NORMAL" else 1
            for img_file in label_dir.iterdir():
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    images.append((str(img_file), label))

    normal_count = sum(1 for _, l in images if l == 0)
    pneumonia_count = sum(1 for _, l in images if l == 1)
    print(f"  [OK] Found {len(images)} images (Normal: {normal_count}, Pneumonia: {pneumonia_count})")
    return images


def process_nih_chestxray(raw_dir):
    """
    Process NIH ChestX-ray14 dataset.
    Reads Data_Entry CSV to find Pneumonia and No Finding images.
    """
    print("\n[DATASET] Processing NIH ChestX-ray14 dataset...")
    images = []

    nih_dir = Path(raw_dir) / "nih_chestxray"

    if not nih_dir.exists():
        print("  [WARN] NIH directory not found, skipping")
        return images

    # Find the CSV file
    csv_path = None
    for dirpath, dirnames, filenames in os.walk(nih_dir):
        for f in filenames:
            if f.endswith(".csv") and ("data_entry" in f.lower() or "entry" in f.lower()):
                csv_path = Path(dirpath) / f
                break
        if csv_path:
            break

    if csv_path is None:
        print("  [WARN] Could not find Data_Entry CSV in NIH dataset")
        return images

    print(f"  Reading: {csv_path}")

    # Build a mapping of image filename -> label
    image_labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("Image Index", "")
            findings = row.get("Finding Labels", "")

            if "Pneumonia" in findings:
                image_labels[img_name] = 1  # Pneumonia
            elif findings.strip() == "No Finding":
                image_labels[img_name] = 0  # Normal

    print(f"  Filtered CSV: {sum(1 for v in image_labels.values() if v == 1)} Pneumonia, "
          f"{sum(1 for v in image_labels.values() if v == 0)} Normal")

    # Find actual image files
    for dirpath, dirnames, filenames in os.walk(nih_dir):
        for f in filenames:
            if f in image_labels and f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append((os.path.join(dirpath, f), image_labels[f]))

    normal_count = sum(1 for _, l in images if l == 0)
    pneumonia_count = sum(1 for _, l in images if l == 1)
    print(f"  [OK] Found {len(images)} matching images (Normal: {normal_count}, Pneumonia: {pneumonia_count})")
    return images


def process_chexpert(raw_dir):
    """
    Process CheXpert dataset.
    Reads train.csv/valid.csv, filters Pneumonia positives and fully-normal cases.
    """
    print("\n[DATASET] Processing CheXpert dataset...")
    images = []

    chexpert_dir = Path(raw_dir) / "chexpert"

    if not chexpert_dir.exists():
        print("  [WARN] CheXpert directory not found, skipping")
        return images

    # Find CSV files
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(chexpert_dir):
        for f in filenames:
            if f.lower() in ("train.csv", "valid.csv"):
                csv_files.append(Path(dirpath) / f)

    if not csv_files:
        print("  [WARN] Could not find train.csv/valid.csv in CheXpert dataset")
        return images

    for csv_path in csv_files:
        print(f"  Reading: {csv_path}")
        csv_parent = csv_path.parent

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pneumonia_val = row.get("Pneumonia", "").strip()
                path_col = row.get("Path", "")

                if not path_col:
                    continue

                # Construct full path
                img_path = csv_parent / path_col
                if not img_path.exists():
                    img_path = chexpert_dir / path_col
                    if not img_path.exists():
                        continue

                if pneumonia_val in ("1.0", "1"):
                    images.append((str(img_path), 1))  # Pneumonia
                elif row.get("No Finding", "").strip() in ("1.0", "1"):
                    images.append((str(img_path), 0))  # Normal

    normal_count = sum(1 for _, l in images if l == 0)
    pneumonia_count = sum(1 for _, l in images if l == 1)
    print(f"  [OK] Found {len(images)} images (Normal: {normal_count}, Pneumonia: {pneumonia_count})")
    return images


def create_unified_dataset(all_images, train_ratio=0.8):
    """Merge all images into a unified dataset with 80/20 train/val split."""
    print(f"\n{'='*60}")
    print("Creating unified dataset with 80/20 split...")
    print(f"{'='*60}")

    # Separate by class
    normal_images = [path for path, label in all_images if label == 0]
    pneumonia_images = [path for path, label in all_images if label == 1]

    print(f"Total Normal: {len(normal_images)}")
    print(f"Total Pneumonia: {len(pneumonia_images)}")
    print(f"Total: {len(all_images)}")

    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)

    # Split 80/20
    normal_train_count = int(len(normal_images) * train_ratio)
    pneumonia_train_count = int(len(pneumonia_images) * train_ratio)

    splits = {
        "train/NORMAL": normal_images[:normal_train_count],
        "val/NORMAL": normal_images[normal_train_count:],
        "train/PNEUMONIA": pneumonia_images[:pneumonia_train_count],
        "val/PNEUMONIA": pneumonia_images[pneumonia_train_count:],
    }

    # Create directories
    for split_name in splits:
        os.makedirs(DATASET_DIR / split_name, exist_ok=True)

    # Copy files
    for split_name, file_list in splits.items():
        dest_dir = DATASET_DIR / split_name
        print(f"\n  Copying {len(file_list)} files to {split_name}...")

        for i, src_path in enumerate(tqdm(file_list, desc=f"  {split_name}", unit="img")):
            ext = Path(src_path).suffix
            dest_path = dest_dir / f"{split_name.split('/')[1].lower()}_{i:06d}{ext}"
            try:
                shutil.copy2(src_path, dest_path)
            except Exception:
                pass  # Skip corrupted/inaccessible files

    # Print summary
    print(f"\n{'='*60}")
    print("Dataset Summary:")
    print(f"{'='*60}")
    for split_name in ["train/NORMAL", "train/PNEUMONIA", "val/NORMAL", "val/PNEUMONIA"]:
        count = len(list((DATASET_DIR / split_name).glob("*")))
        print(f"  {split_name}: {count} images")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("Pneumonia Detection -- Dataset Download & Preparation")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)

    all_images = []

    # -- Dataset 1: Kaggle Chest X-Ray Pneumonia --
    kaggle_dir = RAW_DIR / "kaggle_pneumonia"
    if not kaggle_dir.exists() or not any(kaggle_dir.rglob("*.jpeg")):
        success = download_kaggle_dataset(
            "paultimothymooney/chest-xray-pneumonia",
            str(kaggle_dir)
        )
        if not success:
            print("[WARN] Skipping Kaggle Pneumonia dataset")

    images = process_kaggle_pneumonia(RAW_DIR)
    all_images.extend(images)

    # -- Dataset 2: NIH ChestX-ray14 --
    nih_dir = RAW_DIR / "nih_chestxray"
    if not nih_dir.exists() or not any(nih_dir.rglob("*.png")):
        success = download_kaggle_dataset(
            "nih-chest-xrays/data",
            str(nih_dir)
        )
        if not success:
            print("[WARN] Skipping NIH ChestX-ray14 dataset")

    images = process_nih_chestxray(RAW_DIR)
    all_images.extend(images)

    # -- Dataset 3: CheXpert --
    chexpert_dir = RAW_DIR / "chexpert"
    if not chexpert_dir.exists() or not any(chexpert_dir.rglob("*.jpg")):
        success = download_kaggle_dataset(
            "ashery/chexpert",
            str(chexpert_dir)
        )
        if not success:
            print("[WARN] Skipping CheXpert dataset")

    images = process_chexpert(RAW_DIR)
    all_images.extend(images)

    # -- Merge & Split --
    if len(all_images) == 0:
        print("\n[ERROR] No images found! Please check dataset downloads.")
        sys.exit(1)

    create_unified_dataset(all_images, train_ratio=0.8)

    print("\n[DONE] Dataset preparation complete!")
    print(f"   Dataset directory: {DATASET_DIR}")


if __name__ == "__main__":
    main()
