"""
RSNA Pneumonia Detection Dataset — Download & Merge Script
============================================================
Downloads the RSNA Pneumonia Detection dataset (pre-processed PNGs)
and the processed labels from Kaggle, then merges labeled Normal/Pneumonia
images into the existing dataset/ folder.

Usage:
  python download_rsna_dataset.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import shutil
import random
import zipfile
from pathlib import Path
from PIL import Image
import csv

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = BASE_DIR / "raw_datasets" / "rsna"
MANIFEST_PATH = DATASET_DIR / "MANIFEST.json"

VAL_SPLIT_RATIO = 0.15
IMAGE_SIZE = 224
RANDOM_SEED = 42

# Kaggle datasets (pre-processed — no competition rules needed)
LABELS_DATASET = "iamtapendu/rsna-pneumonia-processed-dataset"
IMAGES_DATASET = "vaillant/rsna-pneu-train-png"


def download_datasets():
    """Download both the labels and PNG images from Kaggle."""
    print("=" * 60)
    print("RSNA Pneumonia Detection — Download & Merge into dataset/")
    print("=" * 60)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("[OK] Kaggle API authenticated")

    # Download labels dataset
    labels_dir = RAW_DIR / "labels"
    images_dir = RAW_DIR / "images"

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Check if labels already exist
    if not any(labels_dir.glob("*.csv")):
        print(f"\n[1/5] Downloading RSNA labels...")
        api.dataset_download_files(LABELS_DATASET, path=str(labels_dir), unzip=True, quiet=False)
        print("      Labels downloaded.")
    else:
        print(f"\n[1/5] Labels already downloaded at {labels_dir}")

    # Check if images already exist
    png_count = len(list(images_dir.rglob("*.png")))
    if png_count < 1000:
        print(f"\n[2/5] Downloading RSNA PNG images (~4 GB, this will take a while)...")
        api.dataset_download_files(IMAGES_DATASET, path=str(images_dir), unzip=True, quiet=False)
        print("      Images downloaded.")
    else:
        print(f"\n[2/5] Images already downloaded ({png_count} PNGs found)")

    return labels_dir, images_dir


def find_labels_csv(labels_dir):
    """Find the stage_2_train_labels.csv or equivalent."""
    # Search for any CSV with 'label' in the name or the standard file
    candidates = list(labels_dir.rglob("*label*.csv")) + list(labels_dir.rglob("*train*.csv"))
    if not candidates:
        candidates = list(labels_dir.rglob("*.csv"))

    if not candidates:
        print("[ERROR] No CSV file found in labels directory!")
        print(f"        Contents of {labels_dir}:")
        for f in labels_dir.rglob("*"):
            print(f"          {f}")
        sys.exit(1)

    # Prefer the one with 'label' in name
    for c in candidates:
        if 'label' in c.name.lower():
            return c

    return candidates[0]


def parse_labels(labels_csv):
    """Parse labels CSV to get patient-level classification."""
    print(f"\n[3/5] Parsing labels from {labels_csv.name}...")

    patient_labels = {}

    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"      CSV columns: {headers}")

        # Detect column names (varies by dataset version)
        pid_col = None
        target_col = None
        for h in headers:
            hl = h.lower().strip()
            if hl in ('patientid', 'patient_id', 'id', 'filename') and pid_col is None:
                pid_col = h
            if hl in ('target', 'label', 'pneumonia') and target_col is None:
                target_col = h

        # If still not found, check 'class' as fallback
        if target_col is None and 'class' in [h.lower().strip() for h in headers]:
            target_col = 'class'

        if pid_col is None:
            pid_col = headers[0]
        if target_col is None:
            target_col = headers[-1]

        print(f"      Using: ID='{pid_col}', Target='{target_col}'")

        for row in reader:
            pid = row[pid_col].strip()
            # Remove .dcm or .png extension if present
            pid = pid.replace('.dcm', '').replace('.png', '')
            try:
                target = int(float(row[target_col].strip()))
            except (ValueError, KeyError):
                continue

            if pid not in patient_labels:
                patient_labels[pid] = target
            else:
                patient_labels[pid] = max(patient_labels[pid], target)

    normal_count = sum(1 for v in patient_labels.values() if v == 0)
    pneumonia_count = sum(1 for v in patient_labels.values() if v == 1)

    print(f"      Total patients: {len(patient_labels)}")
    print(f"      Normal: {normal_count}")
    print(f"      Pneumonia: {pneumonia_count}")

    return patient_labels


def find_image(images_dir, patient_id):
    """Find a PNG image for the given patient ID."""
    # Try common patterns
    for pattern in [f"{patient_id}.png", f"{patient_id}.jpg", f"{patient_id}.jpeg"]:
        matches = list(images_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def merge_into_dataset(patient_labels, images_dir):
    """Copy and resize RSNA images into dataset/ folder."""
    print(f"\n[4/5] Merging RSNA images into dataset/...")

    # Separate by label
    normal_ids = [pid for pid, lbl in patient_labels.items() if lbl == 0]
    pneumonia_ids = [pid for pid, lbl in patient_labels.items() if lbl == 1]

    random.seed(RANDOM_SEED)
    random.shuffle(normal_ids)
    random.shuffle(pneumonia_ids)

    # Split
    n_val_normal = int(len(normal_ids) * VAL_SPLIT_RATIO)
    n_val_pneumonia = int(len(pneumonia_ids) * VAL_SPLIT_RATIO)

    splits = {
        "train": {
            "NORMAL": normal_ids[n_val_normal:],
            "PNEUMONIA": pneumonia_ids[n_val_pneumonia:]
        },
        "val": {
            "NORMAL": normal_ids[:n_val_normal],
            "PNEUMONIA": pneumonia_ids[:n_val_pneumonia]
        }
    }

    for split_name, classes in splits.items():
        for class_name, ids in classes.items():
            print(f"      Planned: {split_name}/{class_name} = {len(ids)} RSNA images")

    import datetime
    manifest = {
        "source": "RSNA Pneumonia Detection Challenge (pre-processed PNGs from Kaggle)",
        "merge_date": datetime.datetime.now().isoformat(),
        "datasets_used": [LABELS_DATASET, IMAGES_DATASET],
        "images_added": {},
        "resize": f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    }

    total_added = 0
    total_missing = 0

    for split_name, classes in splits.items():
        for class_name, patient_ids in classes.items():
            target_dir = DATASET_DIR / split_name / class_name
            os.makedirs(target_dir, exist_ok=True)

            added = 0
            missing = 0

            for pid in patient_ids:
                out_path = target_dir / f"rsna_{pid}.png"
                if out_path.exists():
                    added += 1
                    continue

                src_path = find_image(images_dir, pid)
                if src_path is None:
                    missing += 1
                    continue

                try:
                    img = Image.open(str(src_path)).convert("RGB")
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                    img.save(str(out_path), "PNG")
                    added += 1
                except Exception:
                    missing += 1

            manifest["images_added"][f"{split_name}/{class_name}"] = added
            total_added += added
            total_missing += missing
            print(f"      [OK] {split_name}/{class_name}: {added} added, {missing} not found")

    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n      Total RSNA images merged: {total_added}")
    print(f"      Missing/skipped: {total_missing}")
    return total_added


def print_final_stats():
    """Print merged dataset statistics."""
    print(f"\n[5/5] Final Merged Dataset Statistics:")
    print("-" * 55)

    total = 0
    for split in ["train", "val"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            dir_path = DATASET_DIR / split / cls
            if dir_path.exists():
                all_imgs = [f for f in dir_path.iterdir()
                           if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                count = len(all_imgs)
                rsna = sum(1 for f in all_imgs if f.name.startswith("rsna_"))
                kaggle = count - rsna
                print(f"  {split:>5}/{cls:<10}: {count:>6} total  "
                      f"(Kaggle: {kaggle:>5} | RSNA: {rsna:>5})")
                total += count

    print(f"\n  GRAND TOTAL: {total:,} images")
    print("=" * 60)


def main():
    labels_dir, images_dir = download_datasets()
    labels_csv = find_labels_csv(labels_dir)
    print(f"      Using labels file: {labels_csv}")
    patient_labels = parse_labels(labels_csv)
    merge_into_dataset(patient_labels, images_dir)
    print_final_stats()
    print("\n[DONE] Dataset expansion complete!")


if __name__ == "__main__":
    main()
