"""
Pneumonia Detection — Offline Dataset Augmentation
=====================================================
Expands the dataset by generating augmented versions of each image.
Applies medically appropriate augmentations for chest X-ray images.

Augmentation Techniques:
  1. Horizontal Flip
  2. Random Rotation (±15°)
  3. Brightness Adjustment (±20%)
  4. Contrast Adjustment (CLAHE-style)
  5. Gaussian Noise injection
  6. Combined augmentations (multi-technique per image)

Class Balancing:
  - The minority class (Normal) is augmented MORE to balance the dataset.
  - Pneumonia class gets standard augmentation.

Output:
  Creates dataset_augmented/ with the expanded, balanced dataset.
"""

import os
import sys
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "dataset"               # Original dataset
OUTPUT_DIR = BASE_DIR / "dataset_augmented"      # Augmented output
RANDOM_SEED = 42

# Augmentation factor per class
# Normal gets more augmentation to balance with Pneumonia
AUGMENT_FACTOR_NORMAL = 8       # Each Normal image → 8 augmented versions
AUGMENT_FACTOR_PNEUMONIA = 3    # Each Pneumonia image → 3 augmented versions

# Image size for saving
IMG_SIZE = 256  # Save at 256x256 (training script will crop to 224)


# ─── Augmentation Functions ─────────────────────────────────────
def augment_horizontal_flip(img):
    """Flip the image horizontally."""
    return cv2.flip(img, 1)


def augment_rotation(img, angle_range=15):
    """Rotate by a random angle within ±angle_range degrees."""
    h, w = img.shape[:2]
    angle = random.uniform(-angle_range, angle_range)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h),
                             borderMode=cv2.BORDER_REFLECT_101)
    return rotated


def augment_brightness(img, factor_range=(0.8, 1.2)):
    """Adjust brightness by a random factor."""
    factor = random.uniform(*factor_range)
    adjusted = np.clip(img.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)


def augment_contrast_clahe(img, clip_limit_range=(1.5, 3.0)):
    """Apply CLAHE with random clip limit for contrast enhancement."""
    clip_limit = random.uniform(*clip_limit_range)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if len(img.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def augment_gaussian_noise(img, mean=0, std_range=(5, 15)):
    """Add Gaussian noise to the image."""
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def augment_sharpening(img):
    """Apply sharpening filter."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def augment_zoom(img, zoom_range=(1.05, 1.20)):
    """Random zoom (center crop after scaling)."""
    h, w = img.shape[:2]
    zoom = random.uniform(*zoom_range)
    new_h, new_w = int(h * zoom), int(w * zoom)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Center crop back to original size
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    cropped = resized[start_y:start_y + h, start_x:start_x + w]
    return cropped


def augment_combined(img):
    """Apply 2-3 random augmentations together for maximum diversity."""
    augmentations = [
        augment_horizontal_flip,
        lambda x: augment_rotation(x, 10),
        lambda x: augment_brightness(x, (0.85, 1.15)),
        lambda x: augment_gaussian_noise(x, std_range=(3, 10)),
        augment_zoom,
    ]
    # Pick 2-3 random augmentations
    num_augs = random.choice([2, 3])
    chosen = random.sample(augmentations, num_augs)
    result = img.copy()
    for aug_fn in chosen:
        result = aug_fn(result)
    return result


# All individual augmentation functions
AUGMENTATION_POOL = [
    ("flip", augment_horizontal_flip),
    ("rotate", lambda img: augment_rotation(img, 15)),
    ("bright", lambda img: augment_brightness(img, (0.8, 1.2))),
    ("clahe", augment_contrast_clahe),
    ("noise", augment_gaussian_noise),
    ("sharpen", augment_sharpening),
    ("zoom", augment_zoom),
    ("combined", augment_combined),
]


# ─── Core Pipeline ──────────────────────────────────────────────
def load_image(path):
    """Load an image using OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        # Fallback: use PIL for tricky formats
        pil_img = Image.open(str(path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def resize_image(img, size=IMG_SIZE):
    """Resize image to target size."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def process_class(split, class_name, aug_factor):
    """
    Process one class directory:
      1. Copy all originals to the output directory
      2. Generate `aug_factor` augmented versions per image
    """
    src_dir = SOURCE_DIR / split / class_name
    dst_dir = OUTPUT_DIR / split / class_name
    os.makedirs(dst_dir, exist_ok=True)

    if not src_dir.exists():
        print(f"  ⚠️  {src_dir} not found, skipping.")
        return 0, 0

    # Collect source images
    image_files = sorted([
        f for f in src_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])

    if not image_files:
        print(f"  ⚠️  No images found in {src_dir}")
        return 0, 0

    original_count = len(image_files)
    augmented_count = 0

    desc = f"  {split}/{class_name}"
    for img_file in tqdm(image_files, desc=desc, unit="img"):
        img = load_image(img_file)
        if img is None:
            continue

        img = resize_image(img)

        # 1. Save the original
        orig_name = f"orig_{img_file.stem}.png"
        cv2.imwrite(str(dst_dir / orig_name), img)

        # 2. Generate augmented versions
        for aug_idx in range(aug_factor):
            # Cycle through augmentation types, then use combined for extras
            aug_name, aug_fn = AUGMENTATION_POOL[aug_idx % len(AUGMENTATION_POOL)]
            try:
                aug_img = aug_fn(img.copy())
                aug_img = resize_image(aug_img)  # Ensure consistent size
                save_name = f"aug_{aug_name}_{aug_idx}_{img_file.stem}.png"
                cv2.imwrite(str(dst_dir / save_name), aug_img)
                augmented_count += 1
            except Exception as e:
                # Skip any failed augmentation silently
                pass

    return original_count, augmented_count


def main():
    print("=" * 60)
    print("Pneumonia Detection — Dataset Augmentation")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Validate source dataset
    if not SOURCE_DIR.exists():
        print(f"❌ Source dataset not found at {SOURCE_DIR}")
        print("   Run prepare_dataset.py first.")
        sys.exit(1)

    # Check if augmented dataset already exists
    if OUTPUT_DIR.exists():
        print(f"⚠️  Output directory {OUTPUT_DIR} already exists.")
        print("   Deleting and re-creating...")
        shutil.rmtree(OUTPUT_DIR)

    print(f"\n📂 Source dataset: {SOURCE_DIR}")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"\n🔧 Augmentation Configuration:")
    print(f"   Normal class:    {AUGMENT_FACTOR_NORMAL}x augmentation (class balancing)")
    print(f"   Pneumonia class: {AUGMENT_FACTOR_PNEUMONIA}x augmentation")
    print(f"   Techniques: {len(AUGMENTATION_POOL)} different augmentation methods")

    # Process each split and class
    total_original = 0
    total_augmented = 0

    for split in ["train", "val"]:
        print(f"\n{'─' * 40}")
        print(f"📁 Processing {split} split...")
        print(f"{'─' * 40}")

        # Normal class — gets MORE augmentation to balance
        orig, aug = process_class(split, "NORMAL", AUGMENT_FACTOR_NORMAL)
        total_original += orig
        total_augmented += aug

        # Pneumonia class — standard augmentation
        orig, aug = process_class(split, "PNEUMONIA", AUGMENT_FACTOR_PNEUMONIA)
        total_original += orig
        total_augmented += aug

    # Final summary
    print(f"\n{'=' * 60}")
    print("Dataset Augmentation — Summary")
    print(f"{'=' * 60}")

    for split in ["train", "val"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = OUTPUT_DIR / split / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob("*")))
                print(f"  {split}/{cls}: {count} images")

    # Calculate totals
    train_normal = len(list((OUTPUT_DIR / "train" / "NORMAL").glob("*")))
    train_pneumonia = len(list((OUTPUT_DIR / "train" / "PNEUMONIA").glob("*")))
    val_normal = len(list((OUTPUT_DIR / "val" / "NORMAL").glob("*")))
    val_pneumonia = len(list((OUTPUT_DIR / "val" / "PNEUMONIA").glob("*")))
    grand_total = train_normal + train_pneumonia + val_normal + val_pneumonia

    print(f"\n  📊 Original dataset:  {total_original} images")
    print(f"  📊 Augmented dataset: {grand_total} images")
    print(f"  📊 Expansion factor:  {grand_total / total_original:.1f}x")

    print(f"\n  🏋️ Training set:")
    print(f"     Normal:    {train_normal} images")
    print(f"     Pneumonia: {train_pneumonia} images")
    ratio = max(train_normal, train_pneumonia) / min(train_normal, train_pneumonia)
    print(f"     Ratio:     1:{ratio:.1f}")

    print(f"\n  ✅ Validation set:")
    print(f"     Normal:    {val_normal} images")
    print(f"     Pneumonia: {val_pneumonia} images")

    print(f"\n{'=' * 60}")
    print("✅ Augmentation complete!")
    print(f"   Use dataset_augmented/ as DATASET_DIR in train_pneumonia.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
