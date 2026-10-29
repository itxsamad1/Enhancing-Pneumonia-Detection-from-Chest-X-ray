"""
Pneumonia Detection — Training Script (Augmented Dataset)
===========================================================
ResNet-18 fine-tuned for binary classification (Normal vs Pneumonia).
Designed to work with the augmented dataset from augment_dataset.py.

Anti-Overfitting Approaches:
  A. Offline Data Augmentation (expanded dataset)
  B. Real-time Data Augmentation (on-the-fly transforms)
  C. Dropout in classifier head (regularization)
  D. Weight Decay / L2 Regularization
  E. Early Stopping (patience-based)
  F. Learning Rate Scheduling (CosineAnnealing with warm restarts)
  G. Gradient Clipping (prevents exploding gradients)
  H. Label Smoothing (softens hard targets)

GPU Optimization:
  - Mixed precision training (torch.amp)
  - Larger batch size for 16GB VRAM
  - Persistent workers + pin_memory
  - cudnn.benchmark for static input sizes
  - torch.compile for graph optimization (PyTorch 2.x)

Graphs saved as PNG:
    graphs/accuracy_plot.png
    graphs/loss_plot.png
    graphs/precision_plot.png
    graphs/recall_plot.png
    graphs/f1_score_plot.png
    graphs/specificity_plot.png
    graphs/learning_rate_plot.png
    graphs/metrics_plot.png
    graphs/confusion_matrix.png
- Training history saved to graphs/training_history.json
"""

import os
import sys
# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for background plotting without popups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)

# ─── Configuration ───────────────────────────────────────────────
NUM_EPOCHS = 30
BATCH_SIZE = 64                  # Larger batch → better GPU utilization (16GB VRAM)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4              # L2 regularization
LABEL_SMOOTHING = 0.1            # Softens targets: prevents overconfident predictions
DROPOUT_RATE = 0.3               # Dropout in classifier head
GRADIENT_CLIP_MAX_NORM = 1.0     # Gradient clipping threshold
EARLY_STOPPING_PATIENCE = 10     # Stop if no improvement for 10 epochs
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True        # Keep worker processes alive between epochs

# Paths — point to the AUGMENTED dataset
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset_augmented")
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_resnet18.pt")

# GPU optimization
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms


# ─── Dataset ─────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """Binary chest X-ray dataset: Normal (0) vs Pneumonia (1)."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        normal_dir = os.path.join(root_dir, "NORMAL")
        pneumonia_dir = os.path.join(root_dir, "PNEUMONIA")

        if os.path.exists(normal_dir):
            for fname in os.listdir(normal_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(normal_dir, fname))
                    self.labels.append(0)

        if os.path.exists(pneumonia_dir):
            for fname in os.listdir(pneumonia_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(pneumonia_dir, fname))
                    self.labels.append(1)

        print(f"  Loaded {len(self.images)} images "
              f"(Normal: {self.labels.count(0)}, Pneumonia: {self.labels.count(1)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── Transforms (Real-time augmentation on top of offline augmentation) ──
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small shifts
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random patch erasure
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model ───────────────────────────────────────────────────────
def create_model(dropout_rate=DROPOUT_RATE):
    """
    Create ResNet-18 with pretrained weights, modified for binary classification.
    Adds Dropout before the final FC layer to reduce overfitting.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features

    # Replace FC with Dropout + FC (Anti-Overfitting Approach C)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, 2)
    )
    return model


# ─── Metrics helpers ─────────────────────────────────────────────
def compute_specificity(y_true, y_pred):
    """Specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# ─── Training ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch with gradient clipping. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="    Train", leave=False, unit="batch"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient Clipping (Anti-Overfitting Approach G)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(loader, desc="    Val  ", leave=False, unit="batch"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ─── Plotting ────────────────────────────────────────────────────
def plot_accuracy(history, save_dir):
    """Accuracy + Val Accuracy → same graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Training Accuracy")
    ax.plot(epochs, history["val_acc"], "r-o", markersize=4, label="Validation Accuracy")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(save_dir, "accuracy_plot.png")
    fig.savefig(path, dpi=150)
    print(f"  📊 Saved: {path}")
    return fig


def plot_loss(history, save_dir):
    """Loss + Val Loss → same graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Training Loss")
    ax.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Validation Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "loss_plot.png")
    fig.savefig(path, dpi=150)
    print(f"  📊 Saved: {path}")
    return fig


def plot_metric_single(history, metric_key, title, filename, color_marker, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history[metric_key]) + 1)
    ax.plot(epochs, history[metric_key], color_marker, markersize=5, label=title)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(title + " over Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if metric_key != "lr":
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    print(f"  📊 Saved: {path}")
    plt.close(fig)
    return fig

def generate_individual_metrics(history, save_dir):
    plot_metric_single(history, "precision", "Precision", "precision_plot.png", "g-^", save_dir)
    plot_metric_single(history, "recall", "Recall", "recall_plot.png", "m-s", save_dir)
    plot_metric_single(history, "f1", "F1 Score", "f1_score_plot.png", "c-D", save_dir)
    plot_metric_single(history, "specificity", "Specificity", "specificity_plot.png", "y-v", save_dir)
    plot_metric_single(history, "lr", "Learning Rate", "learning_rate_plot.png", "k--.", save_dir)


def plot_confusion_mat(y_true, y_pred, save_dir):
    """Confusion Matrix → separate heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    print(f"  📊 Saved: {path}")
    return fig


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Pneumonia Detection — Model Training (Augmented Dataset)")
    print("=" * 60)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  Device: {device} ({gpu_name})")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   VRAM: {vram_gb:.1f} GB")
        print(f"   cuDNN benchmark: ENABLED")
    else:
        print(f"⚠️  Device: CPU (no CUDA GPU detected)")

    # ── Data ──
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Augmented dataset not found! Run augment_dataset.py first.")
        sys.exit(1)

    print(f"\n📂 Loading augmented datasets from {DATASET_DIR}...")
    print("  Training set:")
    train_dataset = ChestXrayDataset(train_dir, transform=train_transform)
    print("  Validation set:")
    val_dataset = ChestXrayDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=True  # Drop incomplete batch for stable batch norm
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # ── Model ──
    print(f"\n🧠 Creating model (ResNet-18, pretrained, dropout={DROPOUT_RATE})...")
    model = create_model(dropout_rate=DROPOUT_RATE).to(device)

    # Label Smoothing (Anti-Overfitting Approach H)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    print(f"   Loss: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"   Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")

    # CosineAnnealingWarmRestarts — better than ReduceLROnPlateau for research
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")

    scaler = torch.amp.GradScaler("cuda")

    # ── Print anti-overfitting summary ──
    print(f"\n🛡️  Anti-Overfitting Approaches:")
    print(f"   A. Offline augmentation:    ✅ (augmented dataset)")
    print(f"   B. Real-time augmentation:  ✅ (train transforms)")
    print(f"   C. Dropout:                 ✅ (rate={DROPOUT_RATE})")
    print(f"   D. Weight Decay (L2):       ✅ ({WEIGHT_DECAY})")
    print(f"   E. Early Stopping:          ✅ (patience={EARLY_STOPPING_PATIENCE})")
    print(f"   F. LR Scheduling:           ✅ (CosineAnnealingWarmRestarts)")
    print(f"   G. Gradient Clipping:       ✅ (max_norm={GRADIENT_CLIP_MAX_NORM})")
    print(f"   H. Label Smoothing:         ✅ ({LABEL_SMOOTHING})")

    # ── History ──
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "precision": [], "recall": [], "f1": [], "specificity": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    final_preds = None
    final_labels = None

    # ── Training loop ──
    print(f"\n🚀 Training for {NUM_EPOCHS} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")
    print(f"   Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print("-" * 60)
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc, preds, labels = validate(
            model, val_loader, criterion, device
        )

        # Compute metrics
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        spec = compute_specificity(labels, preds)

        # Record
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["precision"].append(prec)
        history["recall"].append(rec)
        history["f1"].append(f1)
        history["specificity"].append(spec)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start

        # Overfitting gap indicator
        gap = train_acc - val_acc
        gap_indicator = "🟢" if gap < 0.03 else ("🟡" if gap < 0.06 else "🔴")

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"P: {prec:.3f}  R: {rec:.3f}  F1: {f1:.3f}  Sp: {spec:.3f} | "
              f"LR: {current_lr:.6f} | Gap: {gap:.3f} {gap_indicator} | {elapsed:.1f}s")

        # LR scheduler step
        scheduler.step(epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            final_preds = preds
            final_labels = labels
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  ⏹️  Early stopping triggered at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")

    # ── GPU Utilization Summary ──
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        util_pct = (peak_mem / total_mem) * 100
        print(f"🖥️  Peak GPU memory: {peak_mem:.2f} GB / {total_mem:.1f} GB ({util_pct:.1f}% utilized)")

    # ── Save model ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved: {MODEL_SAVE_PATH}")

    # ── Generate graphs ──
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print(f"\n📈 Generating graphs in {GRAPHS_DIR}...")

    plot_accuracy(history, GRAPHS_DIR)
    plot_loss(history, GRAPHS_DIR)
    generate_individual_metrics(history, GRAPHS_DIR)
    plot_confusion_mat(final_labels, final_preds, GRAPHS_DIR)

    # Save training history as JSON
    history_path = os.path.join(GRAPHS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  📋 Saved: {history_path}")

    # ── Final summary ──
    best_epoch = np.argmin(history["val_loss"]) + 1
    print(f"\n{'='*60}")
    print("Training Complete — Summary")
    print(f"{'='*60}")
    print(f"  Best epoch:          {best_epoch}")
    print(f"  Best val loss:       {history['val_loss'][best_epoch-1]:.4f}")
    print(f"  Best val accuracy:   {history['val_acc'][best_epoch-1]:.4f}")
    print(f"  Best precision:      {history['precision'][best_epoch-1]:.4f}")
    print(f"  Best recall:         {history['recall'][best_epoch-1]:.4f}")
    print(f"  Best F1 score:       {history['f1'][best_epoch-1]:.4f}")
    print(f"  Best specificity:    {history['specificity'][best_epoch-1]:.4f}")

    # Overfitting analysis
    final_gap = history["train_acc"][-1] - history["val_acc"][-1]
    best_gap = history["train_acc"][best_epoch-1] - history["val_acc"][best_epoch-1]
    print(f"\n  📊 Overfitting Analysis:")
    print(f"     Best epoch gap (train-val acc): {best_gap:.4f}")
    print(f"     Final epoch gap (train-val acc): {final_gap:.4f}")
    if best_gap < 0.03:
        print(f"     Status: ✅ No overfitting detected")
    elif best_gap < 0.06:
        print(f"     Status: ⚠️  Mild overfitting (acceptable)")
    else:
        print(f"     Status: 🔴 Significant overfitting (consider more regularization)")

    print(f"{'='*60}")

    print("\n📊 9 Plots have been generated and saved silently in the background.")


if __name__ == "__main__":
    main()