"""
Multi-Architecture Pneumonia Detection — Training Script
===========================================================
Trains different CNN architectures on the original chest X-ray dataset.
Models: ResNet18, DenseNet121, VGG19, MobileNetV3, EfficientNet-B0.

Strategy:
- Baseline kaggle dataset ("dataset/") - No offline augmentation.
- 30 Epochs Max + Early Stopping (patience=5) based on validation loss.
- Unified model architecture replacements for binary classification mapping.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    vgg19, VGG19_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)
import torchvision.transforms as transforms
import timm
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Configuration ───────────────────────────────────────────────
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
GRADIENT_CLIP_MAX_NORM = 1.0

NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Data uses Original Dataset (Not augmented)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")

torch.backends.cudnn.benchmark = True

# ─── Model Factory ─────────────────────────────────────────────
def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_resnet18.pt"
    
    elif model_name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_densenet121.pt"
        
    elif model_name == "vgg19":
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # VGG's classifier is a Sequential block, the last Linear is index 6
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_vgg19.pt"
        
    elif model_name == "mobilenetv3":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_mobilenetv3.pt"
        
    elif model_name == "efficientnetb0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_efficientnet_b0.pt"
        
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_resnet50.pt"

    elif model_name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(p=DROPOUT_RATE), nn.Linear(num_ftrs, 2))
        return model, "pneumonia_convnext_tiny.pt"
        
    elif model_name == "deit_tiny":
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=2)
        return model, "pneumonia_deit_tiny.pt"
        
    elif model_name == "swin_tiny":
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
        return model, "pneumonia_swin_tiny.pt"
        
    elif model_name == "vit_b_16":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        return model, "pneumonia_vit_b_16.pt"
        
    else:
        raise ValueError(f"Model {model_name} not supported.")

# ─── Dataset ─────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
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
                    
        print(f"  Loaded {len(self.images)} images (Normal: {self.labels.count(0)}, Pneumonia: {self.labels.count(1)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Same basic transforms applied for all models
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Metrics ─────────────────────────────────────────────────────
def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ─── Training / Eval loops ───────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="    Train", leave=False, unit="batch"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
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
        
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ─── Plotting ────────────────────────────────────────────────────
def plot_and_save(metric1, metric2, label1, label2, title, ylabel, filename, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(metric1) + 1)
    ax.plot(epochs, metric1, "b-o", markersize=4, label=label1)
    if metric2:
        ax.plot(epochs, metric2, "r-o", markersize=4, label=label2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_confusion_mat(y_true, y_pred, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"], ax=ax, annot_kws={"size": 16})
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ─── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-Model Pneumonia Detection Training")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet50", "densenet121", "vgg19", "mobilenetv3", "efficientnetb0", "convnext_tiny", "deit_tiny", "swin_tiny", "vit_b_16"])
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Pneumonia Detection — Training {args.model.upper()} (RP2 - Patience={EARLY_STOPPING_PATIENCE})")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    if not os.path.exists(train_dir):
        print(f"❌ Original dataset not found at {DATASET_DIR}! Exiting.")
        sys.exit(1)

    print(f"\n📂 Loading datasets from {DATASET_DIR}...")
    train_loader = DataLoader(ChestXrayDataset(train_dir, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, drop_last=True)
    val_loader = DataLoader(ChestXrayDataset(val_dir, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    print(f"\n🧠 Initializing {args.model.upper()} architecture...")
    model, save_path = get_model(args.model)
    model = model.to(device)
    model_save_abs_path = os.path.join(os.path.dirname(__file__), save_path)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "precision": [], "recall": [], "f1": [], "specificity": [], "lr": []}

    best_val_loss = float("inf")
    patience_counter = 0
    final_preds, final_labels = None, None

    print(f"\n🚀 Training {args.model.upper()} for Max {NUM_EPOCHS} epochs (Early Stopping Patience={EARLY_STOPPING_PATIENCE})...")
    print("-" * 60)
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        v_loss, v_acc, preds, labels = validate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        spec = compute_specificity(labels, preds)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["precision"].append(prec)
        history["recall"].append(rec)
        history["f1"].append(f1)
        history["specificity"].append(spec)
        history["lr"].append(current_lr)

        gap = t_acc - v_acc
        indicator = "🟢" if gap < 0.03 else ("🟡" if gap < 0.06 else "🔴")
        print(f"  Epoch {epoch:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | F1: {f1:.3f} | Gap: {gap:.3f} {indicator} | {(time.time() - epoch_start):.1f}s")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_abs_path)
            final_preds, final_labels = preds, labels
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch} (No improvement in {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print("-" * 60)
    print(f"⏱️ Total training time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"💾 Best weights restored and saved to: {model_save_abs_path}")

    # Generate Graphs
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    print(f"\n📈 Generating Graphs...")
    plot_and_save(history["train_acc"], history["val_acc"], "Train", "Val", f"{args.model.upper()} Accuracy", "Accuracy", "accuracy_plot.png", GRAPHS_DIR)
    plot_and_save(history["train_loss"], history["val_loss"], "Train", "Val", f"{args.model.upper()} Loss", "Loss", "loss_plot.png", GRAPHS_DIR)
    plot_and_save(history["precision"], None, "Precision", None, f"{args.model.upper()} Precision", "Score", "precision_plot.png", GRAPHS_DIR)
    plot_and_save(history["recall"], None, "Recall", None, f"{args.model.upper()} Recall", "Score", "recall_plot.png", GRAPHS_DIR)
    plot_and_save(history["f1"], None, "F1", None, f"{args.model.upper()} F1", "Score", "f1_score_plot.png", GRAPHS_DIR)
    plot_and_save(history["specificity"], None, "Specificity", None, f"{args.model.upper()} Specificity", "Score", "specificity_plot.png", GRAPHS_DIR)
    plot_and_save(history["lr"], None, "LR", None, f"{args.model.upper()} Learning Rate", "LR", "learning_rate_plot.png", GRAPHS_DIR)
    plot_confusion_mat(final_labels, final_preds, GRAPHS_DIR)
    
    with open(os.path.join(GRAPHS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_epoch = np.argmin(history["val_loss"]) + 1
    print(f"\n{'='*60}\n✅ Training Complete — {args.model.upper()} Summary\n{'='*60}")
    print(f"  Best Validation Performance was at Epoch {best_epoch}")
    print(f"  Val Loss:      {history['val_loss'][best_epoch-1]:.4f}")
    print(f"  Val Accuracy:  {history['val_acc'][best_epoch-1]:.4f}")
    print(f"  F1 Score:      {history['f1'][best_epoch-1]:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
