"""
5-Fold Stratified Cross-Validation — Phase 5
==============================================
Runs stratified 5-fold CV for each architecture on the combined dataset.
Computes: Mean ± Std for Accuracy, F1, Precision, Recall, Specificity, AUC.
Runs McNemar's test between every model pair.

Features:
  - Crash-safe: saves results after every single fold
  - Resume-aware: skips already-completed folds
  - Per-fold progress visible in terminal
  - Saves full results to kfold_results.json

Usage:
  python kfold_trainer.py --model resnet18        # run one model
  python kfold_trainer.py --all                   # run all 10 (overnight)
  python kfold_trainer.py --mcnemar               # only run McNemar tests on saved results

Time estimate per model: ~30-60 min on RTX 5060 Ti (5 folds × ~15 epochs avg)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import time
import argparse
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    vgg19, VGG19_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import chi2_contingency
from tqdm import tqdm

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent
DATASET_DIR        = BASE_DIR / "dataset"
RESULTS_FILE       = BASE_DIR / "kfold_results.json"

N_FOLDS            = 5
MAX_EPOCHS_PER_FOLD = 50
PATIENCE           = 8
BATCH_SIZE         = 64
LEARNING_RATE      = 0.001
WEIGHT_DECAY       = 1e-4
DROPOUT_RATE       = 0.3
GRADIENT_CLIP      = 1.0
NUM_WORKERS        = 4
PIN_MEMORY         = True
RANDOM_STATE       = 42

torch.backends.cudnn.benchmark = True

MODEL_REGISTRY = {
    "resnet18"      : "ResNet-18",
    "resnet50"      : "ResNet-50",
    "densenet121"   : "DenseNet-121",
    "vgg19"         : "VGG-19",
    "efficientnetb0": "EfficientNet-B0",
    "mobilenetv3"   : "MobileNetV3-Small",
    "convnext_tiny" : "ConvNeXt-Tiny",
    "deit_tiny"     : "DeiT-Tiny",
    "swin_tiny"     : "Swin-Tiny",
    "vit_b_16"      : "ViT-B/16",
}

# ─── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """Flat dataset that loads all images from both train/ and val/ splits."""

    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform
        self.images    = []
        self.labels    = []

        for split in ["train", "val"]:
            for cls, lbl in [("NORMAL", 0), ("PNEUMONIA", 1)]:
                d = root_dir / split / cls
                if d.exists():
                    for f in d.iterdir():
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                            self.images.append(str(f))
                            self.labels.append(lbl)

        print(f"    Combined pool: {len(self.images)} images "
              f"(Normal: {self.labels.count(0)}, Pneumonia: {self.labels.count(1)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class TransformSubset(Dataset):
    """Wraps a Subset and applies a specific transform."""
    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path = self.subset.dataset.images[self.subset.indices[idx]]
        label    = self.subset.dataset.labels[self.subset.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── Model Factory ────────────────────────────────────────────────────────────
def build_model(key: str) -> nn.Module:
    k = key.lower()
    if k == "resnet18":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.fc.in_features, 2))
    elif k == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.fc.in_features, 2))
    elif k == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier.in_features, 2))
    elif k == "vgg19":
        m = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[6].in_features, 2))
    elif k == "efficientnetb0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[1].in_features, 2))
    elif k == "mobilenetv3":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[3].in_features, 2))
    elif k == "convnext_tiny":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[2].in_features, 2))
    elif k == "deit_tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm not installed")
        m = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=2)
    elif k == "swin_tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm not installed")
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=2)
    elif k == "vit_b_16":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm not installed")
        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    else:
        raise ValueError(f"Unknown model key: {key}")
    return m


# ─── Train / Eval helpers ─────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == lbls).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_fold(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out  = model(imgs)
            loss = criterion(out, lbls)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(lbls.cpu().numpy())
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "val_loss"  : total_loss / total,
        "accuracy"  : accuracy_score(all_labels, all_preds),
        "precision" : precision_score(all_labels, all_preds, zero_division=0),
        "recall"    : recall_score(all_labels, all_preds, zero_division=0),
        "f1"        : f1_score(all_labels, all_preds, zero_division=0),
        "specificity": spec,
        "auc"       : roc_auc_score(all_labels, all_probs),
        "preds"     : all_preds.tolist(),
        "labels"    : all_labels.tolist(),
    }


def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# ─── Results I/O ─────────────────────────────────────────────────────────────
def load_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ─── Run CV for one model ─────────────────────────────────────────────────────
def run_kfold(model_key: str, dataset: ChestXrayDataset,
              device: torch.device, results: dict) -> dict:

    display = MODEL_REGISTRY[model_key]
    print(f"\n{'='*60}")
    print(f"  {N_FOLDS}-Fold CV: {display}")
    print(f"{'='*60}")

    if model_key not in results:
        results[model_key] = {"display": display, "folds": {}}

    labels_all = np.array(dataset.labels)
    indices    = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_all), 1):
        fold_key = f"fold_{fold_idx}"

        # Resume: skip completed folds
        if fold_key in results[model_key]["folds"]:
            m = results[model_key]["folds"][fold_key]
            print(f"  Fold {fold_idx}/{N_FOLDS}  [SKIPPED — already done]"
                  f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
            continue

        print(f"\n  Fold {fold_idx}/{N_FOLDS}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        train_sub = TransformSubset(Subset(dataset, train_idx), train_transform)
        val_sub   = TransformSubset(Subset(dataset, val_idx),   val_transform)

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  drop_last=True)
        val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model     = build_model(model_key).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        scaler    = torch.amp.GradScaler("cuda")

        best_val_loss   = float("inf")
        patience_counter = 0
        best_metrics    = None
        fold_start      = time.time()

        for epoch in range(1, MAX_EPOCHS_PER_FOLD + 1):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            metrics       = eval_fold(model, val_loader, criterion, device)
            scheduler.step(metrics["val_loss"])

            print(f"    Ep {epoch:02d}  TrainLoss={t_loss:.4f} Acc={t_acc:.4f}"
                  f"  ValLoss={metrics['val_loss']:.4f} Acc={metrics['accuracy']:.4f}"
                  f"  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

            if metrics["val_loss"] < best_val_loss:
                best_val_loss    = metrics["val_loss"]
                patience_counter = 0
                best_metrics     = metrics.copy()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch}")
                    break

        elapsed = (time.time() - fold_start) / 60
        print(f"\n  Fold {fold_idx} done in {elapsed:.1f} min "
              f"| Best -> Acc={best_metrics['accuracy']:.4f}  F1={best_metrics['f1']:.4f}"
              f"  AUC={best_metrics['auc']:.4f}")

        # Save fold result immediately (crash-safe)
        results[model_key]["folds"][fold_key] = {k: v for k, v in best_metrics.items()}
        save_results(results)

        # Clean up GPU memory
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()

    # Aggregate fold results
    fold_data   = results[model_key]["folds"]
    metrics_keys = ["accuracy", "precision", "recall", "f1", "specificity", "auc"]
    summary = {}
    for mk in metrics_keys:
        vals = [fold_data[f"fold_{i}"][mk] for i in range(1, N_FOLDS + 1)
                if f"fold_{i}" in fold_data]
        if vals:
            summary[f"{mk}_mean"] = round(statistics.mean(vals), 4)
            summary[f"{mk}_std"]  = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0

    results[model_key]["summary"] = summary
    save_results(results)

    print(f"\n  {display} — {N_FOLDS}-Fold Summary:")
    for mk in metrics_keys:
        m = summary.get(f"{mk}_mean", "?")
        s = summary.get(f"{mk}_std",  "?")
        print(f"    {mk:<12}: {m:.4f} ± {s:.4f}")

    return results


# ─── McNemar's Test ───────────────────────────────────────────────────────────
def run_mcnemar(results: dict):
    """
    Pairwise McNemar's test between all model pairs using combined fold predictions.
    p < 0.05 → statistically significant difference.
    """
    print(f"\n{'='*60}")
    print("  McNemar's Test — Pairwise Statistical Significance")
    print(f"{'='*60}")

    # Aggregate all predictions per model
    model_preds  = {}
    model_labels = {}

    for key, data in results.items():
        if "folds" not in data:
            continue
        preds_all  = []
        labels_all = []
        for fold_data in data["folds"].values():
            if "preds" in fold_data and "labels" in fold_data:
                preds_all.extend(fold_data["preds"])
                labels_all.extend(fold_data["labels"])
        if preds_all:
            model_preds[key]  = np.array(preds_all)
            model_labels[key] = np.array(labels_all)

    keys = list(model_preds.keys())
    if len(keys) < 2:
        print("  Need at least 2 models with predictions to run McNemar's test.")
        return {}

    # Ensure all models used the same label order
    mcnemar_results = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            # Use intersection of samples (same length check)
            min_len = min(len(model_preds[k1]), len(model_preds[k2]))
            p1 = model_preds[k1][:min_len]
            p2 = model_preds[k2][:min_len]
            lb = model_labels[k1][:min_len]

            # McNemar contingency table
            # b = model1 correct, model2 wrong
            # c = model1 wrong, model2 correct
            correct1 = (p1 == lb)
            correct2 = (p2 == lb)
            b = np.sum( correct1 & ~correct2)
            c = np.sum(~correct1 &  correct2)

            if b + c == 0:
                p_value = 1.0
            else:
                # McNemar's chi-squared with continuity correction
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                from scipy.stats import chi2 as chi2_dist
                p_value = 1 - chi2_dist.cdf(chi2, df=1)

            sig = "**SIGNIFICANT**" if p_value < 0.05 else "not significant"
            pair = f"{MODEL_REGISTRY.get(k1, k1)} vs {MODEL_REGISTRY.get(k2, k2)}"
            print(f"  {pair:<45}  p={p_value:.4f}  {sig}")
            mcnemar_results[f"{k1}_vs_{k2}"] = {
                "model_a": MODEL_REGISTRY.get(k1, k1),
                "model_b": MODEL_REGISTRY.get(k2, k2),
                "b": int(b), "c": int(c),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05,
            }

    results["mcnemar"] = mcnemar_results
    save_results(results)
    return mcnemar_results


# ─── Summary Table ────────────────────────────────────────────────────────────
def print_summary_table(results: dict):
    sep = "-" * 95
    hdr = (f"{'Model':<20} {'Acc':>10} {'F1':>10} {'Prec':>10} "
           f"{'Recall':>10} {'Spec':>10} {'AUC':>10}")
    print(f"\n\n{'='*95}")
    print("  5-Fold Cross-Validation Summary (Mean ± Std)")
    print(f"{'='*95}")
    print(hdr)
    print(sep)

    for key, data in results.items():
        if key == "mcnemar" or "summary" not in data:
            continue
        s = data["summary"]
        name = data.get("display", key)

        def fmt(mk):
            m = s.get(f"{mk}_mean", "?")
            std = s.get(f"{mk}_std", "?")
            return f"{m:.3f}±{std:.3f}" if isinstance(m, float) else "?"

        print(f"  {name:<18} {fmt('accuracy'):>10} {fmt('f1'):>10} "
              f"{fmt('precision'):>10} {fmt('recall'):>10} "
              f"{fmt('specificity'):>10} {fmt('auc'):>10}")
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation for Pneumonia Detection")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Run CV for a single model.")
    parser.add_argument("--all", action="store_true",
                        help="Run CV for all 10 architectures (overnight).")
    parser.add_argument("--mcnemar", action="store_true",
                        help="Only run McNemar's test on already-saved results.")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table from saved results.")
    args = parser.parse_args()

    results = load_results()

    if args.mcnemar or args.summary:
        if args.mcnemar:
            run_mcnemar(results)
        if args.summary:
            print_summary_table(results)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Pneumonia Detection — 5-Fold Cross-Validation (Phase 5)")
    print(f"{'='*60}")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  Folds    : {N_FOLDS}")
    print(f"  Max Ep   : {MAX_EPOCHS_PER_FOLD}  |  Patience: {PATIENCE}")
    print(f"  Dataset  : {DATASET_DIR}")

    print(f"\n  Loading combined dataset (train + val)...")
    dataset = ChestXrayDataset(DATASET_DIR)

    if args.model:
        keys_to_run = [args.model]
    elif args.all:
        keys_to_run = list(MODEL_REGISTRY.keys())
    else:
        parser.print_help()
        print("\n  TIP: Run with --model <name> or --all")
        return

    start_total = time.time()
    for key in keys_to_run:
        results = run_kfold(key, dataset, device, results)

    if len(keys_to_run) > 1:
        run_mcnemar(results)
        print_summary_table(results)

    elapsed_total = (time.time() - start_total) / 60
    print(f"\n  Total time: {elapsed_total:.1f} minutes")
    print(f"  Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
