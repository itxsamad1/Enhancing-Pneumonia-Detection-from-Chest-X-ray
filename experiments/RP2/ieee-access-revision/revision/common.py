"""
Shared utilities for IEEE Access revision experiments.
Reproduces the original StratifiedKFold(n_splits=5, shuffle=True, random_state=42) splits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "revision_results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
ORIG_KFOLD = ROOT / "kfold_results.json"

N_FOLDS = 5
MAX_EPOCHS = 50
PATIENCE = 8
# 64 under-filled the 5060 Ti (~2.5GB / 16GB). 256 fills more of the card
# and cuts steps/epoch ~4x vs Phase 3a; note in paper if comparing absolute LR dynamics.
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
GRADIENT_CLIP = 1.0
# Was 0 → GPU sat idle waiting on PIL loads (~10-20% util). 4 workers
# prefetches on CPU so the 5060 Ti stays busy without changing train math.
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
RANDOM_STATE = 42

MODEL_REGISTRY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "vgg19": "VGG-19",
    "efficientnetb0": "EfficientNet-B0",
    "mobilenetv3": "MobileNetV3-Small",
    "convnext_tiny": "ConvNeXt-Tiny",
    "deit_tiny": "DeiT-Tiny",
    "swin_tiny": "Swin-Tiny",
    "vit_b_16": "ViT-B/16",
    "mobilevit_s": "MobileViT-S",
    "tiny_vit_5m": "TinyViT-5M",
}

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


class ChestXrayDataset(Dataset):
    """Combined train+val pool; returns (img, label, path)."""

    def __init__(self, root_dir: Path = DATASET_DIR, transform=None):
        self.transform = transform
        self.images: list[str] = []
        self.labels: list[int] = []
        for split in ["train", "val"]:
            for cls, lbl in [("NORMAL", 0), ("PNEUMONIA", 1)]:
                d = root_dir / split / cls
                if not d.exists():
                    continue
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        self.images.append(str(f))
                        self.labels.append(lbl)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], path


class TransformSubset(Dataset):
    def __init__(self, base: ChestXrayDataset, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        path = self.base.images[idx]
        label = self.base.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


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
        m.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier.in_features, 2)
        )
    elif k == "vgg19":
        m = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[6].in_features, 2)
        )
    elif k == "efficientnetb0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[1].in_features, 2)
        )
    elif k == "mobilenetv3":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[3].in_features, 2)
        )
    elif k == "convnext_tiny":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(m.classifier[2].in_features, 2)
        )
    elif k == "deit_tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required")
        m = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=2)
    elif k == "swin_tiny":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required")
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=2)
    elif k == "vit_b_16":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required")
        m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    elif k == "mobilevit_s":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required")
        m = timm.create_model("mobilevit_s", pretrained=True, num_classes=2)
    elif k == "tiny_vit_5m":
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm required")
        m = timm.create_model("tiny_vit_5m_224", pretrained=True, num_classes=2)
    else:
        raise ValueError(f"Unknown model: {key}")
    return m


def make_criterion(strategy: str, class_counts: Optional[tuple[int, int]] = None):
    """Return (criterion, use_sampler_flag)."""
    if strategy in ("baseline", "low_lr_warmup", "threshold"):
        return nn.CrossEntropyLoss(), False
    if strategy == "weighted_ce":
        if class_counts is None:
            raise ValueError("class_counts required for weighted_ce")
        n0, n1 = class_counts
        # inverse-frequency weights, normalized to mean 1
        w = torch.tensor([1.0 / n0, 1.0 / n1], dtype=torch.float32)
        w = w / w.mean()
        return nn.CrossEntropyLoss(weight=w), False
    if strategy == "focal":
        return FocalLoss(gamma=2.0, alpha=None), False
    if strategy == "oversample":
        return nn.CrossEntropyLoss(), True
    if strategy == "feature_smote":
        # training uses embedding-SMOTE offline; at train time use weighted CE
        if class_counts is None:
            return nn.CrossEntropyLoss(), False
        n0, n1 = class_counts
        w = torch.tensor([1.0 / n0, 1.0 / n1], dtype=torch.float32)
        w = w / w.mean()
        return nn.CrossEntropyLoss(weight=w), False
    raise ValueError(f"Unknown strategy: {strategy}")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets]
            loss = at * loss
        return loss.mean()


def learning_rate_for(strategy: str, model_key: str) -> float:
    if strategy == "low_lr_warmup":
        return 1e-4
    return LEARNING_RATE


def is_collapsed(preds, labels, auc: float | None = None) -> bool:
    """Constant (or near-constant) predictor = majority-class collapse."""
    preds = np.asarray(preds)
    unique = np.unique(preds)
    if len(unique) == 1:
        return True
    if auc is not None and abs(float(auc) - 0.5) < 1e-6:
        return True
    # < 1% positive predictions among pneumonia-labelled samples
    labels = np.asarray(labels)
    pos = labels == 1
    if pos.sum() > 0 and (preds[pos] == 1).mean() < 0.01:
        return True
    return False


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    for d in (RESULTS_DIR, CKPT_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
