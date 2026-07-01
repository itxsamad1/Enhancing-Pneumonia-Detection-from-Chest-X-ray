import os
import csv
import json
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import seaborn as sns

MODEL_PATH = "pneumonia_resnet18.pt"
MANIFEST_PATH = "raw_datasets/nih_chestxray/extracted/manifest.csv"
OUT_DIR = "external_validation_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 2. Load model
print(f"Loading ResNet-18 model from {MODEL_PATH}...")
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 3. Preprocessing Transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Load dataset from manifest
print(f"Reading manifest: {MANIFEST_PATH}...")
image_paths = []
labels = []
with open(MANIFEST_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if row:
            image_paths.append(row[0])
            labels.append(int(row[1]))

labels = np.array(labels)
print(f"Loaded {len(image_paths)} images (Pneumonia: {np.sum(labels == 1)}, Normal: {np.sum(labels == 0)})")

# 5. Run inference
all_preds = []
all_probs = []

print("Running inference...")
with torch.no_grad():
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"  [WARN] File not found: {img_path}")
            continue
        
        img = Image.open(img_path).convert("RGB")
        tensor = val_transform(img).unsqueeze(0).to(device)
        
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        
        pred = outputs.argmax(dim=1).cpu().item()
        prob_pneumonia = probs[0, 1].cpu().item()
        
        all_preds.append(pred)
        all_probs.append(prob_pneumonia)

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 6. Compute metrics
cm = confusion_matrix(labels, all_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(labels, all_preds)
prec = precision_score(labels, all_preds, zero_division=0)
rec = recall_score(labels, all_preds, zero_division=0)
f1 = f1_score(labels, all_preds, zero_division=0)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc = roc_auc_score(labels, all_probs)

metrics = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "specificity": round(float(spec), 4),
    "f1_score": round(float(f1), 4),
    "auc": round(float(auc), 4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
}

print("\n--- External Validation Metrics ---")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f} (Sensitivity)")
print(f"Specificity: {spec:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"AUC        : {auc:.4f}")

# Save metrics JSON
with open("external_val_results.json", "w") as json_f:
    json.dump(metrics, json_f, indent=2)
print("Metrics saved to external_val_results.json")

# 7. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Normal", "Pneumonia"], 
            yticklabels=["Normal", "Pneumonia"],
            annot_kws={"size": 14, "weight": "bold"})
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.title('Confusion Matrix (External NIH Cohort)', fontsize=13, fontweight='bold')
plt.tight_layout()
cm_plot_path = os.path.join(OUT_DIR, "external_confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=150)
plt.close()
print(f"Confusion matrix plot saved to {cm_plot_path}")

# 8. Plot ROC Curve
fpr, tpr, _ = roc_curve(labels, all_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.title('ROC Curve (External NIH Cohort)', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_plot_path = os.path.join(OUT_DIR, "external_roc_curve.png")
plt.savefig(roc_plot_path, dpi=150)
plt.close()
print(f"ROC curve plot saved to {roc_plot_path}")
