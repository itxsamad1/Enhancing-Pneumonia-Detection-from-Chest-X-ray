"""
Pneumonia Detection — Model Evaluation Script
===============================================
Evaluates the trained model on the validation set and generates:
- Classification report (precision, recall, F1)
- Confusion matrix plot
- ROC curve plot
"""

import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


class TestChestXrayDataset(Dataset):
    """Dataset loader for evaluation."""

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        for label_name, label_val in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            label_dir = os.path.join(data_dir, label_name)
            if os.path.exists(label_dir):
                count = 0
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.images.append(os.path.join(label_dir, img_name))
                        self.labels.append(label_val)
                        count += 1
                print(f"  Found {count} {label_name.lower()} images")

        print(f"  Total: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(model_path, device):
    """Load the trained model."""
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate and return predictions, labels, and probabilities."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"],
                annot_kws={"size": 16})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved ROC curve to {save_path}")


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model_path = os.path.join(os.path.dirname(__file__), "pneumonia_resnet18.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    # Use validation set for evaluation
    val_dir = os.path.join(os.path.dirname(__file__), "dataset", "val")
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory {val_dir} not found.")
        return

    print("Loading validation dataset...")
    test_dataset = TestChestXrayDataset(val_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Evaluating model...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device)

    print("\nClassification Report:")
    print(classification_report(labels, predictions,
                                target_names=["Normal", "Pneumonia"]))

    # Save plots to graphs/
    graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    plot_confusion_matrix(labels, predictions,
                          os.path.join(graphs_dir, "eval_confusion_matrix.png"))
    plot_roc_curve(labels, probabilities,
                   os.path.join(graphs_dir, "eval_roc_curve.png"))

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()