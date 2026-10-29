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

# Dataset class for test images
class TestChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load pneumonia images
        pneumonia_dir = os.path.join(data_dir, 'test', 'PNEUMONIA')
        if os.path.exists(pneumonia_dir):
            for img_name in os.listdir(pneumonia_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(pneumonia_dir, img_name))
                    self.labels.append(1)  # Pneumonia
            print(f"Found {len(self.labels)} pneumonia test images")
        
        # Load normal images
        normal_dir = os.path.join(data_dir, 'test', 'NORMAL')
        if os.path.exists(normal_dir):
            normal_count_before = len(self.labels) - sum(self.labels)
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)  # Normal
            normal_count_after = len(self.labels) - sum(self.labels)
            print(f"Found {normal_count_after - normal_count_before} normal test images")
        
        print(f"Total test dataset size: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_model(model_path):
    """Load the trained pneumonia detection model"""
    # Create a ResNet18 model with 2 output classes
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset"""
    model = model.to(device)
    
    # Collect predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of pneumonia
    
    return all_preds, all_labels, all_probs

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")

def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved ROC curve to {save_path}")

def main():
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model exists
    model_path = 'pneumonia_resnet18.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load test dataset
    test_dataset = TestChestXrayDataset('pneumonia_project/data/kaggle', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Normal', 'Pneumonia']))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions)
    
    # Plot ROC curve
    plot_roc_curve(labels, probabilities)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 