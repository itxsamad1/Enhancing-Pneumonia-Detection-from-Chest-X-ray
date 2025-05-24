import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# Simple Dataset class for chest X-rays
class SimpleChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load pneumonia images
        pneumonia_dir = os.path.join(data_dir, 'train', 'PNEUMONIA')
        if os.path.exists(pneumonia_dir):
            for img_name in os.listdir(pneumonia_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(pneumonia_dir, img_name))
                    self.labels.append(1)  # Pneumonia
            print(f"Found {len(self.labels)} pneumonia images")
        
        # Load normal images
        normal_dir = os.path.join(data_dir, 'train', 'NORMAL')
        if os.path.exists(normal_dir):
            normal_count_before = len(self.labels) - sum(self.labels)
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)  # Normal
            normal_count_after = len(self.labels) - sum(self.labels)
            print(f"Found {normal_count_after - normal_count_before} normal images")
        
        print(f"Total dataset size: {len(self.images)}")
    
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

def create_model():
    # Load pre-trained model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs=2):
    # Training loop
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 20 == 19:  # Print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 20:.3f}, accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0
        
        # Print epoch statistics
        print(f'Epoch {epoch + 1} completed. Accuracy: {100 * correct / total:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'pneumonia_resnet18.pt')
    print('Model saved to pneumonia_resnet18.pt')
    
    return model

def main():
    # Set up transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and data loader
    dataset = SimpleChestXrayDataset('pneumonia_project/data/kaggle', transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create model
    model = create_model()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(model, train_loader, criterion, optimizer, num_epochs=50)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 