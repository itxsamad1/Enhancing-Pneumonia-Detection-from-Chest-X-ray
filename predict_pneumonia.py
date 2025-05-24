import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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

def predict_image(model, image_path, transform):
    """Make a prediction on a single image"""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
    
    # Get the prediction and probability
    is_pneumonia = predicted_class.item() == 1
    pneumonia_probability = probabilities[0][1].item() * 100
    
    return is_pneumonia, pneumonia_probability, image

def main():
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Check if model exists
    model_path = 'pneumonia_resnet18.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Sample images
    sample_images = {
        'Normal Sample': 'pneumonia_project/data/samples/normal_sample.jpeg',
        'Pneumonia Sample': 'pneumonia_project/data/samples/pneumonia_sample.jpeg'
    }
    
    # Make predictions
    results = {}
    for name, path in sample_images.items():
        if os.path.exists(path):
            is_pneumonia, probability, image = predict_image(model, path, transform)
            
            # Display result
            result = "Pneumonia" if is_pneumonia else "Normal"
            confidence = probability if is_pneumonia else (100 - probability)
            print(f"{name}: Predicted as {result} with {confidence:.2f}% confidence")
            
            results[name] = {
                'path': path,
                'image': image,
                'is_pneumonia': is_pneumonia,
                'probability': probability
            }
        else:
            print(f"Warning: Image {path} not found")
    
    # Visualize results
    if results:
        plt.figure(figsize=(12, 6))
        
        for i, (name, result) in enumerate(results.items()):
            plt.subplot(1, len(results), i+1)
            plt.imshow(np.array(result['image']))
            
            # Create title with prediction
            prediction = "Pneumonia" if result['is_pneumonia'] else "Normal"
            confidence = result['probability'] if result['is_pneumonia'] else (100 - result['probability'])
            title = f"{name}\nPrediction: {prediction}\nConfidence: {confidence:.2f}%"
            
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('pneumonia_predictions.png')
        print("Saved predictions visualization to pneumonia_predictions.png")

if __name__ == "__main__":
    main() 