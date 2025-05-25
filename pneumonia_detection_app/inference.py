import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_extractor = model.layer4[-1]
        self.gradients = None
        self.activations = None
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None
            
        def forward_hook(module, input, output):
            self.activations = output
            return None
            
        self.feature_extractor.register_forward_hook(forward_hook)
        self.feature_extractor.register_full_backward_hook(backward_hook)

def predict_image(model, image, device):
    """Make a prediction for a single image."""
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        probability, predicted = torch.max(probabilities, 1)
    
    # Convert to Python types
    prediction = "Normal" if predicted.item() == 0 else "Pneumonia"
    probability = probability.item() * 100
    
    return {
        "prediction": prediction,
        "probability": probability,
        "class_idx": predicted.item()
    }

def generate_gradcam(model, image, device):
    """Generate Grad-CAM visualization."""
    # Initialize GradCAM
    grad_cam = GradCAM(model)
    model.eval()
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass
    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()
    
    # Generate heatmap
    gradients = grad_cam.gradients
    activations = grad_cam.activations
    
    # Weight the channels by corresponding gradients
    weights = gradients.mean((2, 3), keepdim=True)
    weighted_activations = (weights * activations).sum(1, keepdim=True)
    
    # Apply ReLU to focus on features that have a positive influence
    heatmap = F.relu(weighted_activations).squeeze().detach().cpu().numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert original image to numpy array
    original_image = np.array(image.resize((224, 224)))
    
    return original_image, heatmap

def overlay_gradcam(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    # Create colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Convert to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Blend images
    blended = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return blended 