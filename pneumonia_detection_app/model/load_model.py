import torch
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the trained model from the specified path."""
    try:
        # Determine the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a ResNet-18 model with 2 output classes
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        metadata = {
            "device": device,
            "input_size": (224, 224),
            "classes": ["Normal", "Pneumonia"]
        }
        
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None 