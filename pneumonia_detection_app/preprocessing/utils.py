import cv2
import numpy as np
from PIL import Image

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL Image."""
    if len(cv2_image.shape) == 2:  # If grayscale
        return Image.fromarray(cv2_image)
    else:  # If RGB/BGR
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    """Convert PIL Image to CV2 format."""
    numpy_image = np.array(pil_image)
    if len(numpy_image.shape) == 2:  # If grayscale
        return numpy_image
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

def apply_clahe(image):
    """Apply CLAHE to the image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel
        l = clahe.apply(l)
        # Merge channels
        lab = cv2.merge((l, a, b))
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)

def apply_histogram_eq(image):
    """Apply histogram equalization."""
    if len(image.shape) == 3:
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Equalize the Y channel
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def apply_denoising(image):
    """Apply denoising to the image."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def preprocess_image(image, options):
    """Preprocess the image with selected options."""
    # Convert PIL to CV2
    cv2_image = pil_to_cv2(image)
    
    # Initialize list to track applied techniques
    applied_techniques = []
    
    # Apply selected preprocessing techniques
    if options.get('clahe', False):
        cv2_image = apply_clahe(cv2_image)
        applied_techniques.append("CLAHE")
        
    if options.get('histogram_eq', False):
        cv2_image = apply_histogram_eq(cv2_image)
        applied_techniques.append("Histogram Equalization")
        
    if options.get('denoising', False):
        cv2_image = apply_denoising(cv2_image)
        applied_techniques.append("Denoising")
    
    # Ensure image is in the correct format (RGB)
    if len(cv2_image.shape) == 2:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    
    return cv2_image, applied_techniques 