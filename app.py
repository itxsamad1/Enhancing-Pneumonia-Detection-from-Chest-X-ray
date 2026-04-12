"""
Pneumonia Detection — Streamlit Web Application
=================================================
Interactive web interface for pneumonia detection from chest X-ray images.
Uses a fine-tuned ResNet-18 model with Grad-CAM visualization.
"""

import os
import sys
import logging
import time
import io
import shutil
import random
from pathlib import Path

# Configure minimal settings to reduce errors
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

# Import core libraries
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_resnet18.pt")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "assets", "sample_images")

# Ensure the sample directory exists
os.makedirs(SAMPLE_DIR, exist_ok=True)


# ─── Model Loading ──────────────────────────────────────────────
def load_model(model_path):
    """Load the trained ResNet-18 model for pneumonia detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, {"device": device}


# ─── Preprocessing ───────────────────────────────────────────────
def preprocess_image(image, options):
    """Apply selected preprocessing techniques to the image."""
    img_array = np.array(image)
    applied_techniques = []

    # Convert to grayscale for preprocessing
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    if options.get("clahe", False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        applied_techniques.append("CLAHE")

    if options.get("histogram_eq", False):
        gray = cv2.equalizeHist(gray)
        applied_techniques.append("Histogram Equalization")

    if options.get("denoising", False):
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        applied_techniques.append("Denoising")

    if options.get("sharpening", False):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        applied_techniques.append("Image Sharpening")

    # Convert back to RGB
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return result, applied_techniques


# ─── Inference ───────────────────────────────────────────────────
def predict_image(model, image, device):
    """Make a prediction for a single image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        probability, predicted = torch.max(probabilities, 1)

    prediction = "Normal" if predicted.item() == 0 else "Pneumonia"
    probability = probability.item() * 100

    return {
        "prediction": prediction,
        "probability": probability,
        "class_idx": predicted.item()
    }


# ─── Grad-CAM ───────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_extractor = model.layer4[-1]
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.feature_extractor.register_forward_hook(forward_hook)
        self.feature_extractor.register_full_backward_hook(backward_hook)


def generate_gradcam(model, image, device):
    """Generate Grad-CAM visualization."""
    grad_cam = GradCAM(model)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    gradients = grad_cam.gradients
    activations = grad_cam.activations
    weights = gradients.mean((2, 3), keepdim=True)
    weighted_activations = (weights * activations).sum(1, keepdim=True)
    heatmap = F.relu(weighted_activations).squeeze().detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    original_image = np.array(image.resize((224, 224)))

    return original_image, heatmap


def overlay_gradcam(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return blended


# ─── Sample Images ───────────────────────────────────────────────
def setup_sample_images(force_refresh=False):
    """Setup sample images from provided files or create placeholders."""
    normal_path = os.path.join(SAMPLE_DIR, "normal.jpg")
    pneumonia_path = os.path.join(SAMPLE_DIR, "pneumonia.jpg")

    if not force_refresh:
        try:
            if os.path.exists(normal_path) and os.path.exists(pneumonia_path):
                Image.open(normal_path).convert("RGB")
                Image.open(pneumonia_path).convert("RGB")
                return True
        except Exception:
            pass

    # Check root directory for sample images
    root_normal = os.path.join(os.path.dirname(__file__), "normal.jpeg")
    root_pneumonia = os.path.join(os.path.dirname(__file__), "pneumonia.jpg")

    if os.path.exists(root_normal) and os.path.exists(root_pneumonia):
        try:
            shutil.copy(root_normal, normal_path)
            shutil.copy(root_pneumonia, pneumonia_path)
            return True
        except Exception:
            pass

    # Try dataset directory
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    for split in ["val", "train"]:
        normal_src = os.path.join(dataset_dir, split, "NORMAL")
        pneumonia_src = os.path.join(dataset_dir, split, "PNEUMONIA")
        if os.path.exists(normal_src) and os.path.exists(pneumonia_src):
            try:
                normal_files = [f for f in os.listdir(normal_src)
                                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                pneumonia_files = [f for f in os.listdir(pneumonia_src)
                                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if normal_files and pneumonia_files:
                    shutil.copy(os.path.join(normal_src, normal_files[0]), normal_path)
                    shutil.copy(os.path.join(pneumonia_src, pneumonia_files[0]), pneumonia_path)
                    return True
            except Exception:
                pass

    # Create placeholder images
    try:
        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont

        for label, path, color in [
            ("Normal Chest X-Ray", normal_path, "white"),
            ("Pneumonia Chest X-Ray", pneumonia_path, (245, 230, 230)),
        ]:
            img = Image.new("RGB", (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, 100), label, fill=(0, 0, 0), font=font)
            img.save(path)
        return True
    except Exception:
        return False


# ─── Main App ────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Pneumonia Detection App",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0077B6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-top: 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #0077B6;
        margin: 1rem 0;
    }
    .prediction-box-normal {
        background-color: #d1e7dd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #198754;
        margin: 1rem 0;
    }
    .prediction-box-pneumonia {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #dc3545;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #0077B6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">Pneumonia Detection from Chest X-rays</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center">Enhancing Pneumonia Detection from Chest X-ray Images '
        'using Image Preprocessing and Deep Learning</p>',
        unsafe_allow_html=True
    )

    # Setup sample images
    setup_sample_images()

    # Load model
    with st.spinner("Loading model... Please wait..."):
        model_result = get_model()
        if model_result is None or model_result[0] is None:
            st.error("⚠️ Model file not found. Please train the model first using train_pneumonia.py")
            st.stop()
        model, metadata = model_result

    device = metadata["device"]

    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Preprocessing Options</h2>', unsafe_allow_html=True)

        if "clahe" not in st.session_state:
            st.session_state.clahe = False
        if "histogram_eq" not in st.session_state:
            st.session_state.histogram_eq = False
        if "denoising" not in st.session_state:
            st.session_state.denoising = False
        if "sharpening" not in st.session_state:
            st.session_state.sharpening = False

        clahe = st.checkbox("Apply CLAHE", value=st.session_state.clahe, key="clahe_cb")
        histogram_eq = st.checkbox("Apply Histogram Equalization",
                                   value=st.session_state.histogram_eq, key="histeq_cb")
        denoising = st.checkbox("Apply Denoising",
                                value=st.session_state.denoising, key="denoise_cb")
        sharpening = st.checkbox("Apply Image Sharpening",
                                 value=st.session_state.sharpening, key="sharpen_cb")

        st.session_state.clahe = clahe
        st.session_state.histogram_eq = histogram_eq
        st.session_state.denoising = denoising
        st.session_state.sharpening = sharpening

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply All", key="apply_all_btn"):
                st.session_state.clahe = True
                st.session_state.histogram_eq = True
                st.session_state.denoising = True
                st.session_state.sharpening = True
                st.rerun()
        with col2:
            if st.button("Apply None", key="apply_none_btn"):
                st.session_state.clahe = False
                st.session_state.histogram_eq = False
                st.session_state.denoising = False
                st.session_state.sharpening = False
                st.rerun()

        preprocessing_options = {
            "clahe": st.session_state.clahe,
            "histogram_eq": st.session_state.histogram_eq,
            "denoising": st.session_state.denoising,
            "sharpening": st.session_state.sharpening,
        }
        st.session_state.preprocessing_options = preprocessing_options

        with st.expander("About Preprocessing Techniques"):
            st.markdown("""
            ### CLAHE
            Contrast Limited Adaptive Histogram Equalization enhances local contrast
            while limiting noise amplification.

            ### Histogram Equalization
            Improves global contrast by spreading out the most frequent intensity values.

            ### Denoising
            Reduces noise while preserving important features and details.
            """)

        st.markdown("---")
        st.markdown("### Don't have an image to test?")
        sample_col1, sample_col2 = st.columns(2)

        normal_path = os.path.join(SAMPLE_DIR, "normal.jpg")
        pneumonia_path = os.path.join(SAMPLE_DIR, "pneumonia.jpg")
        has_samples = os.path.exists(normal_path) and os.path.exists(pneumonia_path)

        if has_samples:
            with sample_col1:
                if st.button("Load Normal", key="load_normal_btn"):
                    try:
                        st.session_state.uploaded_image = Image.open(normal_path).convert("RGB")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            with sample_col2:
                if st.button("Load Pneumonia", key="load_pneumonia_btn"):
                    try:
                        st.session_state.uploaded_image = Image.open(pneumonia_path).convert("RGB")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2 class="sub-header">Upload Chest X-ray Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image (JPG/PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.image(st.session_state.uploaded_image, caption="Uploaded Chest X-ray",
                         use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.pop("uploaded_image", None)
        elif "uploaded_image" in st.session_state:
            st.image(st.session_state.uploaded_image, caption="Uploaded Chest X-ray",
                     use_container_width=True)
        else:
            st.info("Please upload a chest X-ray image or select a sample image.")

    process_button = st.button(
        "Detect Pneumonia", type="primary",
        disabled="uploaded_image" not in st.session_state
    )

    if process_button and "uploaded_image" in st.session_state:
        preprocessing_options = st.session_state.get(
            "preprocessing_options",
            {"clahe": False, "histogram_eq": False, "denoising": False}
        )

        with st.spinner("Processing image..."):
            try:
                original_image = st.session_state.uploaded_image.copy()
                processed_array, applied_techniques = preprocess_image(
                    original_image, preprocessing_options
                )
                processed_image = Image.fromarray(processed_array)

                prediction_results = predict_image(model, processed_image, device)

                with st.spinner("Generating visualization..."):
                    original_np, heatmap = generate_gradcam(model, processed_image, device)
                    heatmap_overlay = overlay_gradcam(original_np, heatmap)

                st.session_state.processed_image = processed_image
                st.session_state.prediction_results = prediction_results
                st.session_state.applied_techniques = applied_techniques
                st.session_state.heatmap_overlay = heatmap_overlay
                st.rerun()

            except Exception as e:
                st.error(f"Error processing image: {e}")
                logger.exception("Error in processing pipeline")

    # Show results
    if ("processed_image" in st.session_state and
            "prediction_results" in st.session_state and
            "applied_techniques" in st.session_state):

        col1, col2 = st.columns(2)

        col1.markdown('<h3 class="sub-header">Original Image</h3>', unsafe_allow_html=True)
        col1.image(st.session_state.uploaded_image, caption="Original Chest X-ray",
                   use_container_width=True)

        col2.markdown('<h3 class="sub-header">Processed Image</h3>', unsafe_allow_html=True)
        col2.image(st.session_state.processed_image, caption="Processed X-ray",
                   use_container_width=True)

        if st.session_state.applied_techniques:
            st.markdown("**Preprocessing Applied:**")
            st.markdown(", ".join(st.session_state.applied_techniques))
        else:
            st.markdown("**No preprocessing applied**")

        st.markdown("---")

        prediction = st.session_state.prediction_results["prediction"]
        probability = st.session_state.prediction_results["probability"]

        if prediction == "Normal":
            st.markdown('<div class="prediction-box-normal">', unsafe_allow_html=True)
            st.markdown("### Prediction: NORMAL")
            st.markdown(
                '<p style="font-size: 18px; color: #198754; font-weight: bold;">'
                'No pneumonia detected</p>', unsafe_allow_html=True
            )
            st.progress(probability / 100)
            st.markdown(f"**Confidence: {probability:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box-pneumonia">', unsafe_allow_html=True)
            st.markdown("### Prediction: PNEUMONIA DETECTED")
            st.markdown(
                '<p style="font-size: 18px; color: #dc3545; font-weight: bold;">'
                'Pneumonia detected in the X-ray image</p>', unsafe_allow_html=True
            )
            st.progress(probability / 100)
            st.markdown(f"**Confidence: {probability:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)

        if "heatmap_overlay" in st.session_state:
            st.markdown('<h3 class="sub-header">Model Attention Map (Grad-CAM)</h3>',
                        unsafe_allow_html=True)
            st.image(st.session_state.heatmap_overlay,
                     caption="Areas the model focused on for making the prediction",
                     use_container_width=True)

            with st.expander("About Grad-CAM"):
                st.markdown("""
                **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualizes which parts
                of the image the model is focusing on. Warmer colors (red/yellow) indicate areas
                that strongly influenced the model's decision.

                This helps in interpreting the model's decision and ensuring it's focusing on
                clinically relevant features.
                """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1.5rem 0;">
        <p>Part of the research paper:<br/>
        <strong>"Enhancing Pneumonia Detection from Chest X-ray Images
        using Image Preprocessing and Deep Learning"</strong></p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Load model and cache it."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None


if __name__ == "__main__":
    main()