import os
import sys
import logging
import time
import io
import shutil
import random
from pathlib import Path
import glob

# Configure minimal settings to reduce errors
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

# Import core libraries
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add the current directory to sys.path to resolve module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pneumonia_detection_app"))

# Import custom modules - wrap in try/except for demo mode
try:
    from pneumonia_detection_app.preprocessing.utils import preprocess_image, cv2_to_pil
    from pneumonia_detection_app.model.load_model import load_model
    from pneumonia_detection_app.inference import predict_image, generate_gradcam, overlay_gradcam
    DEMO_MODE = False
except Exception as e:
    DEMO_MODE = True
    print(f"Error importing modules: {e}. Running in demo mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path constants
MODEL_PATH = "pneumonia_resnet18.pt"  # Use the 100% accurate model
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "assets", "sample_images")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Chest X-Ray Images(Pneumonia)", "chest_xray")

# Ensure the sample directory exists
os.makedirs(SAMPLE_DIR, exist_ok=True)

def setup_sample_images(force_refresh=False):
    """Setup sample images from the dataset or provided files.
    Uses specific high-quality images rather than random selection."""
    normal_path = os.path.join(SAMPLE_DIR, "normal.jpg")
    pneumonia_path = os.path.join(SAMPLE_DIR, "pneumonia.jpg")
    
    # If files already exist and are valid and no refresh is requested, keep them
    if not force_refresh:
        try:
            if os.path.exists(normal_path) and os.path.exists(pneumonia_path):
                # Test if these images can be opened
                Image.open(normal_path).convert("RGB")
                Image.open(pneumonia_path).convert("RGB")
                return True
        except Exception as e:
            print(f"Existing sample images are not valid: {e}, will replace them")
    
    # First check for pasted images in the root directory
    root_normal = os.path.join(os.path.dirname(__file__), "normal.jpeg")
    root_pneumonia = os.path.join(os.path.dirname(__file__), "pneumonia.jpg")
    
    success = False
    
    # Try to use the pasted images first
    if os.path.exists(root_normal) and os.path.exists(root_pneumonia):
        try:
            # Copy to sample directory
            shutil.copy(root_normal, normal_path)
            shutil.copy(root_pneumonia, pneumonia_path)
            
            # Verify the images
            Image.open(normal_path).convert("RGB")
            Image.open(pneumonia_path).convert("RGB")
            print("Using user-provided images for samples")
            success = True
        except Exception as e:
            print(f"Error using user-provided images: {e}")
            # Will try the dataset next
    
    # If user images couldn't be used, try the dataset with specific files
    if not success and os.path.exists(DATASET_DIR):
        try:
            # Use specific high-quality images from dataset
            normal_dataset_dir = os.path.join(DATASET_DIR, "train", "NORMAL")
            pneumonia_dataset_dir = os.path.join(DATASET_DIR, "train", "PNEUMONIA")
            
            # Specific high-quality normal image
            normal_file = "NORMAL2-IM-1427-0001.jpeg"  # A clear, high-quality normal image
            if not os.path.exists(os.path.join(normal_dataset_dir, normal_file)):
                # Fallback if the specific file doesn't exist
                normal_files = os.listdir(normal_dataset_dir)
                if normal_files:
                    normal_file = normal_files[0]
            
            normal_image_path = os.path.join(normal_dataset_dir, normal_file)
            if os.path.exists(normal_image_path):
                shutil.copy(normal_image_path, normal_path)
                print(f"Using specific normal image: {normal_file}")
            
            # Specific high-quality pneumonia image
            pneumonia_file = "person1_bacteria_1.jpeg"  # A clear case of pneumonia
            if not os.path.exists(os.path.join(pneumonia_dataset_dir, pneumonia_file)):
                # Fallback if the specific file doesn't exist
                pneumonia_files = os.listdir(pneumonia_dataset_dir)
                if pneumonia_files:
                    pneumonia_file = pneumonia_files[0]
            
            pneumonia_image_path = os.path.join(pneumonia_dataset_dir, pneumonia_file)
            if os.path.exists(pneumonia_image_path):
                shutil.copy(pneumonia_image_path, pneumonia_path)
                print(f"Using specific pneumonia image: {pneumonia_file}")
            
            # Verify the images
            Image.open(normal_path).convert("RGB")
            Image.open(pneumonia_path).convert("RGB")
            print("Using dataset images for samples")
            success = True
        except Exception as e:
            print(f"Error using dataset images: {e}")
            # Will try the fallback next
    
    # If still unsuccessful, use text-based fallback images
    if not success:
        try:
            # Create simple text-based images
            normal_img = Image.new('RGB', (224, 224), color='white')
            pneumonia_img = Image.new('RGB', (224, 224), color=(245, 230, 230))
            
            # Add text
            import PIL.ImageDraw as ImageDraw
            import PIL.ImageFont as ImageFont
            
            try:
                # Try to get a font
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            draw_normal = ImageDraw.Draw(normal_img)
            draw_normal.text((10, 100), "Normal Chest X-Ray Sample", fill=(0, 0, 0), font=font)
            
            draw_pneumonia = ImageDraw.Draw(pneumonia_img)
            draw_pneumonia.text((10, 100), "Pneumonia Chest X-Ray Sample", fill=(0, 0, 0), font=font)
            
            normal_img.save(normal_path)
            pneumonia_img.save(pneumonia_path)
            print("Created emergency backup sample images")
            success = True
        except Exception as e:
            print(f"Failed to create emergency backup sample images: {e}")
    
    return success

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Pneumonia Detection App",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS for better UI
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
    st.markdown('<h1 class="main-header">Pneumonia Detection from Chest X-rays</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center">Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning</p>', 
        unsafe_allow_html=True
    )
    
    if DEMO_MODE:
        st.warning("⚠️ Running in DEMO MODE. This is a demonstration with simulated predictions.")
    
    # Load model (real or demo)
    with st.spinner("Loading model... Please wait..."):
        model_load_result = get_model()
        if model_load_result is None or model_load_result[0] is None:
            st.error("⚠️ Error: Failed to load the model. Please check if the model file exists and is valid.")
            st.stop()
        else:
            model, metadata = model_load_result

    # Layout: sidebar and main content
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Preprocessing Options</h2>', unsafe_allow_html=True)
        
        # Preprocessing options
        if 'clahe' not in st.session_state:
            st.session_state.clahe = False
        if 'histogram_eq' not in st.session_state:
            st.session_state.histogram_eq = False
        if 'denoising' not in st.session_state:
            st.session_state.denoising = False
        
        clahe = st.checkbox("Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)", 
                           value=st.session_state.clahe, key="clahe_checkbox")
        histogram_eq = st.checkbox("Apply Histogram Equalization", 
                                  value=st.session_state.histogram_eq, key="histogram_checkbox")
        denoising = st.checkbox("Apply Denoising", 
                               value=st.session_state.denoising, key="denoising_checkbox")
        
        # Update session state based on checkboxes
        st.session_state.clahe = clahe
        st.session_state.histogram_eq = histogram_eq
        st.session_state.denoising = denoising
        
        # Preprocessing buttons
        col1, col2 = st.columns(2)
        
        # Use unique keys for buttons
        with col1:
            if st.button("Apply All", key="apply_all_btn"):
                st.session_state.clahe = True
                st.session_state.histogram_eq = True
                st.session_state.denoising = True
                st.rerun()
                
        with col2:
            if st.button("Apply None", key="apply_none_btn"):
                st.session_state.clahe = False
                st.session_state.histogram_eq = False
                st.session_state.denoising = False
                st.rerun()
        
        # Store preprocessing options in session state
        preprocessing_options = {
            "clahe": st.session_state.clahe,
            "histogram_eq": st.session_state.histogram_eq,
            "denoising": st.session_state.denoising
        }
        
        st.session_state.preprocessing_options = preprocessing_options
        
        # Preprocessing info
        with st.expander("About Preprocessing Techniques"):
            st.markdown("""
            ### CLAHE
            Contrast Limited Adaptive Histogram Equalization enhances local contrast while limiting noise amplification.
            
            ### Histogram Equalization
            Improves global contrast by effectively spreading out the most frequent intensity values.
            
            ### Denoising
            Reduces noise in the image while preserving important features and details.
            """)
            
        # Sample images section
        st.markdown("---")
        st.markdown("### Don't have an image to test?")
        sample_col1, sample_col2 = st.columns(2)
        
        # Check if sample images exist
        normal_path = os.path.join(SAMPLE_DIR, "normal.jpg")
        pneumonia_path = os.path.join(SAMPLE_DIR, "pneumonia.jpg")
        
        has_samples = os.path.exists(normal_path) and os.path.exists(pneumonia_path)
        
        if has_samples:
            with sample_col1:
                if st.button("Load Normal Sample", key="load_normal_btn"):
                    try:
                        img = Image.open(normal_path).convert("RGB")
                        st.session_state.uploaded_image = img
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample: {str(e)}")
            
            with sample_col2:
                if st.button("Load Pneumonia Sample", key="load_pneumonia_btn"):
                    try:
                        img = Image.open(pneumonia_path).convert("RGB")
                        st.session_state.uploaded_image = img
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample: {str(e)}")
                        
            # Add a button to refresh sample images if desired
            if st.button("Refresh Sample Images", key="refresh_samples_btn"):
                with st.spinner("Refreshing sample images..."):
                    setup_sample_images(force_refresh=True)
                st.success("Sample images refreshed!")
                st.rerun()
        else:
            st.info("Sample images not found. Please add sample images to the assets/sample_images directory.")

    # Main content area - using two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Chest X-ray Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image (JPG/PNG)", 
            type=["jpg", "jpeg", "png"]
        )
        
        # Handle uploaded file
        if uploaded_file is not None:
            try:
                # Open image and store in session state
                st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    st.session_state.uploaded_image, 
                    caption="Uploaded Chest X-ray", 
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error opening the image: {str(e)}")
                st.session_state.pop('uploaded_image', None)
        
        # Show the uploaded image from session state if available
        elif 'uploaded_image' in st.session_state:
            st.image(
                st.session_state.uploaded_image, 
                caption="Uploaded Chest X-ray", 
                use_container_width=True
            )
        else:
            # Show placeholder if no image is uploaded
            st.info("Please upload a chest X-ray image or select a sample image.")
    
    # Process image button
    process_button = st.button(
        "Detect Pneumonia",
        type="primary",
        disabled=not ('uploaded_image' in st.session_state)
    )
    
    # When process button is clicked and image is available
    if process_button and 'uploaded_image' in st.session_state:
        # Get preprocessing options from session state
        preprocessing_options = st.session_state.get(
            'preprocessing_options', 
            {"clahe": False, "histogram_eq": False, "denoising": False}
        )
        
        with st.spinner("Processing image..."):
            try:
                # Create a copy of the image for preprocessing
                original_image = st.session_state.uploaded_image.copy()
                
                # Preprocess the image - real or demo
                if DEMO_MODE:
                    processed_image_array, applied_techniques = demo_preprocess_image(
                        original_image,
                        preprocessing_options
                    )
                    processed_image = Image.fromarray(processed_image_array)
                else:
                    processed_image_array, applied_techniques = preprocess_image(
                        original_image, 
                        preprocessing_options
                    )
                    processed_image = cv2_to_pil(processed_image_array)
                
                # Run inference - real or demo
                if DEMO_MODE:
                    prediction_results = demo_predict(processed_image)
                else:
                    prediction_results = predict_image(
                        model,
                        processed_image,
                        metadata["device"]
                    )
                
                # Generate Grad-CAM visualization - real or demo
                with st.spinner("Generating visualization..."):
                    if DEMO_MODE:
                        original_np, heatmap = demo_gradcam(processed_image)
                        heatmap_overlay = demo_overlay_gradcam(original_np, heatmap)
                    else:
                        original_np, heatmap = generate_gradcam(
                            model,
                            processed_image,
                            metadata["device"]
                        )
                        heatmap_overlay = overlay_gradcam(original_np, heatmap)
                
                # Store results in session state
                st.session_state.processed_image = processed_image
                st.session_state.prediction_results = prediction_results
                st.session_state.applied_techniques = applied_techniques
                st.session_state.heatmap_overlay = heatmap_overlay
                
                # Jump to visualization section
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.exception("Error in processing pipeline")
    
    # Show results if available in session state
    if ('processed_image' in st.session_state and 
        'prediction_results' in st.session_state and 
        'applied_techniques' in st.session_state):
        
        col1, col2 = st.columns(2)
        
        # Original vs processed image
        col1.markdown('<h3 class="sub-header">Original Image</h3>', unsafe_allow_html=True)
        col1.image(
            st.session_state.uploaded_image, 
            caption="Original Chest X-ray", 
            use_container_width=True
        )
        
        col2.markdown('<h3 class="sub-header">Processed Image</h3>', unsafe_allow_html=True)
        col2.image(
            st.session_state.processed_image, 
            caption=f"Processed X-ray", 
            use_container_width=True
        )
        
        # Applied techniques
        if st.session_state.applied_techniques:
            st.markdown("**Preprocessing Applied:**")
            st.markdown(", ".join(st.session_state.applied_techniques))
        else:
            st.markdown("**No preprocessing applied**")
        
        st.markdown("---")
        
        # Prediction results
        prediction = st.session_state.prediction_results["prediction"]
        probability = st.session_state.prediction_results["probability"]
        
        # Different styling based on prediction
        if prediction == "Normal":
            st.markdown(f'<div class="prediction-box-normal">', unsafe_allow_html=True)
            st.markdown(f"### Prediction: NORMAL", unsafe_allow_html=True)
            st.markdown('<p style="font-size: 18px; color: #198754; font-weight: bold;">No pneumonia detected</p>', unsafe_allow_html=True)
            st.progress(probability / 100)
            st.markdown(f"**Confidence: {probability:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)
        else:  # Pneumonia
            st.markdown(f'<div class="prediction-box-pneumonia">', unsafe_allow_html=True)
            st.markdown(f"### Prediction: PNEUMONIA DETECTED", unsafe_allow_html=True)
            st.markdown('<p style="font-size: 18px; color: #dc3545; font-weight: bold;">Pneumonia detected in the X-ray image</p>', unsafe_allow_html=True)
            st.progress(probability / 100)
            st.markdown(f"**Confidence: {probability:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Grad-CAM visualization
        if 'heatmap_overlay' in st.session_state:
            st.markdown('<h3 class="sub-header">Model Attention Map (Grad-CAM)</h3>', unsafe_allow_html=True)
            st.image(
                st.session_state.heatmap_overlay,
                caption="Areas the model focused on for making the prediction",
                use_container_width=True
            )
            
            with st.expander("About Grad-CAM"):
                st.markdown("""
                **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualizes which parts of the image 
                the model is focusing on to make its prediction. Warmer colors (red/yellow) indicate areas 
                that strongly influenced the model's decision.
                
                This helps in interpreting the model's decision and ensuring it's focusing on clinically 
                relevant features rather than artifacts or irrelevant areas of the X-ray.
                """)
    
    # Add footer with research paper citation
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1.5rem 0;">
        <p>Part of the research paper:<br/>
        <strong>"Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning"</strong></p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    """Load model and cache it using Streamlit's cache mechanism"""
    if DEMO_MODE:
        return "DEMO_MODEL", {"device": "cpu"}
    
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        # If model file is not found at the specified path, try to find it elsewhere
        for path in [
            "./pneumonia_resnet18.pt",
            "../pneumonia_resnet18.pt",
            "model.pth",
            "pneumonia_model.pth"
        ]:
            if os.path.exists(path):
                logger.info(f"Found model at {path}")
                return load_model(path)
        
        # If still not found, log error and return None
        logger.error("Model file not found!")
        return None, None

# Demo mode functions
def demo_preprocess_image(image, options):
    """Simulate preprocessing for demo mode"""
    # Just return the original image and list of techniques
    applied_techniques = []
    
    if options.get('clahe', False):
        applied_techniques.append("CLAHE (Demo)")
    if options.get('histogram_eq', False):
        applied_techniques.append("Histogram Equalization (Demo)")
    if options.get('denoising', False):
        applied_techniques.append("Denoising (Demo)")
    
    return np.array(image), applied_techniques

def demo_predict(image):
    """Simulate prediction for demo mode"""
    # Randomly decide if the image has pneumonia, slightly biased towards pneumonia
    is_pneumonia = random.random() > 0.4
    confidence = random.uniform(70, 98) if is_pneumonia else random.uniform(65, 95)
    
    return {
        "prediction": "Pneumonia" if is_pneumonia else "Normal",
        "probability": confidence,
        "class_idx": 1 if is_pneumonia else 0
    }

def demo_gradcam(image):
    """Create fake heatmap for demo mode"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create a fake heatmap with a central circular hotspot
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    
    # Random offset for the hotspot
    offset_x = random.uniform(-0.2, 0.2) * w
    offset_y = random.uniform(-0.2, 0.2) * h
    
    # Create circular heatmap
    mask = ((x - center_x - offset_x)**2 + (y - center_y - offset_y)**2) / (min(h, w)/2)**2
    heatmap = 1 - np.clip(mask, 0, 1)
    
    # Add some random noise
    heatmap += np.random.normal(0, 0.1, heatmap.shape)
    heatmap = np.clip(heatmap, 0, 1)
    
    return img_array, heatmap

def demo_overlay_gradcam(image, heatmap, alpha=0.4):
    """Create a fake heatmap overlay for demo mode"""
    # Convert to RGB if not already
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
        
    # Create a colormap for the heatmap (red is hot)
    cmap = plt.cm.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Convert to appropriate data type and scale
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend the images
    blended = image.copy().astype(np.float32)
    for c in range(3):
        blended[:, :, c] = image[:, :, c] * (1 - alpha * heatmap) + heatmap_colored[:, :, c] * (alpha * heatmap)
    
    return np.clip(blended, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    main() 