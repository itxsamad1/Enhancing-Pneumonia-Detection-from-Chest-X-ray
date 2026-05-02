"""
Pneumonia Detection — Streamlit Web Application (Phase 2 RP2)
==============================================================
Self-contained multi-architecture inference app with Grad-CAM++ support.
Supports: ResNet-18/50, DenseNet-121, VGG-19, EfficientNet-B0,
          MobileNetV3, ConvNeXt-Tiny, DeiT-Tiny, Swin-Tiny, ViT-B/16.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import io
import logging
import shutil
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    vgg19, VGG19_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from gradcam import run_gradcam_pp

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
SAMPLE_DIR  = BASE_DIR / "assets" / "sample_images"
DATASET_DIR = BASE_DIR / "dataset"
DROPOUT     = 0.3

MODEL_REGISTRY = {
    "ResNet-18"       : ("pneumonia_resnet18.pt",        "resnet18"),
    "ResNet-50"       : ("pneumonia_resnet50.pt",        "resnet50"),
    "DenseNet-121"    : ("pneumonia_densenet121.pt",     "densenet121"),
    "VGG-19"          : ("pneumonia_vgg19.pt",           "vgg19"),
    "EfficientNet-B0" : ("pneumonia_efficientnet_b0.pt", "efficientnetb0"),
    "MobileNetV3"     : ("pneumonia_mobilenetv3.pt",     "mobilenetv3"),
    "ConvNeXt-Tiny"   : ("pneumonia_convnext_tiny.pt",   "convnext_tiny"),
    "DeiT-Tiny"       : ("pneumonia_deit_tiny.pt",       "deit_tiny"),
    "Swin-Tiny"       : ("pneumonia_swin_tiny.pt",       "swin_tiny"),
    "ViT-B/16"        : ("pneumonia_vit_b_16.pt",        "vit_b_16"),
}

# Available (weight file exists locally)
AVAILABLE_MODELS = {
    name: cfg for name, cfg in MODEL_REGISTRY.items()
    if (BASE_DIR / cfg[0]).exists()
}

# ─── Image Transform (same as training) ──────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Model Factory (mirrors train_model.py exactly) ──────────────────────────
def _build_model(model_key: str) -> nn.Module:
    key = model_key.lower()
    if key == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 2))
    elif key == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 2))
    elif key == "densenet121":
        m = densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier.in_features, 2))
    elif key == "vgg19":
        m = vgg19(weights=None)
        m.classifier[6] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[6].in_features, 2))
    elif key == "efficientnetb0":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[1].in_features, 2))
    elif key == "mobilenetv3":
        m = mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[3].in_features, 2))
    elif key == "convnext_tiny":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(m.classifier[2].in_features, 2))
    elif key in ("deit_tiny", "swin_tiny", "vit_b_16"):
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm not installed. Run: pip install timm")
        timm_name = {
            "deit_tiny": "deit_tiny_patch16_224",
            "swin_tiny": "swin_tiny_patch4_window7_224",
            "vit_b_16" : "vit_base_patch16_224",
        }[key]
        m = timm.create_model(timm_name, pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return m


@st.cache_resource(show_spinner=False)
def load_model_cached(display_name: str):
    weight_file, model_key = MODEL_REGISTRY[display_name]
    weight_path = BASE_DIR / weight_file

    if not weight_path.exists():
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_key)
    state = torch.load(str(weight_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ─── Preprocessing helpers ───────────────────────────────────────────────────
def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def apply_histeq(img_rgb: np.ndarray) -> np.ndarray:
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def apply_denoise(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)

def preprocess_image(pil_img: Image.Image, options: dict):
    img = np.array(pil_img.convert("RGB"))
    applied = []
    if options.get("clahe"):
        img = apply_clahe(img); applied.append("CLAHE")
    if options.get("histeq"):
        img = apply_histeq(img); applied.append("Histogram Equalization")
    if options.get("denoise"):
        img = apply_denoise(img); applied.append("Denoising")
    return Image.fromarray(img), applied


# ─── CSS ─────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #0077B6, #00B4D8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0 0.25rem;
    }
    .sub-title {
        text-align: center; color: #555; font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f8ff, #e8f4fc);
        border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #0077B6;
        margin: 0.4rem 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #28a745; margin: 0.8rem 0;
    }
    .result-pneumonia {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #dc3545; margin: 0.8rem 0;
    }
    .gradcam-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7F00FF, #E100FF);
        color: white; padding: 3px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0077B6, #00B4D8);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,119,182,0.4); }
    </style>
    """, unsafe_allow_html=True)


# ─── App ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Pneumonia Detection — RP2",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection — Phase 2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Multi-Architecture Deep Learning with Grad-CAM++ Explainability · RSNA + Kaggle Dataset · 10 Architectures</p>', unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Model Selection")

        if not AVAILABLE_MODELS:
            st.warning("No trained models found! Train a model first using `train_model.py`.")
            selected_model = None
        else:
            selected_model = st.selectbox(
                "Choose architecture",
                list(AVAILABLE_MODELS.keys()),
                help="Only models with trained weights (.pt files) appear here."
            )

        st.markdown("---")
        st.markdown("## 🔬 Grad-CAM++ Settings")
        gradcam_enabled = st.toggle("Enable Grad-CAM++ Heatmap", value=True)
        explain_class = st.radio(
            "Explain prediction for",
            ["Predicted class", "Pneumonia (class 1)", "Normal (class 0)"],
            help="Which class to generate the heatmap for."
        )
        target_class_map = {
            "Predicted class": None, "Pneumonia (class 1)": 1, "Normal (class 0)": 0
        }
        target_class = target_class_map[explain_class]

        st.markdown("---")
        st.markdown("## 🖼️ Preprocessing")
        clahe   = st.checkbox("CLAHE (Contrast Enhancement)", value=False)
        histeq  = st.checkbox("Histogram Equalization",       value=False)
        denoise = st.checkbox("Denoising",                    value=False)
        prep_opts = {"clahe": clahe, "histeq": histeq, "denoise": denoise}

        st.markdown("---")
        st.markdown("## 🔬 Sample Images")
        normal_sample    = DATASET_DIR / "val" / "NORMAL"
        pneumonia_sample = DATASET_DIR / "val" / "PNEUMONIA"

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Normal", use_container_width=True):
                if normal_sample.exists():
                    f = sorted(normal_sample.iterdir())[0]
                    st.session_state.sample_image = Image.open(f).convert("RGB")
                    st.session_state.pop("results", None)
                    st.rerun()
        with c2:
            if st.button("Load Pneumonia", use_container_width=True):
                if pneumonia_sample.exists():
                    f = sorted(pneumonia_sample.iterdir())[0]
                    st.session_state.sample_image = Image.open(f).convert("RGB")
                    st.session_state.pop("results", None)
                    st.rerun()

        st.markdown("---")
        st.caption("**Dataset**: Kaggle Chest X-Ray + RSNA Challenge\n\n**31,540 images** total")

    # ── Main Content ───────────────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Drag and drop a JPG / PNG X-ray", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.session_state.sample_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.pop("results", None)

        if "sample_image" in st.session_state:
            pil_img = st.session_state.sample_image
            st.image(pil_img, caption="Input X-Ray", use_container_width=True)

            # Show preprocessing preview if any option is ticked
            if any(prep_opts.values()):
                processed_pil, applied = preprocess_image(pil_img, prep_opts)
                st.caption(f"Preprocessing applied: {', '.join(applied)}")
            else:
                processed_pil, applied = pil_img, []

            # Detect button
            if selected_model:
                detect_btn = st.button("🔍 Detect Pneumonia", type="primary", use_container_width=True)
            else:
                st.info("Train a model first, then its `.pt` file will appear in the sidebar.")
                detect_btn = False
        else:
            st.info("Upload an X-ray image or load a sample from the sidebar.")
            detect_btn = False

    # ── Inference ─────────────────────────────────────────────────────────────
    if detect_btn and "sample_image" in st.session_state and selected_model:
        with st.spinner(f"Running inference with {selected_model}..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, device = load_model_cached(selected_model)

            if model is None:
                st.error(f"Could not load weights for {selected_model}.")
            else:
                pil_img_for_inf = processed_pil if any(prep_opts.values()) else st.session_state.sample_image
                tensor = val_transform(pil_img_for_inf).unsqueeze(0).to(device)

                _, model_key = MODEL_REGISTRY[selected_model]

                if gradcam_enabled:
                    with st.spinner("Generating Grad-CAM++ heatmap..."):
                        img_np = np.array(pil_img_for_inf.convert("RGB"))
                        overlay, cam, pred, probs = run_gradcam_pp(
                            model, model_key, tensor, img_np, target_class
                        )
                else:
                    with torch.no_grad():
                        out   = model(tensor)
                        probs = torch.softmax(out, dim=1)[0].tolist()
                        pred  = int(torch.argmax(out, dim=1).item())
                    overlay, cam = None, None

                st.session_state.results = {
                    "pred": pred, "probs": probs,
                    "overlay": overlay, "cam": cam,
                    "applied": applied,
                    "model_name": selected_model,
                }
                st.rerun()

    # ── Results Panel ─────────────────────────────────────────────────────────
    with col_result:
        st.markdown("### 📊 Analysis Results")

        if "results" not in st.session_state:
            st.markdown("""
            <div style="background:#f8f9fa; border-radius:12px; padding:2rem; text-align:center; color:#888; min-height:200px; display:flex; flex-direction:column; justify-content:center;">
                <p style="font-size:3rem; margin:0">🫁</p>
                <p style="margin:0.5rem 0 0">Upload an X-ray and click <strong>Detect</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            r = st.session_state.results
            pred  = r["pred"]
            probs = r["probs"]
            normal_conf    = probs[0] * 100
            pneumonia_conf = probs[1] * 100

            # Prediction Card
            if pred == 0:
                st.markdown(f"""
                <div class="result-normal">
                    <h3 style="margin:0; color:#155724">✅ Normal</h3>
                    <p style="margin:4px 0 0; color:#155724">No pneumonia detected — Confidence: <strong>{normal_conf:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-pneumonia">
                    <h3 style="margin:0; color:#721c24">⚠️ Pneumonia Detected</h3>
                    <p style="margin:4px 0 0; color:#721c24">Pneumonia indicators found — Confidence: <strong>{pneumonia_conf:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Class Probabilities**")
            st.markdown(f"Normal ({normal_conf:.1f}%)")
            st.progress(probs[0])
            st.markdown(f"Pneumonia ({pneumonia_conf:.1f}%)")
            st.progress(probs[1])

            # Model badge
            st.caption(f"Model: **{r['model_name']}** &nbsp;|&nbsp; Preprocessing: {', '.join(r['applied']) if r['applied'] else 'None'}")

            # Grad-CAM++
            if r.get("overlay") is not None:
                st.markdown("---")
                st.markdown('<span class="gradcam-badge">Grad-CAM++</span>', unsafe_allow_html=True)
                st.markdown("**Model Attention Heatmap**")
                st.image(r["overlay"], caption="Red/yellow = high attention (areas influencing the prediction)", use_container_width=True)

                with st.expander("📖 What is Grad-CAM++?"):
                    st.markdown("""
                    **Grad-CAM++** (Chattopadhyay et al., 2018) is an advanced improvement over Grad-CAM.
                    It uses *second-order* gradients to compute per-pixel importance weights, giving
                    **sharper** and **more precise** attention maps compared to standard Grad-CAM.

                    For pneumonia detection, this helps clinicians verify the model is correctly
                    focusing on **lung regions** (consolidation, opacity) rather than irrelevant
                    image artifacts like labels or patient positioning.

                    **Colors:**
                    - 🔴 **Red / Yellow** — High model attention (important for prediction)
                    - 🔵 **Blue / Green** — Low model attention
                    """)

    # ── Architecture Info ──────────────────────────────────────────────────────
    with st.expander("🔬 Architecture Overview — All 10 Models"):
        arch_data = {
            "Model":       ["ResNet-18", "ResNet-50", "DenseNet-121", "VGG-19", "EfficientNet-B0",
                            "MobileNetV3", "ConvNeXt-Tiny", "DeiT-Tiny", "Swin-Tiny", "ViT-B/16"],
            "Type":        ["CNN", "CNN", "CNN", "CNN", "CNN", "CNN", "Modern CNN",
                            "Transformer", "Transformer", "Transformer"],
            "Params (M)":  ["11.7", "25.6", "8.0", "143.7", "5.3",
                            "2.5", "28.6", "5.7", "28.3", "86.6"],
            "Status":      ["✅ Trained" if (BASE_DIR / MODEL_REGISTRY[n][0]).exists()
                           else "🕐 Pending" for n in
                           ["ResNet-18", "ResNet-50", "DenseNet-121", "VGG-19", "EfficientNet-B0",
                            "MobileNetV3", "ConvNeXt-Tiny", "DeiT-Tiny", "Swin-Tiny", "ViT-B/16"]],
        }
        import pandas as pd
        st.dataframe(arch_data, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#888; padding:1rem 0; font-size:0.85rem;">
        <strong>Enhancing Pneumonia Detection from Chest X-ray Images</strong> · Phase 2 Multi-Architecture Study<br/>
        Dataset: Kaggle Chest X-Ray + RSNA Pneumonia Challenge (31,540 images) · PyTorch · Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()