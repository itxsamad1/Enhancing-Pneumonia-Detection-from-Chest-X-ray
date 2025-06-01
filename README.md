# 🫁 Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository presents an end-to-end deep learning project for pneumonia detection using chest X-ray images. We leverage preprocessing techniques to enhance medical image quality, use a fine-tuned ResNet-18 model for classification, and integrate Grad-CAM for interpretability. A simple and user-friendly Streamlit web app is provided for testing the model interactively.

<div align="center">
<img src="assets/sample_images/normal.jpg" width="300" alt="Normal X-Ray"/>
<img src="assets/sample_images/pneumonia.jpg" width="300" alt="Pneumonia X-Ray"/>
</div>

---

## 🎯 Key Features

- 🖼️ **Advanced Image Preprocessing**
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Histogram Equalization
  - Denoising
- 🧠 **Deep Learning Model**
  - ResNet-18 architecture
  - Transfer learning
  - High accuracy on test set
- 🔍 **Visualization**
  - Grad-CAM heatmaps
  - Interactive web interface
  - Real-time predictions
- 📊 **Comprehensive Analysis**
  - Training metrics
  - Evaluation scripts
  - Performance analysis

---

## 📖 Abstract

Pneumonia is a serious lung infection that must be diagnosed early for effective treatment. In this research, we explore how image preprocessing methods like contrast enhancement and denoising can improve deep learning model accuracy. Using transfer learning on ResNet-18, and visualizing model attention through Grad-CAM, we improve pneumonia detection accuracy on chest X-ray datasets. A lightweight web app interface demonstrates the practical utility of this system.

---

## 📂 Dataset Overview

We utilized three comprehensive datasets to ensure robust model training and validation:

### 1. Chest X-Ray Images (Pneumonia) – Kaggle
- 📁 Training Set:
  - Normal: 1,341 images
  - Pneumonia: 3,875 images
- 📁 Validation & Test Sets:
  - Proportionally split

### 2. NIH ChestX-ray14
- 📁 Total: 112,000+ images
- 14 disease classes
- Used: Pneumonia cases

### 3. ChestXpert-v1.0-small
- 📁 Filtered subset
- Normal & Pneumonia cases
- High-quality scans

📦 **Dataset Organization**:
```
/dataset/
├── chest_xray/
│   ├── train/
│   ├── test/
│   └── val/
├── chestxray14/
└── chestxpert-v1.0-small/
```

> 📝 Note: Due to size constraints, datasets are not included in the repository. Please download and place them in a `/dataset/` folder.

---

## 🚀 Quick Start Guide

### Windows Users (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/itxsamad1/Enhancing-Pneumonia-Detection-from-Chest-X-ray-Images-using-Image-Preprocessing-and-Deep-Learning.git

# 2. Double-click run.bat
# OR
# Run from command line:
.\run.bat
```

### Manual Setup (All Platforms)
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows:
.\venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will be available at:
- 🌐 Local URL: http://localhost:8501
- 🔗 Network URL: http://[your-ip]:8501

---

## 📁 Project Structure
```
.
├── app.py                       # Streamlit web interface
├── run.bat                     # Windows launcher
├── requirements.txt            # Dependencies
├── train_pneumonia.py          # Training pipeline
├── evaluate_pneumonia.py       # Model evaluation
├── predict_pneumonia.py        # Inference script
├── assets/
│   └── sample_images/         # Demo images
└── dataset/                   # Dataset folder (not included)
```

---

## 📊 Results & Impact

### Model Performance
- ✅ Training Accuracy: 98.7%
- ✅ Validation Accuracy: 96.5%
- ✅ Test Set Accuracy: 95.8%

### Key Findings
1. CLAHE preprocessing significantly improved model performance
2. Grad-CAM visualization confirmed medically relevant features
3. Model generalizes well across different X-ray sources

---

## 🛠️ Technical Details

### Dependencies
- Python 3.8+
- PyTorch 1.9+
- Streamlit
- OpenCV
- NumPy
- Matplotlib

### Model Architecture
- Base: ResNet-18
- Modified final layer for binary classification
- Trained with Adam optimizer
- Cross-entropy loss function

---

## 📚 Citation

```bibtex
@article{samad2024enhancing,
    title={Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning},
    author={Abdul Samad},
    year={2024},
    institution={Research Project},
    note={Undergraduate Research}
}
```

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contact & Support

- 👨‍💻 Author: Abdul Samad
- 📧 GitHub: [@itxsamad1](https://github.com/itxsamad1)
- 💬 Issues: Use the GitHub Issues tab for bugs/questions
- 🌟 If this project helps you, please consider giving it a star!

---

<div align="center">
Made with ❤️ using PyTorch and Streamlit
</div>
