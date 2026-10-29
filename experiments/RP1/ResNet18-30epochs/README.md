# 🫁 Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository presents an end-to-end deep learning project for pneumonia detection using chest X-ray images. We leverage preprocessing techniques to enhance medical image quality, use a fine-tuned **ResNet-18** model (30 epochs) for binary classification, and integrate **Grad-CAM** for interpretability. A user-friendly **Streamlit** web app is provided for testing the model interactively.

<div align="center">
<img src="assets/sample_images/normal.jpg" width="300" alt="Normal X-Ray"/>
<img src="assets/sample_images/pneumonia.jpg" width="300" alt="Pneumonia X-Ray"/>
</div>

---

## 🎯 Key Features

- 🖼️ **Advanced Image Preprocessing**
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Histogram Equalization
  - Denoising (Non-Local Means)
  - Image Sharpening
- 🧠 **Deep Learning Model**
  - ResNet-18 architecture (ImageNet pre-trained)
  - Transfer learning with 30 epochs of fine-tuning
  - Mixed precision training (`torch.amp`) for GPU acceleration
  - Learning rate scheduling (ReduceLROnPlateau) + early stopping
- 🔍 **Visualization & Interpretability**
  - Grad-CAM heatmaps for model attention
  - Interactive Streamlit web interface
  - Real-time predictions with confidence scores
- 📊 **Comprehensive Training Analytics**
  - 9 performance graphs generated automatically
  - Per-epoch metrics: Accuracy, Loss, Precision, Recall, F1, Specificity, Learning Rate
  - Confusion matrix heatmap
  - Full training history saved as JSON

---

## 📖 Abstract

Pneumonia is a serious lung infection that must be diagnosed early for effective treatment. In this research, we explore how image preprocessing methods like contrast enhancement, denoising, and sharpening can improve deep learning model accuracy. Using transfer learning on ResNet-18 trained for **30 epochs** with mixed precision, and visualizing model attention through Grad-CAM, we achieve strong pneumonia detection performance on chest X-ray datasets. A lightweight web app interface demonstrates the practical utility of this system.

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
git clone https://github.com/itxsamad1/Enhancing-Pneumonia-Detection-from-Chest-X-ray.git

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
├── app.py                        # Streamlit web interface with Grad-CAM
├── train_pneumonia.py            # Training pipeline (30 epochs, mixed precision)
├── evaluate_pneumonia.py         # Model evaluation (classification report, ROC)
├── predict_pneumonia.py          # Standalone inference script
├── prepare_dataset.py            # Dataset preparation & merging
├── download_datasets.py          # Automated dataset downloader (Kaggle API)
├── run.bat                       # Windows launcher (venv + Streamlit)
├── requirements.txt              # Python dependencies
├── pneumonia_resnet18.pt         # Trained model weights (not in repo)
├── assets/
│   └── sample_images/            # Demo images for the web app
├── graphs/                       # Training performance graphs
│   ├── accuracy_plot.png         # Training & Validation Accuracy
│   ├── loss_plot.png             # Training & Validation Loss
│   ├── precision_plot.png        # Precision over Epochs
│   ├── recall_plot.png           # Recall over Epochs
│   ├── f1_score_plot.png         # F1 Score over Epochs
│   ├── specificity_plot.png      # Specificity over Epochs
│   ├── learning_rate_plot.png    # Learning Rate Schedule
│   ├── metrics_plot.png          # Combined Metrics
│   ├── confusion_matrix.png      # Confusion Matrix Heatmap
│   └── training_history.json     # Raw per-epoch metrics (JSON)
└── dataset/                      # Dataset folder (not included)
```

---

## 📊 Training Configuration

| Parameter               | Value                            |
|--------------------------|----------------------------------|
| **Model**               | ResNet-18 (ImageNet pre-trained) |
| **Epochs**              | 30                               |
| **Batch Size**          | 32                               |
| **Initial Learning Rate** | 0.001                         |
| **Optimizer**           | Adam (weight decay = 1e-4)       |
| **LR Scheduler**        | ReduceLROnPlateau (factor=0.5, patience=3) |
| **Early Stopping**      | Patience = 30                    |
| **Mixed Precision**     | torch.amp (CUDA)                 |
| **Loss Function**       | CrossEntropyLoss                 |
| **Data Augmentation**   | RandomResizedCrop, HorizontalFlip, Rotation(15°), ColorJitter |

---

## 📈 Results & Performance

### Best Epoch Metrics (Epoch 9 — Lowest Validation Loss)

| Metric                  | Value     |
|--------------------------|-----------|
| **Training Accuracy**   | 95.87%    |
| **Validation Accuracy** | 96.16%    |
| **Training Loss**       | 0.1174    |
| **Validation Loss**     | 0.0946    |
| **Precision**           | 96.98%    |
| **Recall**              | 97.78%    |
| **F1 Score**            | 97.38%    |
| **Specificity**         | 91.80%    |

### Final Epoch Metrics (Epoch 30)

| Metric                  | Value     |
|--------------------------|-----------|
| **Training Accuracy**   | 99.06%    |
| **Validation Accuracy** | 94.80%    |
| **Training Loss**       | 0.0240    |
| **Validation Loss**     | 0.1868    |
| **Precision**           | 93.53%    |
| **Recall**              | 99.77%    |
| **F1 Score**            | 96.55%    |
| **Specificity**         | 81.39%    |

### Training Graphs

The following performance graphs are generated and saved in the `graphs/` directory:

| # | Graph                        | Description                        |
|---|------------------------------|------------------------------------|
| 1 | `accuracy_plot.png`          | Training & Validation Accuracy     |
| 2 | `loss_plot.png`              | Training & Validation Loss         |
| 3 | `precision_plot.png`         | Precision over Epochs              |
| 4 | `recall_plot.png`            | Recall over Epochs                 |
| 5 | `f1_score_plot.png`          | F1 Score over Epochs               |
| 6 | `specificity_plot.png`       | Specificity over Epochs            |
| 7 | `learning_rate_plot.png`     | Learning Rate Schedule             |
| 8 | `metrics_plot.png`           | Combined Metrics                   |
| 9 | `confusion_matrix.png`       | Confusion Matrix Heatmap           |

### Key Findings

1. **CLAHE preprocessing** significantly improved model performance on low-contrast X-rays
2. **Image Sharpening** enhanced edge detection for identifying pneumonia patterns
3. **Grad-CAM visualization** confirmed the model focuses on medically relevant lung regions
4. Model generalizes well across different X-ray sources
5. Learning rate was reduced 5 times via scheduler (1e-3 → 3.125e-5)

---

## 🛠️ Technical Details

### Dependencies
- Python 3.12+
- PyTorch 2.7+ (CUDA 12.8)
- Streamlit ≥ 1.30
- OpenCV ≥ 4.8
- NumPy ≥ 1.26
- Matplotlib ≥ 3.8
- scikit-learn ≥ 1.3
- scikit-image ≥ 0.21
- seaborn ≥ 0.13
- tqdm ≥ 4.66

### Model Architecture
- **Base**: ResNet-18 (pre-trained on ImageNet)
- **Modified**: Final fully-connected layer → 2 output classes (Normal, Pneumonia)
- **Optimizer**: Adam with weight decay (1e-4)
- **Loss**: CrossEntropyLoss
- **Training**: 30 epochs with mixed precision (torch.amp)
- **Inference**: Grad-CAM visualization for interpretability

### Hardware
- Tested on NVIDIA RTX 5060 Ti (Blackwell architecture)
- CUDA 12.8 with `sm_120` compute capability

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
