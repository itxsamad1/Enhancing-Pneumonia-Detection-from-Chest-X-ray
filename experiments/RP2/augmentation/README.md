# 🫁 Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Branch: `augmentation`** — This branch implements offline dataset augmentation with class balancing and an improved training pipeline with 8 anti-overfitting approaches. Designed as a clean template for experimenting with different model architectures.

---

## 🎯 What's New in This Branch

### Offline Dataset Augmentation (`augment_dataset.py`)
- Expands the dataset from **5,856 → 31,339 images** (5.4× expansion)
- **Class balancing**: Normal class gets 8× augmentation, Pneumonia gets 3× (ratio improved from 1:2.7 → 1:1.2)
- **8 medically valid augmentation techniques**:
  1. Horizontal Flip (lung symmetry)
  2. Random Rotation ±15° (patient positioning)
  3. Brightness Adjustment ±20% (exposure variation)
  4. CLAHE Contrast Enhancement (machine sensor variation)
  5. Gaussian Noise (quantum mottle / sensor noise)
  6. Image Sharpening (edge enhancement)
  7. Random Zoom (patient-to-plate distance)
  8. Combined Augmentations (2-3 mixed techniques)

### Anti-Overfitting Training Pipeline (`train_pneumonia.py`)
| Approach | Technique | Detail |
|:---|:---|:---|
| A | Offline Augmentation | Expanded dataset from `dataset_augmented/` |
| B | Real-time Augmentation | RandomErasing, RandomAffine, ColorJitter |
| C | Dropout | 0.3 rate in classifier head |
| D | Weight Decay (L2) | 1e-4 regularization |
| E | Early Stopping | Patience = 10 epochs |
| F | LR Scheduling | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| G | Gradient Clipping | max_norm = 1.0 |
| H | Label Smoothing | 0.1 (softens hard targets) |

### GPU Optimization
- Batch size: 64 (optimized for 16GB VRAM)
- Mixed precision training (`torch.amp`)
- `cudnn.benchmark` enabled
- Persistent workers + pin_memory
- Peak VRAM utilization tracking
- Per-epoch overfitting gap indicator (🟢🟡🔴)

---

## 📂 Dataset Statistics

### Original Dataset (5,856 images)
| Split | Normal | Pneumonia | Total |
|:---|:---|:---|:---|
| Training | 1,266 | 3,418 | 4,684 |
| Validation | 317 | 855 | 1,172 |
| **Total** | **1,583** | **4,273** | **5,856** |

### Augmented Dataset (31,339 images)
| Split | Normal | Pneumonia | Total |
|:---|:---|:---|:---|
| Training | 11,394 | 13,672 | 25,066 |
| Validation | 2,853 | 3,420 | 6,273 |
| **Total** | **14,247** | **17,092** | **31,339** |

> Class ratio improved from **1:2.7** to **1:1.2** after augmentation.

---

## 🚀 Quick Start Guide

### Step 1: Prepare the Original Dataset
```bash
# Download and prepare the raw Kaggle dataset
python prepare_dataset.py
```

### Step 2: Run Augmentation
```bash
# Expand the dataset with augmentation + class balancing
python augment_dataset.py
```

### Step 3: Train the Model
```bash
# Train ResNet-18 on the augmented dataset (30 epochs)
python train_pneumonia.py
```

### Step 4: Run the Web App
```bash
# Launch the Streamlit interface
streamlit run app.py
```

### Windows Users (Quick Launch)
```bash
.\run.bat
```

---

## 📁 Project Structure

```
.
├── augment_dataset.py            # Offline augmentation pipeline (NEW)
├── train_pneumonia.py            # Training pipeline (anti-overfitting)
├── evaluate_pneumonia.py         # Model evaluation (classification report, ROC)
├── predict_pneumonia.py          # Standalone inference script
├── prepare_dataset.py            # Dataset preparation & merging
├── download_datasets.py          # Automated dataset downloader (Kaggle API)
├── app.py                        # Streamlit web interface with Grad-CAM
├── run.bat                       # Windows launcher (venv + Streamlit)
├── requirements.txt              # Python dependencies
├── assets/
│   └── sample_images/            # Demo images for the web app
├── dataset/                      # Original dataset (not in repo)
└── dataset_augmented/            # Augmented dataset (not in repo)
```

> **Note**: Model weights (`.pt`), training graphs (`graphs/`), datasets, and research paper files are excluded from this branch to keep it clean as a template for architecture experiments.

---

## 📊 Training Configuration

| Parameter               | Value                            |
|:---|:---|
| **Model** | ResNet-18 (ImageNet pre-trained) |
| **Epochs** | 30 |
| **Batch Size** | 64 |
| **Initial Learning Rate** | 0.001 |
| **Optimizer** | Adam (weight decay = 1e-4) |
| **LR Scheduler** | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| **Early Stopping** | Patience = 10 |
| **Dropout** | 0.3 (classifier head) |
| **Label Smoothing** | 0.1 |
| **Gradient Clipping** | max_norm = 1.0 |
| **Mixed Precision** | torch.amp (CUDA) |
| **Loss Function** | CrossEntropyLoss |

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
- **Modified**: Dropout(0.3) → Linear(512, 2) classifier head
- **Optimizer**: Adam with weight decay (1e-4)
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Training**: 30 epochs with mixed precision (torch.amp)
- **Inference**: Grad-CAM visualization for interpretability

### Hardware
- Tested on NVIDIA RTX 5060 Ti (Blackwell architecture, 16GB VRAM)
- CUDA 12.8 with `sm_120` compute capability

---

## 🔀 Using This Branch as a Template

This branch is designed to be a **clean starting point** for training with different architectures. To try a new model:

1. Modify `create_model()` in `train_pneumonia.py`
2. Run `python augment_dataset.py` (if not already done)
3. Run `python train_pneumonia.py`
4. Results will be saved in `graphs/` and `pneumonia_resnet18.pt`

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
