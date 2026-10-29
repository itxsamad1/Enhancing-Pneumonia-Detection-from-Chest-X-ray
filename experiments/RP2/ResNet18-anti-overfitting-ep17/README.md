# 🫁 Enhancing Pneumonia Detection from Chest X-ray Images using Image Preprocessing and Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository presents an end-to-end deep learning approach for pneumonia detection using chest X-ray images. It focuses heavily on dataset balancing, medically-valid offline dataset augmentation, and an explicitly anti-overfitting ResNet-18 architecture.

A user-friendly **Streamlit** web app is provided for testing the model interactively with real-time interpretation mapping via Grad-CAM.

<div align="center">
<img src="assets/sample_images/normal.jpg" width="300" alt="Normal X-Ray"/>
<img src="assets/sample_images/pneumonia.jpg" width="300" alt="Pneumonia X-Ray"/>
</div>

---

## 🎯 Architectural Techniques & Flow

The goal of this project phase is **generalization and preventing overfitting** on complex medical imaging data. 

### 1. The Dataset Pipeline Flow
We implemented a robust two-stage augmentation pipeline to combat natural class imbalances (where pneumonia cases vastly outnumbered normal cases at 2.7:1).
*   **Offline Augmentation**: The dataset was physically expanded from 5,856 images to **31,339 images**. 
    *   To balance the dataset, the minority class (Normal) received **8x augmentation** while the majority class received **3x augmentation**, achieving a nearly perfect **1:1.2** balance.
*   **Real-time Transforms**: As the model trains, images receive additional "on-the-fly" `RandomErasing`, `RandomAffine` shifts, and `ColorJitter`. This creates near-infinite variance.
*   **Techniques applied**: Horizontal mirroring, ±15° rotation, bounding box zooming, Gaussian Noise (simulating poor sensor data), and CLAHE contrast adjustments.

### 2. Model Architecture
We fine-tuned a **ResNet-18** network with structural overrides designed to restrict overfitting:
*   **Modified Classifier Head**: We replaced the standard linear output layer with an `nn.Sequential` block containing aggressive **Dropout (30%)** to randomly zero out neurons, preventing the model from excessively memorizing pixel locations.
*   **Label Smoothing**: Softens the typical one-hot vector (1.0 vs 0.0) into a (0.9 vs 0.1) confidence metric. This prevents the model from assigning 100% confidence to noisy medical images.

### 3. Training Loop Regularization
*   **Gradient Clipping**: Locked gradient norms at `1.0` to prevent "exploding gradients" which can disrupt convergence.
*   **L2 Weight Decay**: Adds `1e-4` penalty to loss calculations for overly large weights.
*   **Cosine Annealing with Warm Restarts**: Slowly decreases the learning rate but periodically "bounces" it back up over 10 epochs. This helps the optimizer escape local minimums.
*   **Mixed Precision (AMP)**: Uses `torch.amp` to accelerate rendering on the RTX 5060 Ti GPU while massively cutting memory costs (Max GPU Mem: 0.94 GB / 15.9 GB utilized!).

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

---

## 📈 Performance & Results

The rigorous regularizations allowed us to hit peak detection efficiency extremely fast without memorizing noise. **Early Stopping triggered at Epoch 17** after recognizing peak performance was achieved at **Epoch 7**.

### Best Epoch Metrics (Epoch 7)

| Metric                  | Score     |
|--------------------------|-----------|
| **Validation Accuracy** | 97.91%    |
| **Validation Loss**     | 0.2347    |
| **Precision**           | 98.24%    |
| **Recall**              | 97.92%    |
| **F1 Score**            | 98.08%    |
| **Specificity**         | 97.90%    |

> 📌 **Overfitting Analysis**:
> At peak performance (Epoch 7), the gap between Train Accuracy (97.88%) and Validation Accuracy (97.91%) was effectively `-0.0003`. **No overfitting was detected**, proving the regularized ResNet-18 architecture successfully mapped to real-world generalization parameters.

> ✨ Our 9 training graphs, alongside the raw `training_history.json`, are stored in the `/graphs/` folder.

---

## 🚀 Quick Start Guide

### Windows Users
```bash
# Clone the repository
git clone https://github.com/itxsamad1/Enhancing-Pneumonia-Detection-from-Chest-X-ray.git

# Execute via Batch Launcher
.\run.bat
```

### Manual Setup (All Environments)
```bash
# 1. Provide an environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # UNIX

# 2. Dependency Installations
pip install -r requirements.txt

# 3. Fire up Streamlit Inference App
streamlit run app.py
```

---

## 📁 Project Structure

```
.
├── augment_dataset.py            # Offline augmentation & dataset balancing
├── train_pneumonia.py            # Training pipeline (anti-overfitting configs)
├── evaluate_pneumonia.py         # Model evaluation script
├── predict_pneumonia.py          # Standalone python inference
├── prepare_dataset.py            # Data fetching/organizing logic
├── download_datasets.py          # Kaggle API integrations
├── app.py                        # Streamlit web interface with interactive Grad-CAM
├── run.bat                       # Windows environment automator
├── pneumonia_resnet18.pt         # 97.91% Accuracy weights
├── assets/
│   └── sample_images/            # Demo images
├── graphs/                       # Auto-generated performance mappings
└── dataset_augmented/            # The localized 31k+ images directory
```

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

---

<div align="center">
Made with ❤️ using PyTorch and Streamlit
</div>
