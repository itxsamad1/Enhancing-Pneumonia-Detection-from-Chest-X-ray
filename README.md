# 🫁 VGG-19 Pneumonia Detection (RP1 Baseline)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg?style=flat&logo=NVIDIA&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This branch (`VGG19-30epochs-patience-RP1`) forms the official VGG-19 baseline for Research Paper 1 (RP1). 

## 🎯 Architecture Configuration
- **Model**: VGG-19 (ImageNet fine-tuned)
- **Classifier Head**: Replaced standard classifier with `nn.Sequential(nn.Dropout(0.3), nn.Linear(4096, 2))`
- **Dataset**: Original Kaggle dataset (5,856 images). **No offline augmentation was applied** to preserve a clean baseline for direct architectural comparison.
- **Transforms**: Resize, CenterCrop, RandomHorizontalFlip(p=0.5), RandomRotation(15), ColorJitter, Normalization.

## ⚙️ Training Configuration
To ensure scientific consistency for the research paper, this model was trained with the exact same hyperparameters as all other architectures evaluated:
- **Max Epochs**: 30
- **Early Stopping**: Patience = 5 (monitoring validation loss)
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Hardware**: NVIDIA RTX 5060 Ti

## 📈 Performance & Results
Training actively prevented overfitting via Early Stopping and Dropout.

**Training Status:**
- Expected Epochs: 30
- **Actual Epochs Run**: 11 (Early stopping triggered automatically)
- **Best Performing Epoch**: **Epoch 6**

### Best Epoch Metrics (Epoch 6)
| Metric                  | Score     |
|--------------------------|-----------|
| **Validation Accuracy** | 90.87%    |
| **Validation Loss**     | 0.2112    |
| **F1 Score**            | 93.93%    |

> ✨ All 9 performance tracking graphs generated during this run are saved in the `graphs/` folder, and the final 90.87% accuracy model weights are stored in `pneumonia_vgg19.pt`.

---

## 📂 Dataset Pre-requisite
Due to size constraints and best practices, the actual X-ray images are not pushed to this repository. You must download the dataset separately before training or inference.
1. Download the **[Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** from Kaggle.
2. Extract and organize the data so that your local root folder structure looks like this:
```txt
.
├── dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── graphs/
├── train_model.py
├── app.py
├── pneumonia_resnet18.pt
└── ...
```

## 🚀 Usage 
To launch the interactive Grad-CAM visualization Streamlit UI on this specific model:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📜 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details. This means you are free to use, modify, and distribute the code, provided you give appropriate credit to the original author.
