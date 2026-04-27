# Alopecia Detection using Deep Learning

A binary image classification project that detects **alopecia** (hair loss) from scalp images using convolutional neural networks and transfer learning.

---

## Overview

This capstone project builds and compares multiple deep learning models to classify scalp images as either **alopecia** or **normal**. The pipeline covers everything from raw image collection and patch extraction to model training, fine-tuning, threshold optimization, and single-image inference.

---

## Project Structure

```
Alopecia_Project/
├── outputs/
│   ├── models/          # Saved .keras model files
│   ├── figures/         # ROC curve, PR curve plots
│   └── metrics/         # CSV/JSON evaluation results
└── processed_patches/   # Extracted image patches (train/val/test)
```

---

## Dataset

- **Classes:** `alopecia` | `normal`
- **Source:** Images stored in Google Drive under `Capstone Project/Alopecia` and `Capstone Project/Normal`
- **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`

Images are validated on load and catalogued into a dataset inventory CSV.

---

## Pipeline

### 1. Data Preprocessing
- Image validation and metadata collection (dimensions, extension, label)
- Train / Validation / Test split
- Patch extraction with a **224×224** sliding window and **stride of 112**
- Quality filtering per patch: brightness (35–225), contrast (std ≥ 18), sharpness (Laplacian variance ≥ 25)

### 2. Data Augmentation
- Random horizontal flip
- Random rotation (±5–8°)
- Random zoom (5–10%)
- Random contrast adjustment

### 3. Models Trained

| Model | Strategy |
|---|---|
| Baseline CNN | Custom 3-block Conv2D from scratch |
| MobileNetV2 | Frozen feature extraction → fine-tune last 30 layers |
| EfficientNetB0 | Frozen feature extraction → fine-tune last 40 layers |
| ResNet50 | Frozen feature extraction → fine-tune last 30 layers |

All transfer learning models use **ImageNet** pretrained weights.

### 4. Training Details
- **Input size:** 224×224×3
- **Batch size:** 32
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Callbacks:** `ReduceLROnPlateau`, `ModelCheckpoint`
- **Epochs:** 20 (frozen) + 15 (fine-tuning)

### 5. Evaluation
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix
- ROC curve & Precision-Recall curve
- **Threshold optimization** sweep (0.30–0.80) to maximize F1
- Misclassification analysis saved to CSV

### 6. Inference
Supports single-image upload and prediction with confidence score display.

---

## Requirements

```bash
pip install tensorflow opencv-python pillow seaborn scikit-learn tqdm
```

> **Note:** This notebook is designed to run on **Google Colab** with GPU acceleration and a Google Drive mount.

---

## Getting Started

1. Upload your dataset to Google Drive under:
   - `MyDrive/Capstone Project/Alopecia/`
   - `MyDrive/Capstone Project/Normal/`

2. Open `Capstone_project.ipynb` in Google Colab.

3. Mount Google Drive when prompted.

4. Run all cells sequentially.

5. The best model (by validation F1) is automatically saved to `outputs/models/best_alopecia_model.keras`.

---

## Outputs

| File | Description |
|---|---|
| `metrics/dataset_inventory.csv` | Full image metadata |
| `metrics/patch_metadata_filtered.csv` | Patch-level records after quality filtering |
| `metrics/model_comparison_validation.csv` | Side-by-side model metrics |
| `metrics/final_test_metrics.json` | Best model test set results |
| `metrics/misclassified_examples.csv` | Misclassified test images |
| `figures/roc_curve.png` | ROC curve (300 dpi) |
| `figures/precision_recall_curve.png` | PR curve (300 dpi) |
| `models/best_alopecia_model.keras` | Best saved model |

---

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Pretrained Models:** MobileNetV2, EfficientNetB0, ResNet50
- **Image Processing:** OpenCV, Pillow
- **Data & Visualization:** NumPy, Pandas, Matplotlib, Seaborn
- **Evaluation:** scikit-learn
- **Environment:** Google Colab (GPU)

---

## Reproducibility

A global seed of **42** is set across Python, NumPy, and TensorFlow for reproducible results.
