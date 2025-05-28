# AI Brain Stroke Detection and Segmentation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)
![PyTorch](https://img.shields.io/badge/pytorch-1.x-red)

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture](#architecture)
* [Installation](#installation)
* [Usage](#usage)

  * [Data Preprocessing](#data-preprocessing)
  * [Classification](#classification)
  * [Segmentation](#segmentation)
  * [Inference](#inference)
* [Evaluation & Results](#evaluation--results)
* [Directory Structure](#directory-structure)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Overview

This repository implements a comprehensive pipeline for brain stroke detection and segmentation on medical imaging data. It leverages:

* **EfficientNet** for image classification (stroke vs. non-stroke).
* **U-Net** for precise stroke lesion segmentation.
* Modular preprocessing utilities for DICOM/CT image handling.

The goal is to provide an end-to-end solution covering data ingestion, preprocessing, model training, evaluation, and inference, suitable for research and clinical prototyping.

## Features

* 🚀 **High-performance classification** using EfficientNet architectures.
* 🎯 **Accurate segmentation** with a customizable U-Net model.
* 🛠️ **Reusable preprocessing**: windowing, normalization, augmentation.
* 📊 **Comprehensive evaluation**: ROC, IoU, Dice coefficient metrics.
* 📁 **Organized structure**: clear separation of classification, segmentation, and results.

## Architecture

```mermaid
graph LR
    A[Raw DICOM/CT Data] --> B[Preprocessing]
    B --> C[Classification (EfficientNet)]
    B --> D[Segmentation (U-Net)]
    C --> E[Classification Metrics]
    D --> F[Segmentation Metrics]
    E & F --> G[Results & Visualizations]
```


* **Preprocessing**: Automatic window-level adjustments, resizing to 224×224 for classification, 256×256 for segmentation.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ismaylovvm/AI-Brain-stroke-detection.git
   cd AI-Brain-stroke-detection
   ```
2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

Preprocess raw scans:

```bash
python image_processing/preprocess.py \
    --input_dir dataset/images \
    --output_dir processed/images \
    --mask_dir dataset/masks \
    --img_size 224
```

### Classification

Train EfficientNet classifier:

```bash
python classification/train_classifier.py \
    --data_dir processed/images \
    --labels dataset/labels.csv \
    --model efficientnet-b4 \
    --epochs 50 \
    --batch_size 16 \
    --output_dir classification/models
```

Evaluate classifier:

```bash
python classification/evaluate_classifier.py \
    --model_path classification/models/best_model.h5 \
    --test_data processed/test \
    --labels test_labels.csv \
    --metrics_out classification/results/classification_metrics.json
```

### Segmentation

Train U-Net segmentation model:

```bash
python segmentation/train_unet.py \
    --images processed/images \
    --masks processed/masks \
    --epochs 100 \
    --batch_size 8 \
    --output_dir segmentation/models
```

Evaluate segmentation:

```bash
python segmentation/evaluate_unet.py \
    --model_path segmentation/models/unet_best.pt \
    --test_images processed/test/images \
    --test_masks processed/test/masks \
    --metrics_out segmentation/results/segmentation_metrics.json
```

### Inference

Run end-to-end inference:

```bash
python inference/run_inference.py \
    --classifier_path classification/models/best_model.h5 \
    --segmenter_path segmentation/models/unet_best.pt \
    --input_dir new_scans/ \
    --output_dir predictions/ \
    --threshold 0.5
```

## Evaluation & Results

* **Classification**:

  * Accuracy: 94.2%
  * AUC-ROC: 0.97
* **Segmentation**:

  * Dice Coefficient: 0.88
  * IoU: 0.83
  * F1: 0.8

Results and visualizations are saved under the `results/` directory for further analysis.

## Directory Structure

```
AI-Brain-stroke-detection/
├── classification/      # Classification scripts and models
├── segmentation/        # Segmentation scripts and models
├── image_processing/    # Data loading and preprocessing utilities
├── results/             # Metrics and sample visualizations
├── inference/           # Inference pipeline
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Dependencies

Key libraries and frameworks:

* Python 3.7+
* TensorFlow 2.x
* PyTorch 1.x
* EfficientNet (via `keras-applications` or `timm`)
* OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib

See `requirements.txt` for the full list.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Open a Pull Request.

Please ensure that code is PEP8 compliant and includes appropriate tests.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or suggestions, please open an issue or reach out directly:

* **GitHub**: [ismaylovvm](https://github.com/ismaylovvm)
* **Email**: [ismaylovv.m@gmail.com](ismaylovv.m@gmail.com)


