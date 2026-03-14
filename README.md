# Image_based_vegetable_identification

Here is a comprehensive `README.md` file tailored to your custom ResNet-18 vegetable classification script.

---

# Custom ResNet-18 Vegetable Image Classification

This repository contains a complete pipeline for training, evaluating, and exporting a custom **ResNet-18** model built entirely from scratch in PyTorch. The model is designed to classify images of various vegetables using the Kaggle Vegetable Image Dataset.

## Features

* **Custom ResNet-18 Architecture:** A PyTorch implementation of the ResNet-18 model, including custom `ResidualBlock` and weight initialization, without relying on pre-trained `torchvision` models.
* **Automatic Dataset Handling:** Uses `kagglehub` to automatically download and route the dataset.
* **Advanced Data Augmentation:** Integrates the `albumentations` library for high-performance image transformations (Random Resized Cropping, Horizontal Flipping, Normalization).
* **Training & Logging:** Includes a robust training loop with TensorBoard integration, label smoothing, AdamW optimizer, and StepLR learning rate scheduling.
* **Comprehensive Evaluation:** Calculates Accuracy, weighted F1 Score, ROC-AUC, and generates a Confusion Matrix.
* **ONNX Export:** Automatically exports the trained PyTorch model (`.pth`) to the ONNX format (`.onnx`) for production deployment and cross-platform compatibility.

## Prerequisites

Ensure you have Python installed along with the following dependencies. You can install them via pip:

```bash
pip install torch torchvision albumentations scikit-learn matplotlib pillow numpy kagglehub tqdm tensorboard

```

## Dataset

The script automatically downloads the **[Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)** by Misrak Ahmed directly from Kaggle using `kagglehub`.

The dataset is expected to have the following structure:

* `Vegetable Images/train/` (Training images categorized by folders)
* `Vegetable Images/validation/` (Validation images categorized by folders)
* `Vegetable Images/test/` (Testing images - *Note: The script currently evaluates on the validation set*)

## 🛠️ Usage

### 1. Run the Pipeline

Simply execute the Python script. It will sequentially:

1. Download/locate the dataset.
2. Initialize the custom ResNet-18 model.
3. Train the model for 30 epochs (saving the best weights to `best_resnet18.pth`).
4. Evaluate the model on the validation set.
5. Display visual predictions for 5 random validation images.


## ⚙️ Model Architecture Details

* **Initial Layer:** 7x7 Convolution, Stride 2, MaxPool.
* **Residual Layers:** 4 main layers containing 2 `ResidualBlock` instances each, with channel dimensions progressively increasing (64 -> 128 -> 256 -> 512).
* **Output Layer:** Adaptive Average Pooling followed by a Fully Connected Linear layer mapping to the dynamic number of dataset classes.
* **Weight Initialization:** Kaiming Normal initialization for Conv2d layers; Constant initialization (1 for weights, 0 for biases) for BatchNorm2d layers.


