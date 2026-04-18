# Sports Image Classification — Fine-Tuning EfficientNet-B0

A PyTorch project that fine-tunes a pretrained **EfficientNet-B0** model to classify images across **100 sports categories**. The project implements a two-phase transfer learning strategy with systematic hyperparameter ablation to maximise accuracy.

| Item | Detail |
|---|---|
| **Base Model** | [EfficientNet-B0](https://arxiv.org/abs/1905.11946) (pretrained on ImageNet) |
| **Dataset** | [Sports Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification) (100 classes) |
| **Platform** | Kaggle — GPU |
| **Framework** | PyTorch / torchvision |
| **Task** | Multi-class image classification (100 sports) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Approach](#approach)
3. [Model Architecture](#model-architecture)
4. [Notebook Walkthrough](#notebook-walkthrough)
5. [Training Process](#training-process)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Requirements](#requirements)

---

## Project Overview

Classifying 100 different sports from images is a challenging fine-grained recognition task — many sports share visual similarities (e.g., different racquet sports, different martial arts, field sports with similar backgrounds). Training a CNN from scratch on a dataset of this size would be impractical.

**Transfer learning** solves this: we start with EfficientNet-B0, a model pretrained on ImageNet's 1.2M images, and adapt it to our sports dataset. The pretrained features (edges, textures, objects) transfer well to sports images, and we only need to train a new classification head on top.

### Why EfficientNet-B0?

- **Efficient**: Uses compound scaling to balance depth, width, and resolution — achieves strong accuracy with fewer parameters than ResNet-50.
- **Lightweight**: ~5.3M parameters (vs ~25M for ResNet-50), making it fast to fine-tune on a single GPU.
- **Strong baseline**: Achieves 77.1% top-1 accuracy on ImageNet despite its small size.
- **Modern architecture**: Uses MBConv blocks with squeeze-and-excitation, which are more parameter-efficient than standard convolutions.

---

## Approach

### Two-Phase Transfer Learning

The project uses a well-established two-phase fine-tuning strategy:

**Phase 1 — Head Training (base frozen):**
- Freeze all EfficientNet-B0 layers (convolutional backbone).
- Replace the original classifier with a custom 3-layer head.
- Train only the classifier head for 20 epochs.
- This allows the head to learn sport-specific features on top of the frozen ImageNet representations.

**Phase 2 — Full Fine-Tuning (all layers unfrozen):**
- Unfreeze the entire model.
- Use **differential learning rates**: lower LR for the base (to preserve pretrained features), 10× higher LR for the head (to continue adapting).
- Train for 20 more epochs.
- This allows the early convolutional layers to adapt slightly to sports-specific visual patterns.

> **Why two phases?** Training everything from the start can corrupt the pretrained weights before the randomly-initialised head has learned anything useful. Phase 1 "warms up" the head; Phase 2 jointly refines the entire network.

### Hyperparameter Ablation

Before full training, the notebook runs systematic **ablation studies** — sweeping one hyperparameter at a time while fixing others at default values:

| Parameter | Candidates Tested |
|---|---|
| Learning rate (head) | `1e-3`, `5e-4`, `1e-4` |
| Dropout | `0.2`, `0.3`, `0.4` |
| Weight decay | `1e-5`, `1e-4`, `1e-3` |
| Label smoothing | `0.0`, `0.1`, `0.2` |
| Learning rate (fine-tune) | `1e-5`, `5e-5`, `1e-4` |

Each ablation run trains for 5 epochs (short Phase 1 only) and records the best validation accuracy. The winning values are then used for the full two-phase training.

---

## Model Architecture

### Base: EfficientNet-B0

EfficientNet-B0 uses a stack of **MBConv** (Mobile Inverted Bottleneck) blocks with squeeze-and-excitation attention. The pretrained model provides rich visual features from its ImageNet training.

| Property | Value |
|---|---|
| Input size | **224×224×3** |
| Base parameters | **~4.0M** (frozen in Phase 1) |
| Feature output | **1280-dimensional** vector |

### Custom Classifier Head

The default EfficientNet-B0 classifier is a single `Linear(1280, 1000)` layer. This notebook replaces it with a **3-layer head** for improved expressiveness:

```
EfficientNet-B0 Features (1280)
    │
    ▼
┌──────────────────────────────────────────┐
│  Dropout(p=dropout)                      │
├──────────────────────────────────────────┤
│  Linear(1280 → 1024)                     │
│  BatchNorm1d(1024)                       │
│  ReLU                                    │
├──────────────────────────────────────────┤
│  Dropout(p=dropout × 0.5)               │
├──────────────────────────────────────────┤
│  Linear(1024 → 512)                      │
│  BatchNorm1d(512)                        │
│  ReLU                                    │
├──────────────────────────────────────────┤
│  Dropout(p=dropout × 0.5)               │
├──────────────────────────────────────────┤
│  Linear(512 → 100)                       │  → 100 class logits
└──────────────────────────────────────────┘
```

**Key design choices:**
- **BatchNorm1d** after each hidden layer — stabilises training and allows higher learning rates.
- **Graduated dropout** — higher dropout at the input (where features are most general), lower between hidden layers. This reduces overfitting without being overly aggressive.
- **Two hidden layers** (1024, 512) — more capacity than a single linear layer, enabling the head to learn complex decision boundaries for 100 classes.

---

## Notebook Walkthrough

### 1 — Title Cell

Notebook title and description.

### 2 — Imports & Device Setup

Imports PyTorch, torchvision, PIL, matplotlib, and sklearn. Confirms GPU availability.

### 3 — Hyperparameters & Ablation Config

Defines all hyperparameters and ablation candidate lists:

| Parameter | Default Value | Purpose |
|---|---|---|
| `IMG_SIZE` | **224** | Standard EfficientNet input size |
| `BATCH_SIZE` | **32** | |
| `NUM_CLASSES` | **100** | Number of sport categories |
| `EPOCHS_HEAD` | **20** | Phase 1 epochs |
| `EPOCHS_FINE` | **20** | Phase 2 epochs |
| `LR_HEAD` | **1e-3** | Default learning rate for head training |
| `LR_FINE` | **1e-5** | Default learning rate for fine-tuning |
| `WEIGHT_DECAY` | **1e-4** | L2 regularisation |
| `DROPOUT` | **0.3** | Classifier head dropout rate |
| `LABEL_SMOOTHING` | **0.1** | Softens hard labels to prevent overconfidence |

### 4 — Data Loading & Augmentation

**Training augmentation pipeline:**

| Transform | Purpose |
|---|---|
| `Resize(244, 244)` + `RandomCrop(224)` | Scale jittering — introduces spatial variation |
| `RandomHorizontalFlip(p=0.5)` | Left–right flip augmentation |
| `RandomRotation(15°)` | Slight rotation for pose variation |
| `ColorJitter(brightness, contrast, saturation)` | Colour variation to improve robustness |
| `RandomGrayscale(p=0.02)` | Occasional greyscale — forces reliance on shape, not colour |
| `Normalize(ImageNet mean/std)` | Required for pretrained model compatibility |

**Validation/Test transforms:** `Resize(224)` + `ToTensor` + `Normalize` only (no augmentation).

Data is loaded using `ImageFolder`, which reads class subfolders automatically.

### 5 — Sample Visualisation

Displays a grid of augmented training images with class labels to verify the data pipeline.

### 6 — Base Model Loading

Loads EfficientNet-B0 with pretrained ImageNet weights (`IMAGENET1K_V1`). Freezes all base parameters for Phase 1.

### 7 — Classifier Head Definition

`build_classifier_head()` function creates the custom 3-layer head (see [Model Architecture](#custom-classifier-head)). Replaces the default classifier and prints parameter counts.

### 8 — Training & Evaluation Functions

- `train_one_epoch()` — standard training loop with loss accumulation and accuracy tracking.
- `evaluate()` — validation/test loop with `torch.no_grad()`.
- `train_model()` — wraps both functions with `ReduceLROnPlateau` scheduling and best-model checkpointing.

### 9 — Hyperparameter Ablation

Runs systematic sweeps:
1. For each hyperparameter, tests all candidate values with 5-epoch Phase 1 runs.
2. Selects the best value for each parameter.
3. Stores the optimal configuration for full training.

### 10 — Phase 1: Head Training

Rebuilds the model with the best ablation hyperparameters. Trains the classifier head for 20 epochs with the base frozen. Uses `ReduceLROnPlateau` (factor=0.3, patience=2).

### 11 — Ablation Results Visualisation

Bar charts showing validation accuracy for each hyperparameter sweep, with the best value highlighted.

### 12 — Phase 2: Full Fine-Tuning

1. Unfreezes all parameters.
2. Runs a quick LR sweep for the fine-tuning learning rate.
3. Creates optimizer with **differential learning rates**:
   - Base features: `LR_FINE`
   - Classifier head: `LR_FINE × 10`
4. Trains for 20 more epochs with `ReduceLROnPlateau` (factor=0.3, patience=3).

### 13 — Test Evaluation

Loads the best checkpoint and evaluates on the held-out test set. Prints:
- Overall test accuracy
- Per-class precision, recall, and F1-score via `classification_report`

### 14 — Training History Visualisation

Plots combined Phase 1 + Phase 2 curves:
- Training and validation **accuracy** over all epochs
- Training and validation **loss** over all epochs
- Vertical line marking the Phase 1 → Phase 2 transition

### 15 — Model Checkpoint Save

Saves a comprehensive checkpoint including:
- Model state dict
- Class-to-index mapping
- Index-to-class mapping
- Number of classes and image size

Also exports class names as a JSON file for deployment.

### 16 — Interactive Inference Widget

An `ipywidgets`-based upload widget that:
1. Accepts an image file upload.
2. Runs inference through the trained model.
3. Displays the image with top-5 predicted classes and confidence scores.

---

## Training Process

### Phase 1: Head Training

| Aspect | Detail |
|---|---|
| **Frozen layers** | All EfficientNet-B0 convolutional layers |
| **Trainable layers** | Custom 3-layer classifier head only |
| **Optimizer** | Adam (lr from ablation, weight decay from ablation) |
| **Scheduler** | `ReduceLROnPlateau` (factor=0.3, patience=2) |
| **Loss** | `CrossEntropyLoss` with label smoothing |
| **Epochs** | 20 |
| **Best model** | Saved based on validation accuracy |

### Phase 2: Full Fine-Tuning

| Aspect | Detail |
|---|---|
| **Frozen layers** | None — all layers trainable |
| **Differential LR** | Base: `LR_FINE`, Head: `LR_FINE × 10` |
| **Optimizer** | Adam with parameter groups |
| **Scheduler** | `ReduceLROnPlateau` (factor=0.3, patience=3) |
| **Loss** | `CrossEntropyLoss` with label smoothing |
| **Epochs** | 20 |
| **Best model** | Saved based on validation accuracy |

### Regularisation Techniques

| Technique | Purpose |
|---|---|
| **Label smoothing** (0.1) | Prevents overconfident predictions, improves generalisation |
| **Dropout** (graduated) | Reduces overfitting in the classifier head |
| **Weight decay** (L2) | Penalises large weights to prevent overfitting |
| **Data augmentation** | Artificial data diversity — flips, rotations, colour jitter |
| **ReduceLROnPlateau** | Automatically reduces LR when validation loss stalls |
| **Early stopping (implicit)** | Best model checkpoint prevents overfitting beyond the sweet spot |

---

## Results

### Test Set Performance

The model is evaluated on a held-out test set using the best checkpoint (selected by validation accuracy). Results include:
- **Overall test accuracy** across all 100 sports classes
- **Per-class metrics** — precision, recall, and F1-score for each sport

### Training Curves

The combined training history shows:
- **Phase 1** (epochs 1–20): Rapid accuracy improvement as the head learns to map frozen features to sport classes.
- **Phase 2** (epochs 21–40): Further improvement as the full network adapts to sport-specific visual patterns. The gain is typically smaller but meaningful.

### Why This Approach Works

1. **Transfer learning** leverages ImageNet features — edges, textures, object parts — that are highly relevant for sports image recognition (athletes, equipment, venues).

2. **Two-phase training** protects pretrained weights during initial head training, then allows gradual adaptation.

3. **Differential learning rates** prevent catastrophic forgetting: early layers (which capture general visual features) change slowly, while the head (which is task-specific) adapts quickly.

4. **Hyperparameter ablation** ensures the training configuration is well-suited to this specific dataset, rather than relying on generic defaults.

5. **Strong augmentation** (flips, rotations, colour jitter, occasional greyscale) exposes the model to diverse variations, improving robustness to real-world conditions (different lighting, angles, zoom levels).

---

## How to Run

1. **Platform:** Upload the notebook to [Kaggle](https://www.kaggle.com/) and attach the [Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification).
2. **GPU:** Enable a GPU accelerator in the Kaggle notebook settings.
3. **Execute cells in order.** The ablation phase takes some time; full training runs for 40 epochs total (20 head + 20 fine-tune).
4. **Inference:** Use the upload widget in the final cell to test the model on your own images.

---

## Requirements

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition, pretrained weights, transforms, DataLoader |
| `PIL` (Pillow) | Image loading and processing |
| `numpy`, `pandas` | Numerical operations, data handling |
| `matplotlib` | Visualisation (training curves, sample grids) |
| `scikit-learn` (`sklearn`) | Classification report (precision, recall, F1) |
| `ipywidgets` | Interactive image upload widget for inference |
| `os`, `json`, `time`, `copy` | File handling, timing, model copying |

All dependencies are pre-installed in the default Kaggle Python 3 Docker image.

---

## License

This is a personal project for learning and exploration. Feel free to use and adapt with attribution.
