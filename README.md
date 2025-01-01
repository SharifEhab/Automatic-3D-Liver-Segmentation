# Liver Segmentation Using Monai and PyTorch

This repository provides all the necessary code and resources to perform liver segmentation using the Monai framework and PyTorch. The project is structured to allow customization for segmenting other organs as well.

---
### Shoutout 🎉

A special shoutout to [Mohammed El Amine Mokhtari](https://github.com/amine0110) for his great tutorials on medical imaging and computer vision. His work has been an inspiration and a valuable resource for the community! [Youtube Tut](https://www.youtube.com/watch?v=AU4KlXKKnac&list=PLQCkKRar9trOubxZm_gfqWmggW4MgVeqq&index=5)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
  - [UNet](#unet)
  - [Adopt Optimizer](#adopt-optimizer)
- [Training and Logging](#training-and-logging)
  - [Mixed Precision Training](#mixed-precision-training)
  - [WandB Integration](#wandb-integration)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

---

## Project Overview

This project focuses on automatic 3D liver segmentation from medical imaging data using a UNet architecture. The implementation is powered by the Monai framework, an open-source PyTorch-based library for deep learning in healthcare imaging. 

Key features include:

- Support for mixed precision training.
- Advanced logging and experiment tracking with Weights and Biases (WandB).
- Incorporation of the Adopt optimizer and Adam optimizer for performance tuning.

---

## Dataset

The datasets used for this project are sourced from Kaggle:

1. [Liver Tumor Segmentation - Part 1](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)
2. [Liver Tumor Segmentation - Part 2](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2)

These datasets consist of liver tumor segmentation volumes stored in NIfTI format. For detailed information, please refer to the Kaggle links above.

---

## Data Preparation

The data preparation process is automated and outlined in the `Datapreparation.ipynb` notebook. Key steps include:

1. **Conversion of NIfTI files to DICOM format.**
2. **Grouping DICOM slices:** Groups of 74 slices are created for consistency.
3. **Reconversion to compressed NIfTI:** For training efficiency.
4. **Empty segmentation cleanup:** Removal of empty segmentation masks and their corresponding volumes.

Ensure you execute the notebook before running the training scripts to prepare the data in the correct format.

---

## Model Architecture

### UNet
The UNet architecture is utilized for segmentation, featuring:

- Encoder-decoder structure with skip connections.
- Convolutional layers to extract and learn spatial features.
- Compatibility with Monai for medical image processing.

For detailed insights into the UNet design, visit the [Monai documentation](https://monai.io/).

### Adopt Optimizer
This project implements the Adopt optimizer, which combines the strengths of:

- Adaptive gradient clipping for robust optimization.
- Momentum-based updates to enhance convergence.

Resources for learning about Adopt:
- [Adopt Paper](https://arxiv.org/abs/2411.02853)
- [Adopt GitHub Repository](https://github.com/iShohei220/adopt)

---

## Training and Logging

### Mixed Precision Training
Mixed precision training leverages both `float32` and `float16` data types to accelerate training while reducing memory usage. It is implemented using PyTorch’s `torch.cuda` package:

- Forward and backward passes are performed in `float16` where possible.
- Critical computations, such as loss scaling, are performed in `float32` to maintain numerical stability.

#### Why Mixed Precision?
- Speeds up training by 1.5–2x.
- Reduces GPU memory usage, enabling larger batch sizes.

#### Implementation in Code
```python
from torch.cuda import autocast, GradScaler

scaler = GradScaler()
with autocast(device=device):
    outputs = model(inputs)
    loss = loss_function(outputs.float(), label.float())
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### WandB Integration
Weights and Biases (WandB) is used for experiment tracking, visualization, and hyperparameter tuning. 

#### Setting Up WandB
1. Install WandB:
    ```bash
    pip install wandb
    ```
2. Login to WandB:
    ```bash
    wandb login
    ```
3. Initialize in your script:
    ```python
    import wandb
    wandb.init(project="Liver-Segmentation")
    ```

#### Features Logged
- Training and validation losses.
- Dice metrics.
- Checkpoints for model performance.

#### Example WandB Dashboard
Below is a sample WandB logging screenshot:

![image](https://github.com/user-attachments/assets/9b66725f-8f56-4b84-9075-b70d82ec9a74)


---

## Installation and Usage

### Prerequisites
Install the required packages:
```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/SharifEhab/Automatic-3D-Liver-Segmentation.git
cd Automatic-3D-Liver-Segmentation
```

### Run Data Preparation
Prepare the dataset by running the `Datapreparation.ipynb` notebook.

### Train the Model
Run the training script:
```bash
python train.py
```

---

## References

1. [Monai Documentation](https://monai.io/)
2. [Adopt Optimizer Paper](https://arxiv.org/abs/2411.02853)
3. [Adopt GitHub Repository](https://github.com/iShohei220/adopt)
4. [Liver Tumor Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)

---

