# Readme

## Overview
This repository provides training scripts for semantic segmentation models, including **DeepLabV3**, **DeepLabV3+**, and **UNet**. The code is structured to accommodate different dataset configurations and segmentation tasks. Key highlights of the repository include:

- Comparison of segmentation model performance on different dataset setups.
- Evaluation of traditional semantic segmentation and instance segmentation approaches.

## Dataset Setup

### Dataset Types
- **01 Dataset**:
  - `0`: Background
  - `1`: Target

- **012 Dataset**:
  - `0`: Background
  - `1`: Non-OOI (Out-of-Interest) Target
  - `2`: Target

## Tasks

### Task 1: Semantic Segmentation Model Comparison
Compare the performance of models trained on:

1. **01 Dataset**
2. **012 Dataset**

### Task 2: Advanced Model Comparisons
Evaluate and compare segmentation approaches between:

1. **012 Dataset-trained segmentation models**
2. **Instance segmentation models (e.g., Mask R-CNN)**
3. **Segment Anything Model (SAM)**

## Model References

### DeepLabV3
- Implementation and training code: [DeepLabV3 Fine-Tuning Repository](https://github.com/msminhas93/DeepLabv3FineTuning)

### DeepLabV3+
- Model details: [DeepLabV3+ in SMP Documentation](https://smp.readthedocs.io/en/v0.1.3/_modules/segmentation_models_pytorch/deeplabv3/model.html)
- 
- **`train/`**: Contains all training scripts for both tasks.



## How to Use

### Training a Semantic Segmentation Model
The base code is implemented for the **DeepLabV3** model. Additional scripts are available for:

- **UNet**
- **DeepLabV3+**

To train a model:

1. **Prepare the dataset** in either **01** or **012** format.
2. **Navigate** to the `train/` directory.
3. **Execute** the script for the desired model with your configuration.

### Instance Segmentation Comparisons
For Mask R-CNN or SAM comparisons:

- Install the required dependencies.
- Follow the instructions provided in the respective scripts.


