README

Overview

This repository provides training scripts for semantic segmentation models, including DeepLabV3, DeepLabV3+, and UNet. The code is structured to accommodate different dataset configurations and segmentation tasks. Key highlights of the repository include the comparison of segmentation model performance on various dataset setups and between traditional semantic segmentation and instance segmentation approaches.

Dataset Setup

Dataset Types

01 Dataset:

Label 0: Background

Label 1: Target

012 Dataset:

Label 0: Background

Label 1: Non-OOI (Out-of-Interest) Target

Label 2: Target

Tasks

Task 1

Objective:
Compare the performance of semantic segmentation models trained using:

01 Dataset

012 Dataset

Task 2

Objective:
Evaluate and compare segmentation approaches between:

012 Dataset-trained segmentation models

Instance segmentation models (e.g., Mask R-CNN)

Segment Anything Model (SAM)

Model References

DeepLabV3

The implementation and training code for DeepLabV3 can be found here:
DeepLabV3 Fine-Tuning Repository

DeepLabV3+

For details on the DeepLabV3+ model implementation, refer to the SMP documentation:
DeepLabV3+ in SMP Documentation

Repository Structure

train/

Contains training scripts for DeepLabV3, DeepLabV3+, and UNet models.

Scripts are organized to handle 01 Dataset and 012 Dataset configurations for flexible experimentation.

How to Use

Training a Semantic Segmentation Model

Ensure the dataset is prepared in the desired format (01 or 012).

Navigate to the train/ directory.

Select the desired model script (e.g., deeplabv3, deeplabv3_plus, or unet).

Run the script using your configuration.

Instance Segmentation Comparison

For comparisons involving Mask R-CNN or SAM, ensure that the required dependencies are installed, and follow the instructions in the respective scripts for setup and execution.
