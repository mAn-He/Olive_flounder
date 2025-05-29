# Semantic Segmentation Model Training and Evaluation

## Overview
This repository offers a collection of Python scripts designed to streamline the training and evaluation of various semantic segmentation models, such as DeepLabV3, DeepLabV3+, and UNet. The primary goal is to facilitate research, development, and benchmarking in semantic segmentation by providing a flexible framework for experimentation across different dataset configurations and model comparison tasks.

## Project Structure
- **`task1/`**: Scripts for comparing semantic segmentation models (e.g., DeepLabV3 vs. DeepLabV3+).
  - `model_paths.txt`: Configuration file storing paths to pre-trained model files. Used by evaluation scripts in `task1`.
- **`task2/`**: Scripts for more advanced model comparisons, potentially involving instance segmentation or other models.
- **`train/`**: Training scripts for different models and dataset types.
  - `deeplabv3/01/`: Training scripts for DeepLabV3 on '01' type datasets.
  - `deeplabv3/012/`: Training scripts for DeepLabV3 on '012' type datasets.
- **`utils/`**: Utility modules.
  - `metrics.py`: Common image segmentation metric calculation functions (Accuracy, Dice, IoU).

## Dataset Setup
The project uses two main types of dataset configurations:

-   **01 Dataset**:
    -   `0`: Background
    -   `1`: Target class
-   **012 Dataset**: (Often used for 3-class segmentation)
    -   `0`: Background
    -   `1`: Non-OOI (Out-of-Interest) Target / First target class
    -   `2`: Target / Second target class

Prepare your datasets with images and corresponding single-channel masks where pixel values represent class IDs.

## Setup and Dependencies

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    A `requirements.txt` file is not yet available. Key dependencies include:
    -   PyTorch (torch, torchvision)
    -   NumPy
    -   Pandas
    -   Matplotlib
    -   Scikit-learn
    -   OpenCV (cv2) - Often used for image processing.
    -   Click (for command-line interfaces in training scripts)
    Install them using pip:
    ```bash
    pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn opencv-python click
    ```
    *Ensure you install a PyTorch version compatible with your CUDA setup if using GPU.*

## How to Use

### 1. Training Models (in `train/` directory)

The primary training script is `train/deeplabv3/01/main.py` (and its counterpart in `train/deeplabv3/012/`). These use `click` for command-line configuration.

**Example: Training DeepLabV3 on an '01' dataset:**
```bash
python train/deeplabv3/01/main.py \
    --data-directory /path/to/your/01_dataset_type/Train \
    --exp_directory /path/to/save/trained_models_and_logs \
    --epochs 30 \
    --batch-size 4
```
-   `--data-directory`: Path to your training data (images and masks folders).
-   `--exp_directory`: Where trained model checkpoints (`.pt` files) and logs will be saved.
-   `--epochs`: Number of training epochs.
-   `--batch-size`: Training batch size.

Other training scripts like `train/unet_finetuning.py` or for DeepLabV3+ variants exist but may require code inspection for specific parameters.

### 2. Evaluating Models (in `task1/` directory)

Scripts in `task1/` are used to evaluate and compare pre-trained models.
The script `task1/deeplabplus01_vs_deeplab_new_metric.py` has been refactored to use command-line arguments for paths.

**Model Path Configuration:**
Create or verify `task1/model_paths.txt`. This file lists the paths to your pre-trained model files (`.pt`). Example format:
```
(01)
deeplab : '/path/to/your/deeplab_model_for_01_dataset.pt'
deeplabv3plus :'/path/to/your/deeplabplus_model_for_01_dataset.pt'

(012)
deeplab : '/path/to/your/deeplab_model_for_012_dataset.pt'
deeplabv3plus : '/path/to/your/deeplabplus_3mask_model_for_012_dataset.pt'
```

**Example: Running `task1/deeplabplus01_vs_deeplab_new_metric.py`:**
```bash
python task1/deeplabplus01_vs_deeplab_new_metric.py \
    --img_folder /path/to/full/evaluation/images \
    --mask_folder /path/to/full/evaluation/masks \
    --model_paths_file task1/model_paths.txt \
    --output_csv_file /path/to/output/results.csv \
    --output_image_folder /path/to/save/comparison_images/ \
    --model_set_key "012" \
    --device "cuda:0"
```
-   `--img_folder`: Path to evaluation images.
-   `--mask_folder`: Path to evaluation masks.
-   `--model_paths_file`: Path to the `model_paths.txt` file.
-   `--output_csv_file`: Path to save the CSV file containing performance metrics.
-   `--output_image_folder`: (Optional) Path to save visual comparison images.
-   `--model_set_key`: Which set of models to use from `model_paths.txt` (e.g., "01" or "012").
-   `--device`: PyTorch device (e.g., "cuda:0", "cpu").

**Note on other scripts:**
*Other evaluation scripts like `task1/deeplabplus_3mask_vs_deeplab_new_metric.py` and those in `task2/` currently have hardcoded paths. These were not refactored due to tool limitations during the process. You may need to modify them directly to set correct paths.*

**Output:**
Evaluation scripts typically generate:
-   A CSV file with detailed metrics per image/object (Accuracy, Dice, IoU, F1-score).
-   Optionally, saved image files showing visual comparisons of model segmentations against ground truth.

## Original Model References
-   DeepLabV3: [DeepLabV3 Fine-Tuning Repository by msminhas93](https://github.com/msminhas93/DeepLabv3FineTuning)
-   DeepLabV3+: [SMP Documentation](https://smp.readthedocs.io/en/v0.1.3/_modules/segmentation_models_pytorch/deeplabv3/model.html)

## TODO
-   Add comprehensive docstrings to all scripts and functions.
-   Complete path refactoring for all evaluation scripts.
-   Generate a `requirements.txt` file.
```
