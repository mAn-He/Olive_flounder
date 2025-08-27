# AI-Based Anomaly Detection & Segmentation in Aquaculture

> A project that overcomes the performance limitations of semantic segmentation models by resolving the ambiguity of overlapping object labels through a Data-Centric approach.

<br>

## 1. Problem
The performance of a semantic segmentation model for aquaculture fish stagnated at an F1 Score of 81.9% due to the ambiguity in data labels. Particularly in cases where objects overlap, the conventional binary labeling method made it difficult for the model to learn clear boundaries, thus limiting performance improvement.

<br>

## 2. My Solution
Instead of introducing a larger, more complex model, this project adopted a **Data-Centric approach**, focusing on the root cause of the problem: the data itself. By improving the quality and structure of the data, the model was enabled to learn from more explicit information.

#### Architecture & Key Features
-   **Redefined Labeling Strategy**: Without changing the model, we transitioned from a standard `'01' dataset` (`0`: Background, `1`: Object) to a new `'012' dataset`. This was achieved by redefining overlapping objects as a **'Third Class (2)'**, which resolved data ambiguity and enabled the model to learn more precise segmentation.
-   **Novel Evaluation Protocol**: A new evaluation protocol was designed to objectively validate the effectiveness of the new labeling method and to allow for a fair comparison against large-scale models like SAM.
-   **Flexible Experimentation Framework**: This repository provides a flexible framework for training and evaluating various models (DeepLabV3, DeepLabV3+, UNet) on both '01' (2-class) and '012' (3-class) dataset configurations.

![image](http://googleusercontent.com/file_content/11)
*Visual comparison of the original (left) vs. proposed (right) labeling and prediction results.*

<br>

## 3. Impact & Results
-   **Maximized Model Performance**: The Data-Centric approach alone boosted the **F1 Score by 7.8%p** (from 81.9% to 89.7%) under the conventional evaluation protocol.
-   **Superiority of the Approach**: Under our new, custom-designed evaluation protocol, our method outperformed the large-scale SAM model by a significant **13.5%p**.
-   **Cost-Efficient Problem Solving**: This project serves as a strong case study, proving that AI system performance limits can be overcome by improving data quality alone, without resorting to expensive model changes or additional computational resources.

| Evaluation Protocol | Model | F1 Score | IoU | Dice | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conventional (2-Class)** | Baseline (DeepLabV3) | 81.9% | - | - | - |
| | **Proposed (Trained on 3-Class)** | **89.7%** | **-** | **-** | **-** |
| **Proposed (3-Class)** | SAM (Large Model) | 84.9% | - | - | - |
| | **Proposed (DeepLabV3)** | **98.4%** | **-** | **-** | **-** |

*Note: IoU, Dice, and Accuracy metrics are included for completeness as they are common in segmentation tasks, but specific values were not detailed in the source documents. The core improvement is demonstrated by the F1 Score.*

<br>

## 4. How to Use

#### a. Setup
```bash
# 1. Clone the repository
git clone <repository_url>
cd <repository_directory>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies (Install PyTorch compatible with your CUDA setup)
pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn opencv-python click
