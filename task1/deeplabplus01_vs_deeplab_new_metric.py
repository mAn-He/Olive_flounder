"""
Compares the performance of a baseline DeepLabV3 model against a DeepLabV3+ model
on a semantic segmentation task for specific objects within images.

Key functionalities:
- Loads two pre-trained segmentation models (DeepLabV3 and DeepLabV3+).
- Processes a dataset of images and corresponding ground truth masks.
- For each image, identifies objects based on masks and crops them.
- Performs inference with both models on these cropped object regions.
- Calculates various performance metrics (Accuracy, Dice, IoU, F1-score) by comparing
  model predictions against the ground truth.
- Saves the aggregated metrics to a CSV file.
- Optionally saves visual comparison images (original crop, ground truth, model outputs).

This script is configurable via command-line arguments for specifying data folders,
model paths (through a configuration file), output locations, and processing device.
To run:
python task1/deeplabplus01_vs_deeplab_new_metric.py --img_folder <path_to_images> \
    --mask_folder <path_to_masks> --model_paths_file <path_to_model_config> \
    --output_csv_file <path_for_output_csv> --output_image_folder <path_for_images_if_saving> \
    --model_set_key <key_for_model_set> --device <cuda_device_or_cpu>
"""
import os
import time
import warnings
import argparse
import configparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision # Not used directly, but models might.
from PIL import Image
from sklearn.metrics import f1_score # roc_auc_score, confusion_matrix not directly used
# from torch.nn import Softmax # Softmax not used explicitly
# from torchmetrics.classification import Dice # Not used
from torchmetrics.classification import BinaryAccuracy
# from torchmetrics import JaccardIndex # Not used
from torchvision import transforms
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used

from utils import metrics # Import custom metrics

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3 and DeepLabV3+ models.")
    parser.add_argument('--img_folder', type=str, default='./Full/img',
                        help='Path to the main image folder.')
    parser.add_argument('--mask_folder', type=str, default='./Full/mask',
                        help='Path to the main mask folder.')
    parser.add_argument('--model_paths_file', type=str, default='task1/model_paths.txt',
                        help='Path to the text file containing model paths.')
    parser.add_argument('--output_csv_file', type=str, 
                        default='/home/fisher/Peoples/hseung/NEW/deeplabv3plus_output_metric/deeplabplus_vs_deeplab_metric_new.csv',
                        help='Path for saving the output CSV metrics file.')
    parser.add_argument('--output_image_folder', type=str, 
                        default='/home/fisher/Peoples/hseung/NEW/segment outputs/deeplabv3plus_30_01/',
                        help='Path to save output images (if enabled).')
    parser.add_argument('--model_set_key', type=str, default='012', choices=['01', '012'], 
                        help="Model set key to use from the paths file (e.g., '01' or '012').")
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Device to load models on (e.g., "cuda:0", "cpu").')
    return parser.parse_args()

def load_model_paths_from_config(file_path: str) -> dict:
    """
    Parses a custom format model paths file.
    The file is expected to have sections like (section_name) followed by
    model_key : 'path_to_model' pairs.

    Args:
        file_path (str): The path to the model paths configuration file.

    Returns:
        dict: A dictionary where keys are section names and values are
              dictionaries of model_key: model_path pairs.
              Returns None if the file is not found or an error occurs during parsing.
    """
    paths_manual = {}
    current_section = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    current_section = line[1:-1]
                    paths_manual[current_section] = {}
                elif current_section and ':' in line:
                    model_name_key, model_path_val = line.split(':', 1)
                    paths_manual[current_section][model_name_key.strip()] = model_path_val.strip().strip("'").strip('"')
    except FileNotFoundError:
        print(f"Error: Model paths file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error parsing model paths file '{file_path}': {e}")
        return None
    return paths_manual

def main():
    """
    Main workflow for model evaluation.
    - Parses arguments.
    - Loads model paths.
    - Loads models and sets them to evaluation mode on the specified device.
    - Iterates through test images and their masks.
    - For each object in an image:
        - Crops the object region from the image and ground truth mask.
        - Resizes cropped regions for model input.
        - Performs inference using both models.
        - Calculates segmentation metrics.
        - Optionally saves a visual comparison of the cropped image, ground truth, and predictions.
    - Saves aggregated metrics to a CSV file.
    """
    args = parse_arguments()
    
    start_time = time.time()

    all_model_paths = load_model_paths_from_config(args.model_paths_file)
    if all_model_paths is None:
        return 

    model_section = all_model_paths.get(args.model_set_key) 
    if not model_section:
        print(f"Error: Model set key '{args.model_set_key}' not found in {args.model_paths_file}")
        print(f"Available sections: {list(all_model_paths.keys())}")
        return

    deeplab_path = model_section.get('deeplab')
    deeplabv3plus_path = model_section.get('deeplabv3plus')

    if not deeplab_path:
        print(f"Error: 'deeplab' path not found in section '{args.model_set_key}' of {args.model_paths_file}")
        return
    if not deeplabv3plus_path: 
        print(f"Error: 'deeplabv3plus' path not found in section '{args.model_set_key}' of {args.model_paths_file}")
        return
        
    device_to_load = torch.device(args.device)
    
    try:
        deeplab_model = torch.load(deeplab_path, map_location=device_to_load) # Renamed for clarity
        deeplabv3plus_model = torch.load(deeplabv3plus_path, map_location=device_to_load) # Renamed for clarity
    except FileNotFoundError as e:
        print(f"Error loading model: {e}. Please check paths in {args.model_paths_file} for section '{args.model_set_key}'.")
        return
    except Exception as e: 
        print(f"Error loading model weights: {e}")
        return

    deeplabv3plus_model.eval()
    deeplab_model.eval()
    
    # Ensure models are on the target device
    deeplab_model.to(device_to_load)
    deeplabv3plus_model.to(device_to_load)

    try:
        all_img_files = [f for f in os.listdir(args.img_folder)]
        # Filter for specific date-coded image files
        img_filenames = sorted([f for f in all_img_files if ('20220817' in f) or ('20220819' in f)])
    except FileNotFoundError:
        print(f"Error: Image folder '{args.img_folder}' not found.")
        return
    
    test_file_list = img_filenames # Using a more descriptive name

    metric_df = pd.DataFrame(columns=['File Name','Deep Lab Accuarcy', 'Deep Plus Accuarcy', 'Deep DIce', 'Plus Dice',
                                      'Deep Jaccard', 'Plus Jaccard', 'Deep f1', 'Plus F1'])
    
    for file_index in range(len(test_file_list)):
        img_filename = test_file_list[file_index]
        img_path = os.path.join(args.img_folder, img_filename)
        
        mask_filename_base = os.path.splitext(img_filename)[0]
        # Construct mask filename by replacing image extension with .npy
        mask_filename = mask_filename_base + '.npy'

        mask_path = os.path.join(args.mask_folder, mask_filename)
        
        try:
            mask_npy = np.load(mask_path)
        except FileNotFoundError:
            print(f"Warning: Mask file '{mask_path}' not found for image '{img_filename}', skipping.")
            continue
        mask_data = mask_npy # Renamed variable for clarity
        
        obj_ids = np.unique(mask_data)
        if len(obj_ids) <= 1: # Skip if only background or empty mask
            print(f"Warning: No objects found or only background in mask for {img_filename}, skipping.")
            continue
        obj_ids = obj_ids[1:] # Remove background ID (assumed to be 0)
        
        # Create a stack of binary masks for each object ID
        masks_all_objects = mask_data == obj_ids[:, None, None] 
        num_objs = len(obj_ids)
        boxes = [] # To store bounding boxes for each object
        valid_object_indices = [] # To track objects that have valid bounding boxes

        # Calculate bounding boxes for each object
        for i in range(num_objs): 
            pos = np.where(masks_all_objects[i])
            if pos[0].size == 0 or pos[1].size == 0: # Check if the mask for this object is empty
                boxes.append(None) 
                continue
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin >= xmax or ymin >= ymax: # Check for invalid box dimensions
                boxes.append(None)
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            valid_object_indices.append(i) # Store original index of valid objects

        img = Image.open(img_path).convert('RGB')
        image_array = np.array(img)

        # Process each valid object detected in the image
        for object_counter, original_object_index in enumerate(valid_object_indices): 
            current_box = boxes[original_object_index]
            metric_dict = {} # Initialize dictionary for this object's metrics

            xmin_obj, ymin_obj, xmax_obj, ymax_obj = current_box
            
            # Crop the region of interest (object) from the main image
            image_region_np = image_array[ymin_obj:ymax_obj, xmin_obj:xmax_obj]
            if image_region_np.size == 0: # Safety check for empty crop
                 print(f"Warning: Empty image region after crop for object at original index {original_object_index} in {img_filename} with box {current_box}. Skipping.")
                 continue
            image_region_for_model_pil = Image.fromarray(image_region_np)
            
            # Crop the ground truth mask for the current object
            current_object_gt_mask_pil = Image.fromarray((masks_all_objects[original_object_index]).astype(np.uint8))
            # Crop relative to the object's own bounding box, not the full image coordinates
            ground_mask_cropped = current_object_gt_mask_pil.crop((xmin_obj, ymin_obj, xmax_obj, ymax_obj))

            # Resize cropped image and mask for model input
            image_region_resized_for_model = image_region_for_model_pil.resize((256,256))
            
            # Prepare tensor for model inference
            input_tensor_for_model = transforms.ToTensor()(image_region_resized_for_model).unsqueeze(0).to(device_to_load)
            
            with torch.no_grad():
                raw_deeplab_output = deeplab_model(input_tensor_for_model) # Corrected variable name
                raw_deeplabplus_output = deeplabv3plus_model(input_tensor_for_model) # Corrected variable name
                
            # Process outputs
            deeplab_argmax_segmentation = np.argmax(np.squeeze(raw_deeplab_output['out']).cpu().numpy(), 0)
            # For the standard DeepLabV3+ (not 3-mask), output processing is typically:
            deeplabplus_processed_segmentation = raw_deeplabplus_output.cpu().numpy() 
            deeplabplus_threshold_segmentation = deeplabplus_processed_segmentation[0][0] > 0.5
            
            deeplab_segmentation_binary = (deeplab_argmax_segmentation==1) 
            # deeplabplus_segmentation_binary should be based on deeplabplus_threshold_segmentation
            deeplabplus_segmentation_binary = deeplabplus_threshold_segmentation

            # Prepare images for plotting
            image_list_for_plot = [image_region_resized_for_model, ground_mask_cropped.resize((256,256)), deeplabplus_segmentation_binary, deeplab_segmentation_binary]
            ground_truth_resized_for_metric = np.array(ground_mask_cropped.resize((256,256))) 
            
            titles = ['Crop_Image','Crop_Ground','DeepLabV3+','DeepLabV3'] # Updated titles
            num_subplots = len(image_list_for_plot)

            fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))
            
            axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes] 
            for i, ax_idx in enumerate(axes_flat): 
                ax_idx.imshow(image_list_for_plot[i])
                ax_idx.set_title(titles[i])
                ax_idx.axis('off')
                ax_idx.set_adjustable('box')
            
            plt.subplots_adjust(top=0.8)

            # Save comparison image if folder is specified
            if args.output_image_folder:
                if not os.path.exists(args.output_image_folder):
                    os.makedirs(args.output_image_folder, exist_ok=True) # Added exist_ok=True
                plt.savefig(os.path.join(args.output_image_folder, f'{img_filename}_{original_object_index}.png'))
            plt.close(fig) # Close figure to free memory
              
            # Prepare binary masks for metric calculation
            final_deeplab_prediction = np.where(deeplab_segmentation_binary ==True, 1 ,0)
            final_ground_truth = np.where(ground_truth_resized_for_metric > 0, 1,0) 
            final_deeplabplus_prediction = np.where(deeplabplus_segmentation_binary==True,1,0)
            
            _ = BinaryAccuracy().to(device_to_load) # This instance is created but its return value not used

            # Calculate and store metrics
            metric_dict = {
                'File Name': f'{img_filename}_{original_object_index}',
                'Deep Lab Accuarcy': metrics.accuracy(final_ground_truth, final_deeplab_prediction), 
                'Deep Plus Accuarcy': metrics.accuracy(final_ground_truth, final_deeplabplus_prediction), 
                'Deep DIce': metrics.dice_coefficient(final_ground_truth, final_deeplab_prediction), 
                'Plus Dice': metrics.dice_coefficient(final_ground_truth, final_deeplabplus_prediction), 
                'Deep Jaccard': metrics.iou_score(final_ground_truth,final_deeplab_prediction), 
                'Plus Jaccard': metrics.iou_score(final_ground_truth, final_deeplabplus_prediction), 
                'Deep f1': f1_score(final_ground_truth, final_deeplab_prediction, average='macro',zero_division=1), 
                'Plus F1': f1_score(final_ground_truth, final_deeplabplus_prediction, average='macro',zero_division=1) 
            }
            metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

    # Save metrics to CSV
    metric_df.to_csv(args.output_csv_file, index=False)
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f'Processing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

if __name__ == '__main__':
    main()