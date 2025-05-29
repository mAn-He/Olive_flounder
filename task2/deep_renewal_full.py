import os
import pickle
import re
import time
import warnings

import matplotlib.pyplot as plt # Retained for saving figures
import numpy as np
import pandas as pd
import torch
import torchvision # Retained as models from torchvision.models might be used implicitly or in future
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score
from torchvision import transforms

# Unused imports like Softmax, FasterRCNNPredictor, MaskRCNNPredictor, nn, and commented torchmetrics are removed.

from utils import metrics # Import custom metrics

warnings.filterwarnings('ignore')

# --- Configuration Section ---
img_folder_path = r'./Full/img'
mask_folder_path = r'./Full/mask'

output_csv_path = '/home/fisher/Peoples/hseung/REAL_ARTICLE/deep_renewal_full1.csv' # Absolute path, kept as is
output_comparison_images_folder = '/home/fisher/Peoples/hseung/REAL_ARTICLE/deeplab_output_full/' # Absolute path, kept as is
pickle_path_bbox_annotations = '/home/fisher/Peoples/hseung/NEW/yolo_swin/pickles/test_bbox_annotation_modify.pkl' # Absolute path
model_path_deeplab = r'/home/fisher/Peoples/hseung/NEW/1st_Trial/dataset_modified/no_pad_3class_3_zero_to_three_dataset3.pt' # Absolute path

processing_device = torch.device('cuda:2') 
# --- End Configuration Section ---

# Load the DeepLab model
try:
    deeplab_model = torch.load(model_path_deeplab, map_location=processing_device)
except FileNotFoundError:
    print(f"Error: DeepLab model not found at {model_path_deeplab}")
    exit()
except Exception as e:
    print(f"Error loading DeepLab model: {e}")
    exit()
deeplab_model.eval()
deeplab_model.to(processing_device)

# List and filter image files (os.chdir removed)
try:
    all_filenames_in_img_folder = os.listdir(img_folder_path)
    test_file_list = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), all_filenames_in_img_folder)))
    if not test_file_list:
        print(f"No matching image files found in {img_folder_path}")
        exit()
except FileNotFoundError:
    print(f"Error: Image folder not found at '{img_folder_path}'.")
    exit()

metric_df = pd.DataFrame(columns=['File Name','Accuarcy', 'DIce', 'Jaccard', 'f1'])
test_file_basenames = [re.sub(r'\.JPG$', '', filename, flags=re.IGNORECASE) for filename in test_file_list]

try:
    with open(pickle_path_bbox_annotations, 'rb') as an_box:
        bbox_annotations_data = pickle.load(an_box)
except FileNotFoundError:
    print(f"Error: Bounding box annotations pickle file not found at {pickle_path_bbox_annotations}")
    exit()
except Exception as e:
    print(f"Error loading bounding box annotations: {e}")
    exit()

# Main processing loop (starts from index 119 as per original script)
for file_index in range(119, len(test_file_list)):
    img_filename = test_file_list[file_index]
    img_basename = test_file_basenames[file_index]

    img_path = os.path.join(img_folder_path, img_filename)
    mask_filename = img_basename + ".npy"
    mask_path = os.path.join(mask_folder_path, mask_filename)

    try:
        mask_numpy_array = np.load(mask_path)
    except FileNotFoundError:
        print(f"Warning: Mask file '{mask_path}' not found for image '{img_filename}'. Skipping.")
        continue
    
    img_pil = Image.open(img_path).convert('RGB')
    
    obj_ids = np.unique(mask_numpy_array)
    if len(obj_ids) <= 1:
        print(f"Warning: No distinct objects found in mask for {img_filename}. Skipping.")
        continue
    obj_ids = obj_ids[1:]
    object_masks_binary = mask_numpy_array == obj_ids[:, None, None]

    if img_basename not in bbox_annotations_data:
        print(f"Warning: No bounding box annotation for {img_basename}. Skipping image.")
        continue
    true_bboxes_for_image = bbox_annotations_data[img_basename]

    for bbox_idx, true_bbox_coords in enumerate(true_bboxes_for_image):
        cropped_img_pil = img_pil.crop(true_bbox_coords)
        cropped_img_pil = cropped_img_pil.convert('RGB')
        original_crop_size = cropped_img_pil.size
        
        resized_cropped_img_pil = cropped_img_pil.resize((256, 256))
        input_tensor = transforms.ToTensor()(resized_cropped_img_pil).unsqueeze(0).to(processing_device)

        with torch.no_grad():
            raw_model_output = deeplab_model(input_tensor)
        
        output_argmax = np.argmax(np.squeeze(raw_model_output['out']).cpu().numpy(), 0)
        prediction_binary_map_resized = (output_argmax == 1)
        
        prediction_pil = Image.fromarray(prediction_binary_map_resized.astype(np.uint8) * 255)
        prediction_resized_to_original_crop = prediction_pil.resize(original_crop_size)
        prediction_array_original_crop_size = np.array(prediction_resized_to_original_crop) / 255
        prediction_array_original_crop_size = prediction_array_original_crop_size.astype(np.uint8)

        x_min, y_min, _, _ = map(int, true_bbox_coords) # x_max, y_max from bbox not needed for placement start
        crop_height, crop_width = prediction_array_original_crop_size.shape
        
        final_prediction_full_image = np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8)
        
        # Ensure slicing for placement is within bounds of the full image
        place_y_start = y_min
        place_y_end = min(y_min + crop_height, img_pil.size[1])
        place_x_start = x_min
        place_x_end = min(x_min + crop_width, img_pil.size[0])
        
        # Ensure source crop matches destination slice
        src_crop_height = place_y_end - place_y_start
        src_crop_width = place_x_end - place_x_start

        final_prediction_full_image[place_y_start:place_y_end, place_x_start:place_x_end] = \
            prediction_array_original_crop_size[0:src_crop_height, 0:src_crop_width]
        
        if bbox_idx < len(object_masks_binary):
            ground_truth_roi_binary = object_masks_binary[bbox_idx].astype(np.uint8)
        else:
            print(f"Warning: bbox_idx {bbox_idx} out of bounds for object_masks_binary for {img_filename}. Skipping object.")
            continue
        
        # Minimal plotting for verification, original extensive plotting block removed
        if output_comparison_images_folder:
            os.makedirs(output_comparison_images_folder, exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_pil); axes[0].set_title(f'Original Full Img: {img_basename}')
            axes[1].imshow(ground_truth_roi_binary, cmap='gray'); axes[1].set_title(f'GT Obj {bbox_idx}')
            axes[2].imshow(final_prediction_full_image, cmap='gray'); axes[2].set_title(f'Pred Obj {bbox_idx} (Full)')
            for ax_item in axes: ax_item.axis('off')
            plt.subplots_adjust(top=0.85) # Adjusted for potentially longer titles
            plot_savename = f'{img_basename}_{bbox_idx}_comparison.png'
            plt.savefig(os.path.join(output_comparison_images_folder, plot_savename))
            plt.close(fig)
        
        metric_dict = {
            'File Name': f'{img_basename}_{bbox_idx}',
            'Accuarcy': metrics.accuracy(ground_truth_roi_binary, final_prediction_full_image),
            'DIce': metrics.dice_coefficient(ground_truth_roi_binary, final_prediction_full_image),
            'Jaccard': metrics.iou_score(ground_truth_roi_binary, final_prediction_full_image),
            'f1': f1_score(ground_truth_roi_binary.flatten(), final_prediction_full_image.flatten(), average='binary', zero_division=1),
        }
        metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

if not metric_df.empty:
    metric_df = metric_df.reset_index(drop=True)
metric_df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Metrics saved to {output_csv_path}")
