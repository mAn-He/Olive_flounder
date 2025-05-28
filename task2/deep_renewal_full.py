import os
import pickle
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw # Added ImageDraw here
from sklearn.metrics import f1_score # roc_auc_score, confusion_matrix were not used
from torch.nn import Softmax # Softmax not used explicitly
# Torchmetrics imports not used in the final metric calculation logic shown for this file
# from torchmetrics.classification import Dice,BinaryAccuracy 
# from torchmetrics import JaccardIndex
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used
import torch.nn as nn # Not used

# import box_match # box_match is not defined in the provided context, assuming it's a custom local module.
# If it's essential and missing, this script will fail. For now, commenting out.
# import box_match 

from utils import metrics # Import custom metrics

warnings.filterwarnings('ignore')

# It's generally not recommended to use os.chdir in scripts.
# Prefer using absolute paths or constructing paths relative to the script's location.
# However, preserving original logic for this refactoring task.
img_folder_path = r'/home/fisher/Peoples/hseung/Full/img'
mask_folder_path = r'/home/fisher/Peoples/hseung/Full/mask'

# Ensure we are in the correct directory context if os.listdir() is used without full paths later
# For robustness, this script should ideally not rely on os.chdir.
# For now, we will assume os.chdir is setting context for os.listdir() if it's used that way.
# However, the code uses img_folder + "/" + test_file[file_index], so chdir might not be strictly needed for that part.

original_cwd = os.getcwd() # Save current working directory
os.chdir(img_folder_path)
filenames = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), os.listdir())))
os.chdir(mask_folder_path)
mask_names_files = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), os.listdir()))) # Renamed to avoid conflict
os.chdir(original_cwd) # Restore original working directory

test_file = filenames # Assuming filenames from img_folder are the primary test files

deeplab_model = torch.load(
                   r'/home/fisher/Peoples/hseung/NEW/1st_Trial/dataset_modified/no_pad_3class_3_zero_to_three_dataset3.pt',map_location=torch.device('cuda:2'))

deeplab_model.eval()
metric_df = pd.DataFrame(columns=['File Name','Accuarcy', 'DIce', 'Jaccard', 'f1'])

device = torch.device('cuda:2')

test_file_word = [re.sub(r'\.JPG$', '', filename) for filename in test_file]

with open('/home/fisher/Peoples/hseung/NEW/yolo_swin/pickles/test_bbox_annotation_modify.pkl', 'rb') as an_box:
    bbox_annotations_data = pickle.load(an_box)

for file_index in range(119,len(test_file)):
     metric_dict = {}
     img_path = os.path.join(img_folder_path, test_file[file_index]) # Use os.path.join for robustness
     mask_path = os.path.join(mask_folder_path, test_file[file_index].replace(".JPG", ".npy")) # Assuming mask has same name but .npy
     mask_npy = np.load(mask_path)
     mask=mask_npy
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]

     # split the color-encoded mask into a set
     # of binary masks
     img = Image.open(img_path).convert('RGB')
     masks = mask == obj_ids[:, None, None]
     # img_trans = transforms.ToTensor()(img).unsqueeze(0) # This was for the full image, not used in the loop for cropped parts.
     
     new_name=test_file_word[file_index]
     # match_bboxes = [] # Unused
     # match_box = {} # Unused

     true_bboxes = bbox_annotations_data[new_name]
     for bbox_index in range(len(true_bboxes)):
        true_bbox_coords = true_bboxes[bbox_index]
        # It seems new_image is cropped from 'img' which is the full image.
        # This is correct.
        new_image = img.crop(true_bbox_coords) 
        new_image = new_image.convert('RGB')
        new_image_target_size= new_image.size
        new_image = new_image.resize((256,256))
        cropped_image_tensor = transforms.ToTensor()(new_image).unsqueeze(0)
        # start_time = time.time()
        with torch.no_grad():
            raw_deeplab_output = deeplab_model(cropped_image_tensor.to(torch.device('cuda:2')))
        # end_time= time.time()
        # inference_time= end_time - start_time
        deeplab_argmax_segmentation = np.argmax(np.squeeze(raw_deeplab_output['out']).cpu().numpy(), 0)
        # ground_resize = ground_mask.resize((256,256))
        deeplab_segmentation_binary = (deeplab_argmax_segmentation==1)
        
        deeplab_segmentation_image = Image.fromarray((deeplab_segmentation_binary).astype(np.uint8))
        deeplab_segmentation_image = deeplab_segmentation_image.resize(new_image_target_size)
        deeplab_segmentation_array = np.array(deeplab_segmentation_image)
                
        x_min, y_min, x_max, y_max = map(int, true_bbox_coords)
        y_min, y_max = y_min, y_min + deeplab_segmentation_array.shape[0]
        x_min, x_max = x_min, x_min + deeplab_segmentation_array.shape[1]
        target_size = img.size
        full_image_mask_array = np.zeros((target_size[1],target_size[0]), dtype=int)
        # print('Area :', abs(y_min-y_max),abs(x_min-x_max)) # Comment translated by previous worker
        deeplab_segmentation_array = np.where(deeplab_segmentation_array==True, 1 ,0)
        full_image_mask_array[y_min:y_max, x_min:x_max] = deeplab_segmentation_array
        # new_name (already defined)
        # '20220817_202GOPRO_G0084450' (example data)
        # source of the error (comment translated by previous worker)
        image_list = [img,masks[bbox_index],full_image_mask_array] # img is full image, masks[bbox_index] is GT for current object
            
        # title list (comment translated by previous worker)
        titles = ['Crop_Image','Crop_Ground','deeplab'] # 'Crop_Image' here is actually the full image.
        num_subplots = len(image_list)

        # subplot layout settings (comment translated by previous worker)
        rows = 1
        cols = num_subplots

        # create subplots (comment translated by previous worker)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(image_list[i])
            ax.set_title(titles[i])
            ax.axis('off')
            ax.set_adjustable('box')  # maintain image ratio (comment translated by previous worker)
            
        # adjust title spacing (comment translated by previous worker)
        plt.subplots_adjust(top=0.8)
        plt.savefig(f'/home/fisher/Peoples/hseung/REAL_ARTICLE/deeplab_output_full/{new_name}_{bbox_index}.png')
        
        
        # From here, it's code for viewing metrics (comment translated by previous worker)
        final_deeplab_prediction_binary = np.where(full_image_mask_array ==1, 1 ,0)
        # ground_truth_roi_binary is the mask for the current object (ROI)
        ground_truth_roi_binary = np.where(masks[bbox_index]>0, 1,0) 

        metric_dict = {
            'File Name': f'{new_name}_{bbox_index}',
            'Accuarcy': metrics.accuracy(ground_truth_roi_binary,final_deeplab_prediction_binary), # .item() removed
            'DIce': metrics.dice_coefficient(ground_truth_roi_binary, final_deeplab_prediction_binary), # .item() removed
            'Jaccard': metrics.iou_score(ground_truth_roi_binary,final_deeplab_prediction_binary), # .item() removed
            'f1': f1_score(ground_truth_roi_binary, final_deeplab_prediction_binary, average='macro',zero_division=1), # .item() not needed
        }
        metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

metric_df.to_csv('/home/fisher/Peoples/hseung/REAL_ARTICLE/deep_renewal_full1.csv', index=False)
