import os
import pickle
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision # Not used directly
from PIL import Image, ImageDraw # Added ImageDraw
from sklearn.metrics import f1_score # roc_auc_score, confusion_matrix not used
# from torch.nn import Softmax # Not used
# Torchmetrics imports not used
# from torchmetrics.classification import Dice,BinaryAccuracy 
# from torchmetrics import JaccardIndex 
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used
import torch.nn as nn # Not used

# import box_match # Assuming this is a custom local module
from utils import metrics # Import custom metrics

warnings.filterwarnings('ignore')

img_folder_path = r'./Full/img'
mask_folder_path = r'./Full/mask'

original_cwd = os.getcwd()
os.chdir(img_folder_path)
filenames = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), os.listdir())))
os.chdir(mask_folder_path)
# mask_names_files = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), os.listdir()))) # Not directly used
os.chdir(original_cwd)

test_file = filenames

# Now the model can be used to perform inference on images. (Comment translated by previous worker)
# unet_model= torch.load(r'/home/fisher/Peoples/suyeon/Paper/Unet/Unet012/Save_model/ES_epoch_9.pth')
unet_model= torch.load(r'/home/fisher/Peoples/suyeon/Paper/Unet/Unet012/Save_model/epoch_50.pth')
unet_model.to(torch.device('cuda:0'))
unet_model.eval()
metric_df = pd.DataFrame(columns=['File Name','Accuarcy', 'DIce', 'Jaccard', 'f1'])

device = torch.device('cuda:0')

test_file_word = [re.sub(r'\.JPG$', '', filename) for filename in test_file]

with open('/home/fisher/Peoples/hseung/NEW/yolo_swin/pickles/test_bbox_annotation_modify.pkl', 'rb') as an_box:
    bbox_annotations_data = pickle.load(an_box)

for file_index in range(90,len(test_file)):
     # metric_dict = {} # This metric_dict is defined but not used as the metric calculation part is commented out
     img_path = os.path.join(img_folder_path, test_file[file_index])
     mask_path = os.path.join(mask_folder_path, test_file[file_index].replace(".JPG", ".npy"))
     mask_npy = np.load(mask_path)
     mask=mask_npy
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]

     # split the color-encoded mask into a set
     # of binary masks
     img = Image.open(img_path).convert('RGB')
     masks = mask == obj_ids[:, None, None]
     # img_trans = transforms.ToTensor()(img).unsqueeze(0) # For full image, not used in loop
     
     new_name=test_file_word[file_index]
     # match_bboxes = [] # Unused
     # match_box = {} # Unused

     true_bboxes = bbox_annotations_data[new_name]
     for bbox_index in range(len(true_bboxes)):
        true_bbox_coords = true_bboxes[bbox_index]
        new_image = img.crop(true_bbox_coords) # Cropping from full image 'img'
        new_image = new_image.convert('RGB')
        new_image_target_size= new_image.size
        new_image = new_image.resize((256,256))
        cropped_image_tensor = transforms.ToTensor()(new_image).unsqueeze(0)
        with torch.no_grad():
            raw_unet_output = unet_model(cropped_image_tensor.to(torch.device('cuda:0')))
        threshold=0.5
        unet_output_squeezed_numpy = raw_unet_output.cpu().numpy()
        unet_output_squeezed_numpy = np.squeeze(unet_output_squeezed_numpy)
        unet_output_thresholded = (unet_output_squeezed_numpy >= threshold)
        unet_output_thresholded_transposed = unet_output_thresholded.transpose((1, 2, 0))

        channel_0 = unet_output_thresholded_transposed[:, :, 0]
        channel_1 = unet_output_thresholded_transposed[:, :, 1]
        channel_2 = unet_output_thresholded_transposed[:, :, 2]
        # cc = np.argmax(np.squeeze(c['out']).cpu().numpy(), 0)
        # deeplab_output = (cc==1)
        unet_segmentation_binary_channel = channel_1 # Assuming channel 1 is the target segmentation

        
        unet_segmentation_image = Image.fromarray((unet_segmentation_binary_channel).astype(np.uint8))
        unet_segmentation_image = unet_segmentation_image.resize(new_image_target_size)
        unet_segmentation_array = np.array(unet_segmentation_image)
                
        x_min, y_min, x_max, y_max = map(int, true_bbox_coords)
        y_min, y_max = y_min, y_min + unet_segmentation_array.shape[0]
        x_min, x_max = x_min, x_min + unet_segmentation_array.shape[1]
        target_size = img.size
        full_image_mask_array = np.zeros((target_size[1],target_size[0]), dtype=int)
        # print('Area :', abs(y_min-y_max),abs(x_min-x_max)) # Comment translated by previous worker
        unet_segmentation_array = np.where(unet_segmentation_array==True, 1 ,0)
        full_image_mask_array[y_min:y_max, x_min:x_max] = unet_segmentation_array
        # new_name (already defined)
        # '20220817_202GOPRO_G0084450' (example data)
        # source of the error (comment translated by previous worker)
        image_list = [img,masks[bbox_index],full_image_mask_array] # img is full image, masks[bbox_index] is GT for current object
            
        # title list (comment translated by previous worker)
        titles = ['Crop_Image','Crop_Ground','unet'] # 'Crop_Image' here is actually the full image.
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
        plt.savefig(f'/home/fisher/Peoples/hseung/REAL_ARTICLE/unet_output_full/{new_name}_{bbox_index}.png')
        
        # The following metric calculation is commented out in the original script.
        # Applying changes to it as if it were active, to show usage of utils.metrics.
#         final_unet_prediction_binary = np.where(full_image_mask_array ==1, 1 ,0)
#         ground_truth_roi_binary = np.where(masks[bbox_index]>0, 1,0)
    
#         # metric_dict = {
#         #     'File Name': f'{new_name}_{bbox_index}',
#         #     'Accuarcy': metrics.accuracy(ground_truth_roi_binary,final_unet_prediction_binary), # .item() removed
#         #     'DIce': metrics.dice_coefficient(ground_truth_roi_binary, final_unet_prediction_binary), # .item() removed
#         #     'Jaccard': metrics.iou_score(ground_truth_roi_binary,final_unet_prediction_binary), # .item() removed
#         #     'f1': f1_score(ground_truth_roi_binary, final_unet_prediction_binary, average='macro',zero_division=1), # .item() not needed
#         # }
#         # metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

# # metric_df.to_csv('/home/fisher/Peoples/hseung/REAL_ARTICLE/unet_full1.csv', index=False)
