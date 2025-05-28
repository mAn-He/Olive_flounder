from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
# from torchmetrics.classification import Dice,BinaryAccuracy # Not used
# from torchmetrics import JaccardIndex # Not used
from sklearn.metrics import f1_score # roc_auc_score, confusion_matrix not used
import time

# from PIL import Image # Redundant
# from torchvision import transforms # Redundant
# import numpy as np # Redundant
import pandas as pd
import warnings
# warnings.filterwarnings('ignore') # Moved to top
# # mask_files = 

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# start = time.time() # Moved to top
# import torch # Redundant
# import torchvision # Redundant
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Redundant
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Redundant
import torch.nn as nn # Not used directly for model definition here, but MaskRCNN uses it.
rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False) # pretrained is standard
num_classes=2
# get the number of input features for the classifier
in_features = rcnn.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                hidden_layer,
                                                num_classes,)

weights_path = '/home/fisher/Peoples/kcpark/saved_model/train_test_compare_epoch_100.pt'  # weights file path

checkpoint = torch.load(weights_path, map_location=torch.device('cuda:0'))

rcnn.load_state_dict(checkpoint)

rcnn.eval()


import new_metric_by_s # Retaining for process_mask
import os # Moved to top
# mask_metrics_calculator = new_metric_by_s.MaskMetrics() # Instance created after imports
# img_folder =r'/home/fisher/Peoples/hseung/Full/img' # Path construction improved
# os.chdir(img_folder) # os.chdir removed for better practice
# filenames = os.listdir() # Used with chdir, now using full paths
# test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), filenames)))
import re # Moved to top
# test_file_word = [re.sub(r'\.JPG$', '', filename) for filename in test_file]
# mask_folder = r'/home/fisher/Peoples/hseung/Full/mask' # Path construction improved
# os.chdir(mask_folder) # os.chdir removed
# mask_names = os.listdir() # Not directly used, mask_path constructed
# mask_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), mask_names))) # Not directly used
import pickle # Moved to top
# import box_match # Assuming this is a custom local module. If not found, this will error.
# from torchvision import transforms # Redundant

from utils import metrics # Import custom metrics
import box_match # Assuming this is a custom local module or installed package

# ground truth (comment from prev worker)
with open('/home/fisher/Peoples/hseung/NEW/yolo_swin/pickles/test_bbox_annotation_modify.pkl', 'rb') as an_box:
    bbox_annotations_data = pickle.load(an_box)
# from swin (comment from prev worker)

mask_metrics_calculator = new_metric_by_s.MaskMetrics() # Instance for process_mask

img_folder_path = r'/home/fisher/Peoples/hseung/Full/img'
mask_folder_path = r'/home/fisher/Peoples/hseung/Full/mask'

original_cwd = os.getcwd()
os.chdir(img_folder_path) # Preserving original os.chdir logic for now
test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), os.listdir())))
os.chdir(original_cwd)

test_file_word = [re.sub(r'\.JPG$', '', filename) for filename in test_file]


metric_df = pd.DataFrame(columns=['File Name','time', 'Mask DIce',
                                  'Mask Jaccard',  'Mask f1','BOX IOU'])
for file_index in range(len(test_file)):
     metric_dict = {}
     img_path = os.path.join(img_folder_path, test_file[file_index])
     mask_path = os.path.join(mask_folder_path, test_file[file_index].replace(".JPG", ".npy"))
     # save_name =test_file[file_index].split('.')[0]
     filename= test_file_word[file_index]
     mask_numpy_array = np.load(mask_path)
     mask=mask_numpy_array # mask will be used later for masks = mask == obj_ids[:, None, None]
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]
     device = torch.device('cuda:0')
     img = Image.open(img_path).convert('RGB')
     image_array = np.array(img)
     # arr = np.load(mask_path) # This is redundant, mask_numpy_array is already loaded
     ground_truth_boxes = bbox_annotations_data[filename]
     target_size = (img.size[1],img.size[0])
     # rcnn_trial = np.zeros(target_size,dtype=int) # Initialized later inside the loop
     for bbox_index in range(len(ground_truth_boxes)):


     # split the color-encoded mask into a set
     # of binary masks
        masks = mask == obj_ids[:, None, None] # mask is from mask_numpy_array
        box_mask_for_cropping = np.zeros(target_size,dtype=int)
        box_mask_for_cropping[ground_truth_boxes[bbox_index][1]:ground_truth_boxes[bbox_index][3],ground_truth_boxes[bbox_index][0]:ground_truth_boxes[bbox_index][2]]=1
        rgb_box_mask_for_cropping = np.stack([box_mask_for_cropping,box_mask_for_cropping,box_mask_for_cropping], axis=-1)
        ground_truth_roi_numpy = masks[bbox_index] # Assuming obj_ids correspond to the order of masks
     # here, it's given as a box
        masked_image = image_array * rgb_box_mask_for_cropping
 
 
        cropped_image_pil = Image.fromarray(masked_image.astype('uint8'))

        # device = torch.device('cuda:0')
        rcnn.to(device)
        # new_img= crop_img.resize()
        # mask_1 = masks[bbox_index] # Redundant with ground_truth_roi_numpy
        # ground_truth_roi_expanded_dims = ground_truth_roi_numpy[np.newaxis, :] # Unused
        cropped_image_tensor = transforms.ToTensor()(cropped_image_pil).unsqueeze(0)
        # gt_dict = [{'boxes': torch.tensor([ground_truth_boxes[bbox_index]], dtype=torch.float).to(device),
        #     'labels': torch.tensor([1]).to(device),
        #     }]

        with torch.no_grad():
            rcnn_prediction_output = rcnn(cropped_image_tensor.to(torch.device('cuda:0')))
        predicted_boxes_tensor = rcnn_prediction_output[0]['boxes']
        # match_bboxes = [] # Unused accumulation
        current_file_bbox_matches = {}
        current_ground_truth_box_list = [ground_truth_boxes[bbox_index]]
        bbox_matches = box_match.match_bbox_xyxy(predicted_boxes_tensor, current_ground_truth_box_list)
        current_file_bbox_matches[filename] = bbox_matches
        # match_bboxes.append(current_file_bbox_matches) # Unused accumulation
        for match_key_index in range(len(bbox_matches.keys())):
            prediction_box_index =  bbox_matches[match_key_index]
            if prediction_box_index is None:
                # continue
                predicted_segmentation_mask_binary = np.zeros((4176,5568)) # Using fixed size, consider dynamic
            else:
                predicted_segmentation_mask_binary = rcnn_prediction_output[0]['masks'][prediction_box_index].cpu().numpy()
                predicted_segmentation_mask_binary = predicted_segmentation_mask_binary[0]>0.5
        #     plt.figure(figsize=(15, 5))
       
        #     titles = ['Image', 'Ground Truth', 'rcnn output']
        #     image_list = [cropped_image_pil, ground_truth_roi_numpy, predicted_segmentation_mask_binary]
        #     num_subplots = len(image_list)
        #     rows = 1
        #     cols = num_subplots

        # # create subplots
        #     fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        #     name = test_file[file_index] # name should be filename for consistency with image saving
        #     for i, ax in enumerate(axes):
        #          ax.imshow(image_list[i])
        #          ax.set_title(titles[i])
        #          ax.axis('off')
        #          ax.set_adjustable('box')  # maintain image ratio
       
        #     # Save the subplot
        #     plt.subplots_adjust(top=0.8) # adjust title spacing

        #     plt.savefig(f'/home/fisher/Peoples/hseung/NEW/segment outputs/new_mask_match_full/{filename}_{bbox_index}')
            final_predicted_mask_binary = np.where(predicted_segmentation_mask_binary ==True, 1 ,0)
            final_ground_truth_mask_binary = np.where(ground_truth_roi_numpy==True, 1,0)
            
            # Calls to mask_metrics_calculator.process_mask are retained as this function is specific to new_metric_by_s
            if np.array_equal(np.unique(final_predicted_mask_binary), [0]):
                processed_pred_mask = final_predicted_mask_binary
            else:
                # Assuming process_mask returns tuple: (processed_mask_visual, processed_mask_binary)
                # And we need the binary mask for metric calculation.
                _, processed_pred_mask = mask_metrics_calculator.process_mask(final_predicted_mask_binary)
            _, processed_groundtruth_mask = mask_metrics_calculator.process_mask(final_ground_truth_mask_binary)
        
            metric_dict = {
                    'File Name': f'{filename}_{bbox_index}',
                    'Mask Accuarcy': metrics.accuracy(final_ground_truth_mask_binary,final_predicted_mask_binary), # .item() removed
                    'Mask DIce': metrics.dice_coefficient(final_ground_truth_mask_binary, final_predicted_mask_binary), # .item() removed
                    'Mask Jaccard': metrics.iou_score(final_ground_truth_mask_binary,final_predicted_mask_binary), # .item() removed
                    'Mask f1': f1_score(final_ground_truth_mask_binary, final_predicted_mask_binary, average='micro',zero_division=1), # .item() not needed
                    'BOX IOU': metrics.iou_score(processed_groundtruth_mask, processed_pred_mask) # Using processed masks for BOX IOU
                }
            metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)


        #  metric_df = metric_df.reset_index(True)
metric_df.to_csv('/home/fisher/Peoples/hseung/NEW/mask_segment/maskrcnn_mathcing_full_f1_micro.csv', index=False)
print('done')
