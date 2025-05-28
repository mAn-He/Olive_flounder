import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score # roc_auc_score, confusion_matrix not used directly
# from torch.nn import Softmax # Softmax not used explicitly
# from torchmetrics.classification import Dice # Dice from torchmetrics is not used
from torchmetrics.classification import BinaryAccuracy # BinaryAccuracy is used
# from torchmetrics import JaccardIndex # JaccardIndex from torchmetrics is not used
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used

from utils import metrics # Import custom metrics
import argparse # New import

warnings.filterwarnings('ignore')

start = time.time()
deeplab = torch.load(
                   r'/home/fisher/Peoples/hseung/NEW/1st_Trial/dataset_modified/no_pad_3class_3_zero_to_three_dataset3.pt',map_location=torch.device('cuda:2'))
deeplabv3plus = torch.load(r'/home/fisher/Peoples/hseung/SMP_PYROCH/exp/deeplabplus_3mask_30.pt',map_location=torch.device('cuda:2'))
deeplabv3plus.eval()
deeplab.eval()

img_folder =r'/home/fisher/Peoples/hseung/Full/img' 
os.chdir(img_folder)
filenames = os.listdir()
test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), filenames))) 
# len(test_file)
mask_folder = r'/home/fisher/Peoples/hseung/Full/mask'
os.chdir(mask_folder)
mask_names = os.listdir()
mask_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), mask_names)))

metric_df = pd.DataFrame(columns=['File Name','Deep Lab Accuarcy', 'Deep Plus Accuarcy', 'Deep DIce', 'Plus Dice',
                                  'Deep Jaccard', 'Plus Jaccard', 'Deep f1', 'Plus F1'])# metric_df = pd.DataFrame(columns = ['Deep Lab Accuarcy','Mask Accuarcy', 'Deep DIce', 'Mask Dice', 'Deep Jaccard', 'Mask Jaccard', 'Deep f1', 'Mask F1'])
for file_index in range(len(test_file)):
     metric_dict = {}
     img_path = img_folder + "/"+ test_file[file_index]
     mask_path = mask_folder +"/" + mask_file[file_index]
     mask_npy = np.load(mask_path)
     mask=mask_npy
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]

     # split the color-encoded mask into a set
     # of binary masks
     masks = mask == obj_ids[:, None, None]


     # get bounding box coordinates for each mask
     num_objs = len(obj_ids)
     # save_name =test_file[file_index].split('.')[0]
     boxes = []
     for i in range(num_objs):
          pos = np.where(masks[i])
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])
          boxes.append([xmin, ymin, xmax, ymax])
          # load boxes
     # this is the whole image
     img = Image.open(img_path).convert('RGB')
     image_array = np.array(img)
     for object_index in range(num_objs):
        boxes[object_index]
        target_size = (img.size[1],img.size[0])
        rcnn_trial = np.zeros(target_size,dtype=int)
        rcnn_trial[boxes[object_index][1]:boxes[object_index][3],boxes[object_index][0]:boxes[object_index][2]]=1
        mask_trial = np.stack([rcnn_trial,rcnn_trial,rcnn_trial], axis=-1)
        
        # here, it's given as a box
        masked_image = image_array * mask_trial


        masked_images = Image.fromarray(masked_image.astype('uint8'))
        img_trans = transforms.ToTensor()(masked_images).unsqueeze(0)
        deeplabv3plus.to(torch.device('cuda:2'))
        deeplab.to(torch.device('cuda:2'))

        # rcnn_mask = a_mask[0]['masks'].cpu().numpy()
    
        coords = boxes[object_index]

        ground_truth = Image.fromarray((masks[object_index]).astype(np.uint8))
        ground_mask = ground_truth.crop(coords)

        new_image = masked_images.crop(coords)
        new_image = new_image.resize((256,256))
        # new_image= new_image.convert('RGB')
        newimg_trans = transforms.ToTensor()(new_image).unsqueeze(0)
        with torch.no_grad():
            raw_deeplab_output = deeplab(newimg_trans.to(torch.device('cuda:2')))
            raw_deeplabplus_output = deeplabv3plus(newimg_trans.to(torch.device('cuda:2')))
        deeplab_argmax_segmentation = np.argmax(np.squeeze(raw_deeplab_output['out']).cpu().numpy(), 0)
        deeplabplus_argmax_segmentation = np.argmax(np.squeeze(raw_deeplabplus_output).cpu().numpy(), 0)
        deeplab_segmentation_binary = (deeplab_argmax_segmentation==1)
        deeplabplus_segmentation_binary = (deeplabplus_argmax_segmentation==1)
        image_list = [new_image,ground_mask.resize((256,256)),deeplabplus_segmentation_binary, deeplab_segmentation_binary]
        ground_resize = ground_mask.resize((256,256))
        ground_truth_resized_numpy = np.array(ground_resize)
        titles = ['Crop_Image','Crop_Ground','plus','1_channel']
        num_subplots = len(image_list)

        # subplot layout settings
        rows = 1
        cols = num_subplots

        # create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        name = test_file[file_index]

        #   add each image and title to the subplot
        for i, ax in enumerate(axes):
            ax.imshow(image_list[i])
            ax.set_title(titles[i])
            ax.axis('off')
            ax.set_adjustable('box')  # maintain image ratio
          
# # adjust title spacing
        plt.subplots_adjust(top=0.8)

        # plt.savefig(f'/home/fisher/Peoples/hseung/NEW/segment outputs/deeplabv3_plus_vs_deeplab_3mask/{name}_{object_index}.png')
          
        #metric time
        final_deeplab_prediction = np.where(deeplab_segmentation_binary ==True, 1 ,0)
        final_ground_truth = np.where(ground_truth_resized_numpy>0, 1,0)
        final_deeplabplus_prediction = np.where(deeplabplus_segmentation_binary==True,1,0)
        
        # Note: accuracy_metric using torchmetrics.BinaryAccuracy is defined but not used in filling metric_dict.
        # The custom metrics.accuracy is used instead.
        accuracy_metric = BinaryAccuracy().to(torch.device('cuda:2'))

        metric_dict = {
            'File Name': f'{name}_{object_index}',
            'Deep Lab Accuarcy': metrics.accuracy(final_ground_truth, final_deeplab_prediction), # .item() removed
            'Deep Plus Accuarcy': metrics.accuracy(final_ground_truth,final_deeplabplus_prediction), # .item() removed
            'Deep DIce': metrics.dice_coefficient(final_ground_truth, final_deeplab_prediction), # .item() removed
            'Plus Dice': metrics.dice_coefficient(final_ground_truth, final_deeplabplus_prediction), # .item() removed
            'Deep Jaccard': metrics.iou_score(final_ground_truth,final_deeplab_prediction), # .item() removed
            'Plus Jaccard': metrics.iou_score(final_ground_truth, final_deeplabplus_prediction), # .item() removed
            'Deep f1': f1_score(final_ground_truth, final_deeplab_prediction, average='macro',zero_division=1), # .item() not needed
            'Plus F1': f1_score(final_ground_truth, final_deeplabplus_prediction, average='macro',zero_division=1) # .item() not needed
        }
        metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

metric_df.to_csv('/home/fisher/Peoples/hseung/NEW/deeplabv3plus_output_metric/deeplabplus_3mask_vs_deeplab_metric_new.csv', index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for model evaluation.")
    parser.add_argument('--test_arg', type=str, default='hello', help='A test argument.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Test argument: {args.test_arg}")
    # Original script's main logic would eventually go here or be called from here
    print("Original script logic would run after this.")