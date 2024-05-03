from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
import time

from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# mask_files = 

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
start = time.time()
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn
rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
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

weights_path = '/home/fisher/Peoples/kcpark/saved_model/train_test_compare_epoch_100.pt'  # 가중치 파일 경로

checkpoint = torch.load(weights_path, map_location=torch.device('cuda:0'))

rcnn.load_state_dict(checkpoint)

rcnn.eval()


import new_metric_by_s 
import os
metric= new_metric_by_s.MaskMetrics()
img_folder =r'/home/fisher/Peoples/hseung/Full/img' 
os.chdir(img_folder)
filenames = os.listdir()
test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), filenames))) 
import re
test_file_word = [re.sub(r'\.JPG$', '', filename) for filename in test_file]
mask_folder = r'/home/fisher/Peoples/hseung/Full/mask'
os.chdir(mask_folder)
mask_names = os.listdir()
mask_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), mask_names)))
import pickle
import box_match
from torchvision import transforms
# 정답
with open('/home/fisher/Peoples/hseung/NEW/yolo_swin/pickles/test_bbox_annotation_modify.pkl', 'rb') as an_box:
    data = pickle.load(an_box)
# swin꺼

metric_df = pd.DataFrame(columns=['File Name','time', 'Mask DIce', 
                                  'Mask Jaccard',  'Mask f1','BOX IOU'])# metric_df = pd.DataFrame(columns = ['Deep Lab Accuarcy','Mask Accuarcy', 'Deep DIce', 'Mask Dice', 'Deep Jaccard', 'Mask Jaccard', 'Deep f1', 'Mask F1'])
for n in range(len(test_file)):
     metric_dict = {}
     img_path = img_folder + "/"+ test_file[n]
     mask_path = mask_folder +"/" + mask_file[n]
     # save_name =test_file[n].split('.')[0]
     filename= test_file_word[n]
     mask_npy = np.load(mask_path)
     mask=mask_npy
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]
     device = torch.device('cuda:0')
     img = Image.open(img_path).convert('RGB')
     image_array = np.array(img)
     arr = np.load(mask_path)
     boxes=data[filename]
     target_size = (img.size[1],img.size[0])
     rcnn_trial = np.zeros(target_size,dtype=int)
     for m in range(len(boxes)):


     # split the color-encoded mask into a set
     # of binary masks
        masks = mask == obj_ids[:, None, None]
        rcnn_trial = np.zeros(target_size,dtype=int)
        rcnn_trial[boxes[m][1]:boxes[m][3],boxes[m][0]:boxes[m][2]]=1
        mask_trial = np.stack([rcnn_trial,rcnn_trial,rcnn_trial], axis=-1)
        gt_npy=masks[m]
     #여기선 box로 줬어
        masked_image = image_array * mask_trial
 
 
        masked_images = Image.fromarray(masked_image.astype('uint8'))

        # device = torch.device('cuda:0')
        rcnn.to(device)
        # new_img= crop_img.resize()
        mask_1 = masks[m]
        mask_2 = mask_1[np.newaxis, :]
        img_trans = transforms.ToTensor()(masked_images).unsqueeze(0)
        # gt_dict = [{'boxes': torch.tensor([boxes[m]], dtype=torch.float).to(device), 
        #     'labels': torch.tensor([1]).to(device), 
        #     }]

        with torch.no_grad():
            mask2 = rcnn(img_trans.to(torch.device('cuda:0')))
        output= mask2[0]['boxes']
        match_bboxes = []
        match_box = {}
        ground_box= [boxes[m]]
        matches = box_match.match_bbox_xyxy(output, ground_box)
        match_box[filename] = matches
        match_bboxes.append(match_box)
        for s in range(len(matches.keys())):
            s_index=  matches[s]
            if s_index is None:
                # continue
                output_mask = np.zeros((4176,5568))
            else:
                output_mask = mask2[0]['masks'][s_index].cpu().numpy()
                output_mask = output_mask[0]>0.5
        #     plt.figure(figsize=(15, 5))
       
        #     titles = ['Image', 'Ground Truth', 'rcnn output']
        #     image_list = [masked_images, gt_npy, output_mask]
        #     num_subplots = len(image_list)
        #     rows = 1
        #     cols = num_subplots

        # # 서브플롯 생성
        #     fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        #     name = test_file[n]
        #     for i, ax in enumerate(axes):
        #          ax.imshow(image_list[i])
        #          ax.set_title(titles[i])
        #          ax.axis('off')
        #          ax.set_adjustable('box')  # 이미지 비율 유지
       
        #     # Save the subplot
        #     plt.subplots_adjust(top=0.8)

        #     plt.savefig(f'/home/fisher/Peoples/hseung/NEW/segment outputs/new_mask_match_full/{filename}_{m}')
            mask_pred= np.where(output_mask ==True, 1 ,0)
            truee = np.where(gt_npy==True, 1,0)
            # mask_true= np.where(original_output==True,1,0)

            
            # mask_pred= np.where(crop_rcnn_numpy==True, 1, 0)
            pred_t = mask_pred
            truee_t = truee
            if np.array_equal(np.unique(pred_t), [0]):
                pred_mask = pred_t
            else:
                _,pred_mask = metric.process_mask(pred_t)
            _,groundtruth_mask = metric.process_mask(truee_t)
                # truee_mask = torch.tensor(mask_true)
        
                # acc = BinaryAccuracy().to(torch.device('cuda:2'))
                # print('Deep lab의 구성 : ', np.unique(deep_pred))
        # 'File Name','Accuarcy', 'DIce', 'Jaccard','f1'
            metric_dict = {
                    'File Name': f'{filename}_{m}',
                    'Mask Accuarcy': metric.accuracy(truee_t,pred_t).item(),
                    # 'Deep origin Accuarcy': accuracy(truee_t,mask_pred_t).item(),
                    'Mask DIce': metric.dice_coef(truee_t, pred_t).item(),
                    # 'origin Dice': dice_coef(truee_t, mask_pred_t).item(),
                    'Mask Jaccard': metric.iou(truee_t,pred_t).item(),
                    # 'origin Jaccard': iou(truee_t, mask_pred_t).item(),
                    'Mask f1': f1_score(truee, mask_pred, average='micro',zero_division=1).item(),
                    # 'origin F1': f1_score(truee_t, mask_pred, average='macro',zero_division=1).item()
                    'BOX IOU': metric.iou(groundtruth_mask, pred_mask).item()
                }
            metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)


        #  metric_df = metric_df.reset_index(True)
metric_df.to_csv('/home/fisher/Peoples/hseung/NEW/mask_segment/maskrcnn_mathcing_full_f1_micro.csv', index=False)
print('done')
