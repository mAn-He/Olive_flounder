from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix

# img_path ='/home/fisher/Peoples/hseung/Full/img/20220816_100GOPRO_G0020073.JPG'
# '/home/fisher/Peoples/hseung/NEW/Train/img_same_no_pad/20220816_100GOPRO_G0050536_3.jpg'
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
import torch.nn as nn


import os
from PIL import ImageDraw
import box_match


img_folder =r'/home/fisher/Peoples/hseung/Full/img' 
os.chdir(img_folder)
filenames = os.listdir()
test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), filenames))) 
# len(test_file)
mask_folder = r'/home/fisher/Peoples/hseung/Full/mask'
os.chdir(mask_folder)
mask_names = os.listdir()
mask_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), mask_names)))
deeplab = torch.load(
                   r'/home/fisher/Peoples/hseung/NEW/1st_Trial/dataset수정/no_pad_3class_3_zero_to_three_dataset3.pt',map_location=torch.device('cuda:2'))

import time
deeplab.eval()
metric_df = pd.DataFrame(columns=['File Name','Accuarcy', 'DIce', 'Jaccard', 'f1' 
                              ])# metric_df = pd.DataFrame(columns = ['Deep Lab Accuarcy','Mask Accuarcy', 'Deep DIce', 'Mask Dice', 'Deep Jaccard', 'Mask Jaccard', 'Deep f1', 'Mask F1'])
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
import os 
device = torch.device('cuda:2')
import torch
import new_metric_by_s
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
with open('/home/fisher/Peoples/hseung/NEW/yolo_swin/pickles/test_bbox_annotation_modify.pkl', 'rb') as an_box:
    data = pickle.load(an_box)
for n in range(119,len(test_file)):
    # len(test_file)):
     metric_dict = {}
     img_path = img_folder + "/"+ test_file[n]
     mask_path = mask_folder +"/" + mask_file[n]
     mask_npy = np.load(mask_path)
     mask=mask_npy
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]

     # split the color-encoded mask into a set
     # of binary masks
     img = Image.open(img_path).convert('RGB')
     masks = mask == obj_ids[:, None, None]
     img_trans = transforms.ToTensor()(img).unsqueeze(0)

    #  mask_mask = a_mask[0]['masks'].cpu().numpy()
     
     new_name=test_file_word[n]
     match_bboxes = []
     match_box = {}

    #  print(f"filename = {new_name}")
     true_bboxes = data[new_name] 
     for m in range(len(true_bboxes)):
        coords2= true_bboxes[m]
        new_image = img.crop(true_bboxes[m])
        new_image = new_image.convert('RGB')
        new_image_target_size= new_image.size
        new_image = new_image.resize((256,256))
        newimg_trans = transforms.ToTensor()(new_image).unsqueeze(0)
        # start_time = time.time()
        with torch.no_grad():
            b = deeplab(newimg_trans.to(torch.device('cuda:2')))
        # end_time= time.time()
        # inference_time= end_time - start_time
        bb = np.argmax(np.squeeze(b['out']).cpu().numpy(), 0)
        # ground_resize = ground_mask.resize((256,256))
        deeplab_output = (bb==1)
        
        deeplab_output_img = Image.fromarray((deeplab_output).astype(np.uint8))
        deeplab_output_img = deeplab_output_img.resize(new_image_target_size)
        deeplab_arr= np.array(deeplab_output_img)
                
        x_min, y_min, x_max, y_max = map(int, coords2)
        y_min, y_max = y_min, y_min + deeplab_arr.shape[0]
        x_min, x_max = x_min, x_min + deeplab_arr.shape[1]
        target_size = img.size
        blank = np.zeros((target_size[1],target_size[0]), dtype=int)
        # print('넓이 :', abs(y_min-y_max),abs(x_min-x_max)) 
        deeplab_arr = np.where(deeplab_arr==True, 1 ,0)
        blank[y_min:y_max, x_min:x_max] = deeplab_arr
                # new_name
# '20220817_202GOPRO_G0084450'
# # 오류의 진원지
        image_list = [img,masks[m],blank]
            
 
        # crop_original = np.array(original_output)
        # 제목 리스트
        titles = ['Crop_Image','Crop_Ground','deeplab']
        num_subplots = len(image_list)

    # 서브플롯 배치 설정
        rows = 1
        cols = num_subplots

        # 서브플롯 생성
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(image_list[i])
            ax.set_title(titles[i])
            ax.axis('off')
            ax.set_adjustable('box')  # 이미지 비율 유지
            
# # 제목 간 간격 조정
        plt.subplots_adjust(top=0.8)
        plt.savefig(f'/home/fisher/Peoples/hseung/REAL_ARTICLE/deeplab_output_full/{new_name}_{m}.png')
        
        
        # 여기서 부터는 metric 보기 위한 코드
        mask_pred= np.where(blank ==1, 1 ,0)
        truee = np.where(masks[m]>0, 1,0)
    # mask_true= np.where(original_output==True,1,0)

    
    # mask_pred= np.where(crop_rcnn_numpy==True, 1, 0)
        pred_t = mask_pred
        truee_t = truee
        # truee_mask = torch.tensor(mask_true)

                # acc = BinaryAccuracy().to(torch.device('cuda:2'))
                # print('Deep lab의 구성 : ', np.unique(deep_pred))
        # 'File Name','Accuarcy', 'DIce', 'Jaccard','f1'
        metric_dict = {
            'File Name': f'{new_name}_{m}',
            'Accuarcy': metric.accuracy(truee_t,pred_t).item(),
            # 'Deep origin Accuarcy': accuracy(truee_t,mask_pred_t).item(),
            'DIce': metric.dice_coef(truee_t, pred_t).item(),
            # 'origin Dice': dice_coef(truee_t, mask_pred_t).item(),
            'Jaccard': metric.iou(truee_t,pred_t).item(),
            # 'origin Jaccard': iou(truee_t, mask_pred_t).item(),
            'f1': f1_score(truee, mask_pred, average='macro',zero_division=1).item(),
            # 'origin F1': f1_score(truee_t, mask_pred, average='macro',zero_division=1).item()
        }
        metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)


        #  metric_df = metric_df.reset_index(True)
metric_df.to_csv('/home/fisher/Peoples/hseung/REAL_ARTICLE/deep_renewal_full1.csv', index=False)
