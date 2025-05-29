from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
from torchmetrics.classification import BinaryAccuracy # Dice, JaccardIndex from torchmetrics not used
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
import torchvision # Not used directly, but models might use it
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used
# start = time.time() # Moved to top

deeplab_model_012 = torch.load(
                   r'/home/fisher/Peoples/hseung/NEW/1st_Trial/dataset_modified/no_pad_3class_3_zero_to_three_dataset3.pt',map_location=torch.device('cuda:2'))
deeplab_model_01 = torch.load(r'/home/fisher/Peoples/hseung/NEW/1st_Trial/new_learning_Rate/original_real.pt',map_location=torch.device('cuda:2'))
deeplab_model_01.eval()
deeplab_model_012.eval()

# from PIL import Image # Redundant
# from torchvision import transforms # Redundant
# import matplotlib.pyplot as plt # Redundant
# import numpy as np # Redundant
from torch.nn import Softmax # Not used explicitly
# from torchmetrics.classification import Dice,BinaryAccuracy # BinaryAccuracy moved to top
# from torchmetrics import JaccardIndex # Not used
# from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix # f1_score moved to top
import os # Moved to top

# Local metric functions removed, will use utils.metrics

img_folder =r'./Full/img'
os.chdir(img_folder)
filenames = os.listdir()
test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), filenames))) 
# len(test_file)
mask_folder = r'./Full/mask'
os.chdir(mask_folder)
mask_names = os.listdir()
mask_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), mask_names)))

metric_df = pd.DataFrame(columns=['File Name','Deep Lab Model 012 Accuarcy', 'Deep Model 01 Accuarcy', 'Deep Model 012 DIce', 'Deep Model 01 Dice',
                                  'Deep Model 012 Jaccard', 'Deep Model 01 Jaccard', 'Deep Model 012 f1', 'Deep Model 01 F1'])# metric_df = pd.DataFrame(columns = ['Deep Lab Accuarcy','Mask Accuarcy', 'Deep DIce', 'Mask Dice', 'Deep Jaccard', 'Mask Jaccard', 'Deep f1', 'Mask F1'])
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

# The process of creating masked_image is unnecessary... it ended up like this while reusing code from mask rcnn... you can just crop it directly..
        masked_images = Image.fromarray(masked_image.astype('uint8'))

        deeplab_model_01.to(torch.device('cuda:2'))
        deeplab_model_012.to(torch.device('cuda:2'))

        # rcnn_mask = a_mask[0]['masks'].cpu().numpy()
    
        coords = boxes[object_index]

        ground_truth = Image.fromarray((masks[object_index]).astype(np.uint8))
        ground_mask = ground_truth.crop(coords)

        new_image = masked_images.crop(coords)
        new_image = new_image.resize((256,256))
        # new_image= new_image.convert('RGB')
        newimg_trans = transforms.ToTensor()(new_image).unsqueeze(0)
        with torch.no_grad():
            raw_deeplab_model_012_output = deeplab_model_012(newimg_trans.to(torch.device('cuda:2')))
            raw_deeplab_model_01_output = deeplab_model_01(newimg_trans.to(torch.device('cuda:2')))
        deeplab_model_012_argmax_segmentation = np.argmax(np.squeeze(raw_deeplab_model_012_output['out']).cpu().numpy(), 0)
        raw_deeplab_model_01_segmentation = raw_deeplab_model_01_output['out'].cpu().numpy()
        deeplab_model_012_segmentation_binary = (deeplab_model_012_argmax_segmentation==1)
        deeplab_model_01_segmentation_binary = raw_deeplab_model_01_segmentation[0][0]>0.5
        image_list = [new_image,ground_mask.resize((256,256)),deeplab_model_01_segmentation_binary, deeplab_model_012_segmentation_binary]
        ground_resize = ground_mask.resize((256,256))
        ground_truth_resized_numpy = np.array(ground_resize)
        # crop_original = np.array(deeplab_model_01_segmentation_binary)
        # title list
        titles = ['Crop_Image','Crop_Ground','original','1_channel']
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

import os # Already imported at top
import time # Already imported at top
import warnings # Already imported at top

import matplotlib.pyplot as plt # Already imported at top
import numpy as np # Already imported at top
import pandas as pd # Already imported at top
import torch # Already imported at top
# import torchvision # Already imported at top
# from PIL import Image # Already imported at top
# from sklearn.metrics import f1_score # Already imported at top
# from torch.nn import Softmax # Not used
# from torchmetrics.classification import BinaryAccuracy # Already imported at top
# from torchvision import transforms # Already imported at top
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Not used
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Not used

from utils import metrics # Import custom metrics

warnings.filterwarnings('ignore')
start = time.time()
# ... (rest of the imports and code from the top of the file, already handled)
# This section is just to structure the diff correctly for the bottom part.
# Actual changes for imports are at the top of the file.
# The following is the existing code that will be modified below for metric calls:

# ... (previous code for loading models, data, loops etc.)
# The change is in the metric_dict population

        plt.savefig(f'/home/fisher/Peoples/hseung/NEW/segment outputs/original_vsdeeplab/{name}_{object_index}.png')
# While reusing code... I didn't change the variable names much, so it says mask_pred, but it's actually deeplab_model_01. (Comment from previous worker)
        #metric time
        final_deeplab_model_012_prediction = np.where(deeplab_model_012_segmentation_binary ==True, 1 ,0)
        final_ground_truth = np.where(ground_truth_resized_numpy>0, 1,0)
        final_deeplab_model_01_prediction = np.where(deeplab_model_01_segmentation_binary==True,1,0)
        
        # Note: accuracy_metric using torchmetrics.BinaryAccuracy is defined but not used.
        accuracy_metric = BinaryAccuracy().to(torch.device('cuda:2'))

        metric_dict = {
            'File Name': f'{name}_{object_index}',
            'Deep Lab Model 012 Accuarcy': metrics.accuracy(final_ground_truth, final_deeplab_model_012_prediction), # .item() removed
            'Deep Model 01 Accuarcy': metrics.accuracy(final_ground_truth,final_deeplab_model_01_prediction), # .item() removed
            'Deep Model 012 DIce': metrics.dice_coefficient(final_ground_truth, final_deeplab_model_012_prediction), # .item() removed
            'Deep Model 01 Dice': metrics.dice_coefficient(final_ground_truth, final_deeplab_model_01_prediction), # .item() removed
            'Deep Model 012 Jaccard': metrics.iou_score(final_ground_truth,final_deeplab_model_012_prediction), # .item() removed
            'Deep Model 01 Jaccard': metrics.iou_score(final_ground_truth, final_deeplab_model_01_prediction), # .item() removed
            'Deep Model 012 f1': f1_score(final_ground_truth, final_deeplab_model_012_prediction, average='macro',zero_division=1), # .item() not needed
            'Deep Model 01 F1': f1_score(final_ground_truth, final_deeplab_model_01_prediction, average='macro',zero_division=1) # .item() not needed
        }
        metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

     metric_df = metric_df.reset_index(True) # This line was present in the original, keeping it.
metric_df.to_csv('/home/fisher/Peoples/hseung/NEW/deeplabv3plus_output_metric/deeplab01_vs_deeplabv3_012_new_metric_222222_0215.csv', index=False)