import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp

class_names =['background','main','around']

 
# Get class RGB values
class_rgb_values = [[0, 0, 0], [255, 255, 255]]

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = None # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,in_channels=3
)


"""
Author: Manpreet Singh Minhas
Contact: msminhas at uwaterloo ca
"""
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import matplotlib.pyplot as plt

class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "rgb") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        
        
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        # print(self.root)
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                # indices = np.arange(len(self.image_list)-1)
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            arr = np.load(mask_file)
            arr[0] = np.where(arr[0] ==3, 1, 0)
            arr[1] = np.where(arr[1] ==1, 1, 0)
            arr[2] = np.where(arr[2] ==2, 1, 0)
            mask_arr =np.transpose(arr,(1,2,0)).astype(np.uint8)*255

            mask = Image.fromarray(mask_arr)

            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            # mask = Image.open(mask_file)

            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")

            elif self.mask_color_mode == "grayscale":
                # mask = mask.convert("L")
                mask = mask.convert("L")

            sample = {"image": image, "mask": mask}
          
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
   


            return sample

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms



def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str ='img_same_no_pad',
                                # 'img'
                                 mask_folder: str ='mask_zero_to_three', 
                                #  
                                # #'mask',
                                # Plot the input image, ground truth and the predicted output',

                                 fraction: float = 0.1 ,
                                #  0.1,
                                 batch_size: int = 16
                                #  3
                                ):
  
    # data_transforms = transforms.Compose([transforms.ToTensor()])
    data_transforms = transforms.Compose([ transforms.ToTensor() ])
    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders
data_directory='/home/fisher/Peoples/hseung/NEW/Train'
exp_directory='/home/fisher/Peoples/hseung/SMP_PYROCH/exp'
data_directory = Path(data_directory)
# Create the experiment directory if not present
exp_directory = Path(exp_directory)
if not exp_directory.exists():
    exp_directory.mkdir()

# Specify the loss function
# criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-6)
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
metrics = {'f1_score': f1_score ,
# }
    #    'auroc': roc_auc_score ,

        'accuracy':BinaryAccuracy
    #    }
    #    'confusion_matrix' : confusion_matrix,

    #    'StreamSegMetrics' : StreamSegMetrics}
    #    , 'AverageMeter ': AverageMeter
    }

dataloaders = get_dataloader_single_folder(
    data_directory, batch_size=16)

import copy
import csv
import os
import time
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
from torch import randint, tensor
import torch.nn.functional as F
import warnings
import itertools
def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    warnings.filterwarnings(action='ignore')

    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    # model = nn.DataParallel(model, device_ids = [0, 1])
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}_{n}' for m,n in itertools.product(metrics.keys(),range(0,1))] + \
        [f'Test_{m}_{n}' for m,n in itertools.product(metrics.keys(),range(0,1))]
    with open(os.path.join(bpath, 'deeplabv3plus_3mask_100epoch_sgd.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                
                inputs = sample['image'].to(device)
                # masks = sample['mask'].to(device)
                masks = sample["mask"].to(device)
                
                optimizer.zero_grad()
      
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    # outputs[0][0]
                    loss = criterion(outputs, masks)

                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    

    
                    for name, metric in metrics.items():
                        if name == 'f1_score':

                            batchsummary[f'{phase}_{name}_0'].append(
                                metric(y_true > 0, y_pred > 0.8,average='binary'))          
                            # 5는 내가 직접 만든 ㅈㄴ체 mean 값      
                       
                        elif name =='auroc':

                            pass                                                     
                        elif name =='jaccard':
                            n=0
                            # outputs['out'] 의 jaccard와
                            # 각 배치별 jaccard를 구하자 
                            jaccard = metric(task='binary').to(device)
                            
                            pred = outputs['out']>0.8
                            answer = masks>0
                            
                                       
                            jaccard_total = jaccard(answer, pred).cpu().numpy().ravel()
                            batchsummary[f'{phase}_{name}_0'].append(jaccard_total[n]
                                )                           
                                                       
                            n+=1              
                        elif name =='dice':            
                        # else: /
                            dice= metric(average='micro'
                                        #  ,num_classes=3,multiclass=True
                                         ).to(device)

                            n=0
                            pred = outputs['out']>0.1
                            answer = masks>0
                            
                            dice_total = dice(answer, pred).cpu().numpy().ravel()
                            batchsummary[f'{phase}_{name}_0'].append(dice_total[n]
                                )                           
                                                                 
                        else: 
                            accuracy= metric(
                                        #  ,num_classes=3,multiclass=True
                                         ).to(device)
                            
                            n=0
                            pred = outputs>0.5
                            answer = masks>0
                            
                            accuracy_total = accuracy(answer, pred).cpu().numpy().ravel()
                            batchsummary[f'{phase}_{name}_0'].append(accuracy_total[n]
                                )                           
                            
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        # print(batchsummary)
        with open(os.path.join(bpath, 'deeplabv3plus_3mask_100epoch_sgd.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


_ = train_model(model,
                criterion,
                dataloaders,
                optimizer,
                bpath=exp_directory,
                metrics=metrics,
                num_epochs=30)

torch.save(model, exp_directory / 'deeplabplus_3mask_100_with_sgd.pt')
#sgd로 변경해서 해봄