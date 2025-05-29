"""
Data loading utilities for '012' type semantic segmentation tasks.

This module provides functions to create PyTorch DataLoaders for image segmentation
on datasets structured for 3-class (0, 1, 2) segmentation. It includes:
- `get_dataloader_sep_folder`: For datasets structured with separate Train/Test folders.
- `get_dataloader_single_folder`: For datasets where Train/Test images are in a single
  set of folders and split based on a fraction.
It utilizes the `SegmentationDataset` class, which should be adapted for '012' data
(e.g., mask processing).
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

# Assuming segdataset.py is in the same directory or accessible in PYTHONPATH
from segdataset import SegmentationDataset 


def get_dataloader_sep_folder(data_dir: str,
                              image_folder: str = 'Image', # Default image folder name
                              mask_folder: str = 'Mask',   # Default mask folder name
                              batch_size: int = 4):
    """ 
    Creates Train and Test DataLoaders from separate 'Train' and 'Test' subdirectories
    for '012' type datasets.

    The expected directory structure is:
    data_dir/
    ├── Train/
    │   ├── Image/ (or specified image_folder)
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── Mask/ (or specified mask_folder)
    │       ├── image1.npy (masks expected to handle 0, 1, 2 classes)
    │       └── ...
    └── Test/
        ├── Image/
        │   └── ...
        └── Mask/
            └── ...

    Args:
        data_dir (str): The root data directory containing 'Train' and 'Test' folders.
        image_folder (str, optional): Name of the subfolder containing images. 
                                      Defaults to 'Image'.
        mask_folder (str, optional): Name of the subfolder containing masks.
                                     Defaults to 'Mask'.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.

    Returns:
        dict: A dictionary containing 'Train' and 'Test' PyTorch DataLoaders.
    """
    # Standard transformations: Convert images to PyTorch tensors.
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               transforms=data_transforms,
                               image_folder=image_folder,
                               mask_folder=mask_folder)
        for x in ['Train', 'Test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True if x == 'Train' else False, # Shuffle only for training
                      num_workers=8) 
        for x in ['Train', 'Test']
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'img_same_no_pad', # Specific to '012' setup
                                 mask_folder: str = 'mask_zero_to_three', # Specific to '012' setup
                                 fraction: float = 0.1,    # Default fraction for test set
                                 batch_size: int = 3):     # Default batch size
    """
    Creates Train and Test DataLoaders from a single directory for '012' type datasets.

    The dataset is split into training and testing sets based on `fraction`.
    Directory structure:
    data_dir/
    ├── img_same_no_pad/ (or specified image_folder)
    │   └── ...
    └── mask_zero_to_three/ (or specified mask_folder, masks handle 0,1,2)
        └── ...

    Args:
        data_dir (str): Root directory for the dataset.
        image_folder (str, optional): Image subfolder name. Defaults to 'img_same_no_pad'.
        mask_folder (str, optional): Mask subfolder name. Defaults to 'mask_zero_to_three'.
        fraction (float, optional): Fraction for the Test set. Defaults to 0.1.
        batch_size (int, optional): Batch size. Defaults to 3.

    Returns:
        dict: A dictionary containing 'Train' and 'Test' PyTorch DataLoaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])
    
    image_datasets = {
        x: SegmentationDataset(root=data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100, # For reproducible splits
                               fraction=fraction,
                               subset=x, # 'Train' or 'Test'
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True if x == 'Train' else False, # Shuffle only for training
                      num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders
