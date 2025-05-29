"""
Data loading utilities for semantic segmentation tasks.

This module provides functions to create PyTorch DataLoaders for image segmentation.
It includes:
- `get_dataloader_sep_folder`: For datasets structured with separate Train/Test folders.
- `get_dataloader_single_folder`: For datasets where Train/Test images are in a single
  set of folders and split based on a fraction.
It utilizes the `SegmentationDataset` class defined in `segdataset.py`.
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset # Assuming segdataset.py is in the same directory


def get_dataloader_sep_folder(data_dir: str,
                              image_folder: str = 'Image', # Standardized to 'Image' from 'Images'
                              mask_folder: str = 'Mask',   # Standardized to 'Mask' from 'Masks'
                              batch_size: int = 4):
    """ 
    Creates Train and Test DataLoaders from separate 'Train' and 'Test' subdirectories.

    The expected directory structure is:
    data_dir/
    ├── Train/
    │   ├── Image/ (or specified image_folder)
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── Mask/ (or specified mask_folder)
    │       ├── image1.png (or .npy)
    │       └── ...
    └── Test/
        ├── Image/
        │   └── ...
        └── Mask/
            └── ...

    Args:
        data_dir (str): The root data directory containing 'Train' and 'Test' folders.
        image_folder (str, optional): Name of the subfolder containing images within
                                      'Train' and 'Test'. Defaults to 'Image'.
        mask_folder (str, optional): Name of the subfolder containing masks within
                                     'Train' and 'Test'. Defaults to 'Mask'.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.

    Returns:
        dict: A dictionary containing 'Train' and 'Test' PyTorch DataLoaders.
    """
    # Define standard transformations for the images.
    # Primarily converts images to PyTorch tensors. Normalization might be added here if needed.
    data_transforms = transforms.Compose([transforms.ToTensor()])

    # Create dataset instances for training and testing sets
    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x, # Construct path to Train or Test folder
                               transforms=data_transforms,
                               image_folder=image_folder,
                               mask_folder=mask_folder)
        for x in ['Train', 'Test'] # Iterate for Train and Test phases
    }
    
    # Create DataLoaders for training and testing
    # Shuffle is True for training to ensure randomness, usually False for testing if order matters.
    # num_workers can be increased to speed up data loading if CPU allows.
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True if x == 'Train' else False, # Shuffle only for training
                      num_workers=8) # Number of subprocesses to use for data loading
        for x in ['Train', 'Test']
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'img', # Default image folder name
                                 mask_folder: str = 'mask',  # Default mask folder name
                                 fraction: float = 0.2,    # Fraction of data to be used for testing
                                 batch_size: int = 4):
    """
    Creates Train and Test DataLoaders from a single directory.

    The dataset is split into training and testing sets based on the provided fraction.
    The expected directory structure is:
    data_dir/
    ├── img/ (or specified image_folder)
    │   ├── image1.jpg
    │   └── ...
    └── mask/ (or specified mask_folder)
        ├── image1.png (or .npy)
        └── ...

    Args:
        data_dir (str): Root directory containing the image and mask folders.
        image_folder (str, optional): Name of the subfolder for images. Defaults to 'img'.
        mask_folder (str, optional): Name of the subfolder for masks. Defaults to 'mask'.
        fraction (float, optional): Fraction of the dataset to allocate to the Test set.
                                    Defaults to 0.2 (i.e., 20% for Test, 80% for Train).
        batch_size (int, optional): Number of samples per batch. Defaults to 4.

    Returns:
        dict: A dictionary containing 'Train' and 'Test' PyTorch DataLoaders.
    """
    # Define standard transformations (convert to tensor).
    data_transforms = transforms.Compose([transforms.ToTensor()])
    
    # Create dataset instances for training and testing subsets
    # The SegmentationDataset is responsible for splitting the data based on 'subset' and 'fraction'.
    image_datasets = {
        x: SegmentationDataset(root=data_dir, # Root directory for the dataset
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100, # Seed for reproducible train/test split
                               fraction=fraction, # Fraction for the test set
                               subset=x, # Specifies 'Train' or 'Test' subset
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    
    # Create DataLoaders
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True if x == 'Train' else False, # Shuffle only for training
                      num_workers=8) # Number of worker processes for data loading
        for x in ['Train', 'Test']
    }
    return dataloaders
