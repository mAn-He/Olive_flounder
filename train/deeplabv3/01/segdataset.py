"""
Defines the custom PyTorch Dataset class for loading image-mask pairs
for semantic segmentation and applying transformations.
This version is typically used for '01' type datasets (binary segmentation).
"""
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
# import matplotlib.pyplot as plt # Not used in this script

class SegmentationDataset(VisionDataset):
    """
    A PyTorch Dataset for image segmentation tasks.
    
    The dataset loads images and their corresponding masks from specified folders.
    It supports splitting into training and testing subsets and applying
    torchvision transforms to both images and masks.
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
                 mask_color_mode: str = "grayscale") -> None: # Changed default mask_color_mode to grayscale
        """
        Initializes the SegmentationDataset.

        Args:
            root (str): Root directory path of the dataset.
            image_folder (str): Name of the folder containing images within the root directory.
            mask_folder (str): Name of the folder containing masks within the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes a
                sample (PIL Image) and returns a transformed version. Applied to both
                image and mask. Defaults to None.
            seed (int, optional): Seed for random number generator, used for reproducible
                train/test splits if `fraction` is specified. Defaults to None.
            fraction (float, optional): Fraction of the dataset to use for the 'Test' set
                (or 'Train' set if `subset` is 'Test' and fraction is for the other part).
                If None, the entire dataset is used. Defaults to None.
            subset (str, optional): Specifies 'Train' or 'Test' to select the appropriate
                subset if `fraction` is provided. If None, the entire dataset is used.
                Defaults to None.
            image_color_mode (str, optional): Color mode for images ('rgb' or 'grayscale').
                Defaults to "rgb".
            mask_color_mode (str, optional): Color mode for masks ('rgb' or 'grayscale').
                'grayscale' is typical for single-channel binary masks. Defaults to "grayscale".

        Raises:
            OSError: If `image_folder` or `mask_folder` doesn't exist in `root`.
            ValueError: If `subset` is invalid when `fraction` is specified.
            ValueError: If `image_color_mode` or `mask_color_mode` is invalid.
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        
        if not image_folder_path.exists():
            raise OSError(f"Image folder {image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"Mask folder {mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(f"Invalid image_color_mode: {image_color_mode}. Choose 'rgb' or 'grayscale'.")
        if mask_color_mode not in ["rgb", "grayscale"]: # Grayscale is typical for segmentation masks
            raise ValueError(f"Invalid mask_color_mode: {mask_color_mode}. Choose 'rgb' or 'grayscale'.")

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        # Load image and mask file names
        if not fraction: # Use all data if no fraction is specified
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else: # Split data into train/test subsets
            if subset not in ["Train", "Test"]:
                raise ValueError(f"Invalid subset: {subset}. Choose 'Train' or 'Test' when fraction is specified.")
            
            self.fraction = fraction
            # Ensure consistency in sorting before splitting
            all_image_names = sorted(image_folder_path.glob("*"))
            all_mask_names = sorted(mask_folder_path.glob("*"))

            if len(all_image_names) != len(all_mask_names):
                warnings.warn(f"Number of images ({len(all_image_names)}) and masks ({len(all_mask_names)}) do not match. This may lead to errors.")

            # Use numpy for shuffling with seed for reproducibility
            indices = np.arange(len(all_image_names))
            if seed:
                np.random.seed(seed)
                np.random.shuffle(indices)
            
            # Convert Path objects to lists for numpy array indexing
            self.image_list = np.array([str(p) for p in all_image_names])[indices]
            self.mask_list = np.array([str(p) for p in all_mask_names])[indices]
            
            split_index = int(np.ceil(len(self.image_list) * (1 - self.fraction)))
            
            if subset == "Train":
                self.image_names = [Path(p) for p in self.image_list[:split_index]]
                self.mask_names = [Path(p) for p in self.mask_list[:split_index]]
            else: # Test subset
                self.image_names = [Path(p) for p in self.image_list[split_index:]]
                self.mask_names = [Path(p) for p in self.mask_list[split_index:]]
        
        if len(self.image_names) != len(self.mask_names):
             warnings.warn(f"Mismatch after split in {subset} set: {len(self.image_names)} images vs {len(self.mask_names)} masks. Check data and naming.")


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves the image and mask at the given index and applies transforms.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the 'image' and 'mask'.
                  The image and mask are PIL Images if no transforms are applied,
                  otherwise they are transformed (e.g., to PyTorch tensors).
        """
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        
        # Open image and mask files
        # It's good practice to use 'with open(...)' but PIL's Image.open() handles file closing.
        image = Image.open(image_path)
        
        # Load mask: assumes .npy format for masks as per original code's np.load()
        # The original mask loading logic:
        # arr = np.load(mask_file)
        # mask_arr =np.transpose(arr,(1,2,0)).astype(np.uint8)*255
        # mask = Image.fromarray(mask_arr)
        # This implies the .npy file stores a (C, H, W) array, and it's transposed to (H, W, C).
        # For typical binary (01) or single-channel integer masks, this might be simpler.
        # Assuming mask is saved as a 2D array (H, W) or (H, W, 1) in .npy
        
        mask_data = np.load(mask_path)
        if mask_data.ndim == 3 and mask_data.shape[0] == 1: # (1, H, W)
            mask_data = np.squeeze(mask_data, axis=0)
        elif mask_data.ndim == 3 and mask_data.shape[2] == 1: # (H, W, 1)
             mask_data = np.squeeze(mask_data, axis=2)
        # Ensure mask is 2D (H, W) before converting to PIL
        
        # Convert numpy array to PIL Image. 
        # If mask_data represents class indices (e.g., 0 and 1), ensure it's scaled if needed for PIL 'L' mode.
        # For binary masks (0, 1), multiplying by 255 makes class 1 white.
        if np.max(mask_data) == 1: # Common for binary 0/1 masks
             mask_pil = Image.fromarray(mask_data.astype(np.uint8) * 255, mode='L')
        else: # Grayscale or other integer masks
             mask_pil = Image.fromarray(mask_data.astype(np.uint8), mode='L') # 'L' mode for grayscale

        # Apply color mode conversions
        if self.image_color_mode == "rgb":
            image = image.convert("RGB")
        elif self.image_color_mode == "grayscale":
            image = image.convert("L")

        if self.mask_color_mode == "rgb": # Usually not for segmentation masks
            mask_pil = mask_pil.convert("RGB")
        elif self.mask_color_mode == "grayscale": # Ensure it's 'L' mode if not already
            if mask_pil.mode != 'L':
                 mask_pil = mask_pil.convert("L")
        
        sample = {"image": image, "mask": mask_pil}
      
        # Apply transformations if any are provided
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample["mask"] = self.transforms(sample["mask"]) # Note: ToTensor on mask scales to [0,1]
            
        return sample
