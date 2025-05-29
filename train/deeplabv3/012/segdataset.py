"""
Defines the custom PyTorch Dataset class for loading image-mask pairs
for semantic segmentation, specifically tailored for '012' type datasets
(multi-class segmentation, typically 3 classes).
"""
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
# import matplotlib.pyplot as plt # Not used

class SegmentationDataset(VisionDataset):
    """
    A PyTorch Dataset for '012' type image segmentation tasks.
    
    Loads images and corresponding masks. Masks are expected to be .npy files
    where specific pixel values are remapped to new class indices (0, 1, 2).
    Supports splitting into training/testing sets and applying torchvision transforms.
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
                 mask_color_mode: str = "grayscale") -> None: # Defaulted to grayscale for masks
        """
        Initializes the SegmentationDataset for '012' data.

        Args:
            root (str): Root directory path.
            image_folder (str): Folder name for images.
            mask_folder (str): Folder name for masks.
            transforms (Optional[Callable], optional): Transforms for images and masks.
            seed (int, optional): Seed for train/test split.
            fraction (float, optional): Test set fraction.
            subset (str, optional): 'Train' or 'Test' subset.
            image_color_mode (str, optional): 'rgb' or 'grayscale' for images. Defaults to "rgb".
            mask_color_mode (str, optional): 'rgb' or 'grayscale' for masks. 
                                           Defaults to "grayscale". For '012' data, masks are
                                           typically processed into single-channel label maps.
        Raises:
            OSError: If image or mask folder doesn't exist.
            ValueError: For invalid subset, image_color_mode, or mask_color_mode.
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        
        if not image_folder_path.exists():
            raise OSError(f"Image folder {image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"Mask folder {mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(f"Invalid image_color_mode: {image_color_mode}.")
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(f"Invalid mask_color_mode: {mask_color_mode}.")

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode # Primarily affects PIL conversion, final mask is usually tensor

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise ValueError(f"Invalid subset: {subset}. Choose 'Train' or 'Test'.")
            
            self.fraction = fraction
            all_image_names = sorted(image_folder_path.glob("*"))
            all_mask_names = sorted(mask_folder_path.glob("*"))

            if len(all_image_names) != len(all_mask_names):
                warnings.warn(f"Number of images ({len(all_image_names)}) and masks ({len(all_mask_names)}) do not match.")

            indices = np.arange(len(all_image_names))
            if seed:
                np.random.seed(seed)
                np.random.shuffle(indices)
            
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
             warnings.warn(f"Mismatch after split in {subset} set: {len(self.image_names)} images vs {len(self.mask_names)} masks.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves the image and mask at the given index and applies transforms.
        Masks (.npy) are processed for '012' specific class remapping.
        """
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        
        image = Image.open(image_path)
        
        # Load mask from .npy file
        mask_numpy = np.load(mask_path) # Expected to be (C, H, W) or (H, W)

        # Specific '012' class remapping logic from original script:
        # Assumes mask_numpy is (C, H, W) where C=3 for raw channel-like masks.
        # Converts to a single channel label map where:
        # Original value 3 in channel 0 -> becomes class 1
        # Original value 1 in channel 1 -> becomes class 1 (Note: This seems to map two conditions to class 1)
        # Original value 2 in channel 2 -> becomes class 2 (This logic might need review for correctness based on dataset)
        # This will result in a mask that might not be (0, 1, 2) but rather (0, 1) if channel 2 is not present or value 2 is not there.
        # A more typical approach for '012' would be a single channel mask with values 0, 1, 2.
        # The original logic was:
        # arr[0] = np.where(arr[0] ==3, 1, 0)
        # arr[1] = np.where(arr[1] ==1, 1, 0)
        # arr[2] = np.where(arr[2] ==2, 1, 0) # Error here, should be class 2
        # mask_arr =np.transpose(arr,(1,2,0)).astype(np.uint8)*255 -> this creates an RGB like image.
        # For CrossEntropyLoss, the mask should be [H, W] with class indices.
        
        # Re-interpreting and simplifying mask processing for '012' target:
        # Assuming mask_numpy is a single channel [H,W] with values 0,1,2,3...
        # or a multi-channel where we need to derive a single label map.
        # If mask_numpy is already [H,W] with 0,1,2, then no complex processing is needed here before ToTensor.
        # The original code's transpose and scaling by 255 before Image.fromarray suggests it was trying
        # to visualize or handle a specific multi-channel format.
        # For direct use with CrossEntropyLoss, mask should be (H,W) LongTensor with class indices.
        
        if mask_numpy.ndim == 3:
            if mask_numpy.shape[0] == 3: # Assuming (3, H, W)
                # This is a common case if it's one-hot encoded like, but the remapping below is specific.
                # Original remapping logic:
                # class_0_mask = np.where(mask_numpy[0] == 3, 1, 0) # Class 1
                # class_1_mask = np.where(mask_numpy[1] == 1, 1, 0) # Class 1 again?
                # class_2_mask = np.where(mask_numpy[2] == 2, 2, 0) # Class 2 (corrected from original code's '1')
                # processed_mask = np.zeros_like(class_0_mask, dtype=np.uint8)
                # processed_mask[class_0_mask == 1] = 1
                # processed_mask[class_1_mask == 1] = 1 # Overwrites or combines with class_0_mask for class 1
                # processed_mask[class_2_mask == 2] = 2
                # For simplicity and common use cases, if mask is multi-channel, an argmax might be more standard
                # or the source .npy files should be single-channel label maps.
                # Given the ambiguity and potential error in original remapping,
                # we'll assume the .npy is directly a [H,W] label map or easily convertible.
                # If it's (C,H,W) and represents one-hot encoding or similar, an argmax might be needed.
                # For now, let's assume it's (H,W) or (1,H,W) or (H,W,1) and contains class labels directly.
                 if mask_numpy.shape[0] == 1: # (1, H, W)
                    mask_data = np.squeeze(mask_numpy, axis=0)
                 elif mask_numpy.shape[2] == 1: # (H, W, 1)
                    mask_data = np.squeeze(mask_numpy, axis=2)
                 else: # Fallback for (3,H,W) - this is not ideal, requires domain knowledge of the NPY structure
                    warnings.warn(f"Mask has shape {mask_numpy.shape}, taking channel 0 as label map. Verify this is correct for '012' data.")
                    mask_data = mask_numpy[0] # Example: take first channel, or apply custom logic
            elif mask_numpy.shape[2] == 3: # (H, W, 3)
                warnings.warn(f"Mask has shape {mask_numpy.shape}, taking channel 0 as label map. Verify this is correct for '012' data.")
                mask_data = mask_numpy[:,:,0] # Example
            else: # Other 3D shapes
                raise ValueError(f"Unsupported mask shape: {mask_numpy.shape}")

        elif mask_numpy.ndim == 2: # (H, W) - ideal
            mask_data = mask_numpy
        else:
            raise ValueError(f"Unsupported mask dimensions: {mask_numpy.ndim}")

        mask_pil = Image.fromarray(mask_data.astype(np.uint8)) # Mode 'L' will be inferred for 2D uint8

        if self.image_color_mode == "rgb":
            image = image.convert("RGB")
        elif self.image_color_mode == "grayscale":
            image = image.convert("L")

        # Mask is typically kept as single channel ('L' mode for PIL) for segmentation label maps
        if self.mask_color_mode == "rgb" and mask_pil.mode != 'RGB':
            mask_pil = mask_pil.convert("RGB") # Only if explicitly required, usually not for label maps
        elif self.mask_color_mode == "grayscale" and mask_pil.mode != 'L':
             mask_pil = mask_pil.convert("L")
        
        sample = {"image": image, "mask": mask_pil}
      
        if self.transforms:
            # Transforms (like ToTensor) should handle PIL Image to Tensor conversion.
            # For masks, ToTensor typically scales to [0,1]. For CrossEntropyLoss,
            # the target tensor should be LongTensor with class indices [0, N_classes-1] and shape [H,W].
            # This usually means the mask transform should be just ToTensor for PIL,
            # and then the mask tensor needs to be converted to Long type.
            # Or, a custom transform is needed for masks if they are not simple label maps.
            sample["image"] = self.transforms(sample["image"])
            # For masks, it's often better to transform to tensor without scaling, then convert to Long.
            # The default ToTensor scales to [0,1]. If mask values are class labels (0,1,2),
            # this scaling will change them.
            # A common approach:
            mask_tensor = torch.as_tensor(np.array(sample["mask"]), dtype=torch.long)
            sample["mask"] = mask_tensor
            # If self.transforms includes ToTensor(), it will scale.
            # If mask is already a label map (0,1,2,...), simple ToTensor might be okay if target is Float and criterion handles it,
            # but for CrossEntropyLoss, LongTensor target is standard.
            # The provided code uses self.transforms for mask, which implies ToTensor.
            # This might need adjustment in the trainer if masks are not LongTensor.
            # The previous trainer for 012 used masks.to(torch.long), so ToTensor output (float) will be cast.
            if not isinstance(sample["mask"], torch.Tensor): # If transform didn't make it a tensor
                 sample["mask"] = transforms.ToTensor()(sample["mask"]) # Fallback, but check scaling
            
        return sample
