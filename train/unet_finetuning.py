"""
Script for fine-tuning a UNet model for semantic segmentation.

This script defines a custom PyTorch Dataset (`UnetDatahandler`) for loading
image and mask data. It sets up a UNet model from the `segmentation-models-pytorch`
library, initializes it with ImageNet weights, and then fine-tunes it on a custom dataset.
The training loop includes data loading, model forward/backward passes, loss calculation
(BCEWithLogitsLoss), optimization (Adam), and an early stopping mechanism.
Hardcoded paths are used for data, model saving, and logging, which should be
parameterized in a production system.
"""
import segmentation_models_pytorch as smp
import numpy as np
import torch
import cv2 # Used for reading images
import os
import pickle
import matplotlib.pyplot as plt # Retained for potential debug plotting, though main plotting is commented
from torch.nn.functional import threshold, normalize # threshold, normalize are not explicitly used
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from early_stopping import EarlyStopping # Assumes early_stopping.py is available

from tqdm import tqdm

# global labels # 'labels' is loaded from pickle and used locally in __main__ scope, global keyword not needed

# Device configuration (hardcoded)
device = "cuda:1" # Consider making this configurable

# Hardcoded paths - for production, these should be arguments or config entries
label_data_path = "/home/fisher/DATA/GMISSION/annotations/annotation_v3.pkl" # For 'labels' which is not used by UnetDatahandler directly
train_data_path = "/home/fisher/Peoples/hseung/NUBchi/Training/img/"
mask_data_path = "/home/fisher/Peoples/hseung/NUBchi/Training/mask/"
model_save_path = "/home/fisher/Peoples/suyeon/Paper/Unet/Save_model/" # Renamed from model_path
log_path = "/home/fisher/Peoples/suyeon/Paper/Unet/log/"

# Ensure output directories exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# data_size = len(os.listdir(train_data_path)) # Calculated later in UnetDatahandler.__len__

class UnetDatahandler(Dataset):
    """
    Custom PyTorch Dataset for UNet fine-tuning.
    Loads images and corresponding .npy masks.
    """
    def __init__(self,
                 # label: list, # 'label' parameter was not used in __getitem__
                 train_data_path: str,
                 mask_data_path: str,
                 # batch_size: int, # batch_size is a DataLoader param, not Dataset
                 # model # model is not typically passed to a Dataset
                 transforms=None # Added transforms argument
                 ):
        """
        Initializes the UnetDatahandler dataset.

        Args:
            train_data_path (str): Path to the directory containing training images.
            mask_data_path (str): Path to the directory containing training masks (.npy files).
            transforms (Optional[Callable]): torchvision transforms to be applied to image and mask.
        """
        self.train_data_path = train_data_path
        self.train_data_filenames = sorted(os.listdir(self.train_data_path))
        # Filter out non-image files if necessary, assuming all are images for now
        self.filename_list = [t.split('.jpg')[0] for t in self.train_data_filenames if t.endswith('.jpg')]
        self.mask_data_path = mask_data_path
        # self.label_list = list(label.values()) # Not used
        self.transforms = transforms


    def __getitem__(self, idx: int):
        """
        Retrieves an image-mask pair by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing 'image' (Tensor) and 'mask' (Tensor).
                  'original_image' and 'original_size' are also included.
        """
        img_filename = self.filename_list[idx] + '.jpg'
        mask_filename = self.filename_list[idx] + '.npy'
        
        # Load image
        image_cv = cv2.imread(os.path.join(self.train_data_path, img_filename))
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # Convert BGR (cv2 default) to RGB
        
        # Load mask
        mask_np = np.load(os.path.join(self.mask_data_path, mask_filename))
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1 : # H,W,C format, take first channel
            mask_np = mask_np[:, :, 0]
        elif mask_np.ndim == 3 and mask_np.shape[0] == 1: # C,H,W format, take first channel
             mask_np = mask_np[0,:,:]
        # Ensure mask is 2D [H, W] and boolean/float {0,1} for BCEWithLogitsLoss target
        mask_np = mask_np.astype(np.float32) # Ensure correct type for loss
        if np.max(mask_np) > 1.0: # Normalize if mask is not 0/1
            mask_np = mask_np / 255.0 if np.max(mask_np) == 255 else mask_np / np.max(mask_np) # Basic normalization

        # Convert to PIL Images for torchvision transforms if they expect PIL
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(mask_np) # Mode 'F' for float32, or 'L' if converting to uint8

        if self.transforms:
            image_tensor = self.transforms(image_pil)
            mask_tensor = self.transforms(mask_pil)
            # Ensure mask is [1, H, W] or [H, W] as expected by BCEWithLogitsLoss (often [N, H, W] after batching)
            if mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1: # e.g. 3 channels from ToTensor on RGB
                 mask_tensor = mask_tensor[0, :, :].unsqueeze(0) # Take first channel and ensure [1,H,W]
            elif mask_tensor.ndim == 2: # H, W
                 mask_tensor = mask_tensor.unsqueeze(0) # Add channel dim [1,H,W]


        # For direct tensor conversion if no other transforms are needed:
        # image_numpy_HWC_to_CHW = image_rgb.transpose((2, 0, 1))
        # image_tensor = torch.from_numpy(image_numpy_HWC_to_CHW).float() / 255.0
        # mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) # Add channel dim: [1, H, W]

        return {
            'original_image': image_rgb, # Store original RGB numpy array
            'image': image_tensor, 
            'original_size': image_rgb.shape[:2], # H, W
            'mask': mask_tensor,
            'mask_size': mask_tensor.shape[1:] # H, W after transform
        }
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.filename_list)

# --- Main script execution ---
if __name__ == '__main__': # Encapsulate script logic
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=3, verbose=True, path=os.path.join(model_save_path, 'unet_early_stop_checkpoint.pth'))

    # Define the UNet model using segmentation_models_pytorch
    model = smp.Unet(
        encoder_weights='imagenet', # Use pre-trained weights for the encoder
        in_channels=3,              # Number of input image channels (RGB)
        classes=1                   # Number of output classes (binary segmentation)
    )

    # Load saved model state if continuing training or for inference (optional)
    # model.load_state_dict(torch.load(os.path.join(model_save_path,"epoch_0.pth"))) # Example to load a specific epoch

    model.to(device) # Move model to the configured device

    # Load labels (not directly used by UnetDatahandler as refactored, but kept if other logic needs it)
    # try:
    #     with open(label_data_path,"rb") as fr:
    #         labels = pickle.load(fr)
    # except FileNotFoundError:
    #     print(f"Warning: Label data file not found at {label_data_path}. Not used by current UnetDatahandler.")
    #     labels = {} # Provide an empty dict if not found

    # Hyperparameters
    batch_size = 32 # Consider making this an argument
    learning_rate = 0.0001 # Consider making this an argument
    epochs = 50 # Consider making this an argument

    # Define transformations - basic ToTensor for now
    # For more advanced augmentation, add them here.
    data_transforms = transforms.Compose([
        transforms.ToTensor() # Converts PIL Image to tensor and scales to [0,1]
    ])
    
    # Create dataset instance
    # Removed unused 'label', 'batch_size', 'model' arguments from UnetDatahandler constructor
    dataset = UnetDatahandler(
        train_data_path=train_data_path, 
        mask_data_path=mask_data_path,
        transforms=data_transforms
    )

    # Split dataset into training and validation
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    print(f"Total dataset size: {dataset_size}")
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Ensure all model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss() # Suitable for binary segmentation (output is raw logits)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    best_val_loss = float('inf') # Changed from best_loss to avoid confusion with training loss variable

    # Training loop
    for epoch in tqdm(range(epochs), desc='Epoch Loop', position=0):
        model.train() # Set model to training mode
        epoch_train_loss = 0.0
        
        for batch_idx, batched_inputs in tqdm(enumerate(train_loader), desc='Training Batch', position=1, leave=False, total=len(train_loader)):
            input_image = batched_inputs['image'].to(device, dtype=torch.float)
            input_mask = batched_inputs['mask'].to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            outputs = model(input_image) # Model output is expected to be [N, 1, H, W] for classes=1
            
            # Squeeze channel dimension if model output is [N,1,H,W] and target is [N,H,W] or vice-versa for BCEWithLogitsLoss
            if outputs.shape != input_mask.shape:
                if outputs.ndim == 4 and outputs.shape[1] == 1 and input_mask.ndim == 3 : # Output: N,1,H,W, Target: N,H,W
                     outputs = outputs.squeeze(1)
                elif input_mask.ndim == 4 and input_mask.shape[1] == 1 and outputs.ndim == 3: # Target: N,1,H,W, Output: N,H,W
                     input_mask = input_mask.squeeze(1)
                # Add more shape adjustments if necessary or ensure datahandler provides them perfectly matched

            loss = criterion(outputs, input_mask)
            
            # Logging training loss (simplified)
            if (batch_idx % 50 == 0) or (batch_idx == len(train_loader) -1) : # Log every 50 batches and last batch
                with open(os.path.join(log_path, "training_log.txt"), 'a') as log_file:
                    log_file.write(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}\n")
            
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_train_loss:.4f}")
            
        # Validation phase
        model.eval() # Set model to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculations for validation
            for batched_inputs in tqdm(validation_dataloader, desc='Validation Batch', position=1, leave=False):
                input_image = batched_inputs['image'].to(device, dtype=torch.float)
                input_mask = batched_inputs['mask'].to(device, dtype=torch.float)
                
                outputs = model(input_image)
                if outputs.shape != input_mask.shape: # Adjust shapes like in training
                    if outputs.ndim == 4 and outputs.shape[1] == 1 and input_mask.ndim == 3 :
                         outputs = outputs.squeeze(1)
                    elif input_mask.ndim == 4 and input_mask.shape[1] == 1 and outputs.ndim == 3:
                         input_mask = input_mask.squeeze(1)

                loss = criterion(outputs, input_mask)
                epoch_val_loss += loss.item()
        
        avg_epoch_val_loss = epoch_val_loss / len(validation_dataloader)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_epoch_val_loss:.4f}")
        with open(os.path.join(log_path, "validation_log.txt"), 'a') as log_file:
             log_file.write(f"Epoch {epoch+1}, Validation Loss: {avg_epoch_val_loss:.4f}\n")

        # Early stopping and model saving
        early_stopping(avg_epoch_val_loss, model) # Pass validation loss to early stopping
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        # Save model if validation loss improved (or at last epoch)
        if avg_epoch_val_loss < best_val_loss or epoch == epochs - 1:
            best_val_loss = avg_epoch_val_loss
            # Save the model (consider saving state_dict for more flexibility)
            torch.save(model.state_dict(), os.path.join(model_save_path, f"unet_epoch_{epoch+1}_val_loss_{best_val_loss:.4f}.pth"))
            print(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

    print("Training finished.")
    # Load best model if early stopping occurred and saved a checkpoint
    if early_stopping.early_stop:
        print("Loading best model from early stopping checkpoint.")
        model.load_state_dict(torch.load(early_stopping.path)) # Load the best model
    elif epochs > 0 : # ensure epochs were run
        # If no early stopping, the last saved model (or best_val_loss model) is already potentially the best.
        # Or, explicitly load the overall best model if tracked separately from early_stopping's best.
        # For now, assumes the model in memory (if not early stopped) or from early_stopping.path is the one to use.
        print("Using model from the last epoch or best validation checkpoint.")

    print("Fine-tuning process completed.")
