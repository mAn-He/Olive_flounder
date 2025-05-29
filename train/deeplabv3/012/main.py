"""
Main training script for the DeepLabV3 model on '012' type datasets.

This script handles:
- Setting up the DeepLabV3 model.
- Loading and preparing the '012' dataset using `datahandler.py`.
- Defining the loss function (CrossEntropyLoss) and optimizer (Adam).
- Running the training loop using `trainer.py`.
- Saving the trained model.

Configuration is managed via command-line options using the `click` library.
"""
from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
from torch.utils import data

import datahandler
from model import create_deep_lab_v3 # Corrected from createDeepLabv3 for consistency
from trainer import train_model
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
import torch.nn as nn

import matplotlib.pyplot as plt 

@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory for '012' type dataset.",
              default=
            #   '/home/fisher/Peoples/hseung/NUBchi/Training' # Example of old path
              '/home/fisher/Peoples/hseung/NEW/Train' # Default path for '012' data
            )
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory to save outputs.",
              default = '/home/fisher/Peoples/hseung/NEW/1st_Trial/new_optim') # Example experiment directory
@click.option(
    "--epochs",
    default=30,
    type=int,
    help="Specify the number of epochs for training.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
def main(data_directory, exp_directory, epochs, batch_size):
    """
    Trains the DeepLabV3 model on a '012' type dataset.

    DATA_DIRECTORY: Path to the root of the training dataset. 
                    This directory should contain subfolders for images and masks
                    formatted for '012' type segmentation (typically 3 classes).
    EXP_DIRECTORY: Path to the directory where trained models and experiment
                   artifacts will be saved. This directory will be created if it
                   doesn't exist.
    EPOCHS: Number of complete passes through the training dataset.
    BATCH_SIZE: Number of training examples utilized in one iteration.
    """
    
    # Create DeepLabV3 model with ResNet101 backbone.
    # The model from `model.py` is expected to be pre-trained on COCO.
    # The output channels should match the number of classes for the '012' dataset (e.g., 3).
    model = create_deep_lab_v3(output_channels=3) # Assuming 3 classes for '012' dataset
    # model = nn.DataParallel(model, device_ids = [1,2]) # Optional: For multi-GPU training
    model.train() # Set the model to training mode

    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir(parents=True, exist_ok=True) # Added parents=True, exist_ok=True

    # Specify the loss function
    # CrossEntropyLoss is suitable for multi-class segmentation tasks.
    criterion = torch.nn.CrossEntropyLoss(reduction='mean') 
    
    # Specify the optimizer
    # Optimizer: Adam with a learning rate of 1e-4.
    optimizer = torch.optim.ADAM(model.parameters(), lr=1e-4)

    # Define metrics to be tracked during training.
    # Note: For multi-class, metrics like f1_score, JaccardIndex, Dice might need 'average' parameter.
    # Torchmetrics classes handle this internally based on their configuration.
    metrics = {'f1_score': f1_score , # sklearn.metrics.f1_score might need specific averaging for multi-class
               'jaccard': JaccardIndex(task="multiclass", num_classes=3), # Specify task and num_classes
               'dice': Dice(task="multiclass", num_classes=3),             # Specify task and num_classes
               'accuracy': BinaryAccuracy() # BinaryAccuracy might not be ideal for multi-class, consider MulticlassAccuracy
        }

    # Get data loaders for training and validation sets.
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size) # Ensure datahandler is appropriate for '012'
    
    # Start the training process.
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    base_path=exp_directory, # Directory to save checkpoints and logs
                    metrics=metrics,
                    num_epochs=epochs,
                    num_classes=3) # Pass num_classes to trainer if needed by metrics

    # Save the final trained model.
    # Consider a more descriptive name reflecting the '012' dataset.
    torch.save(model, exp_directory / 'deeplabv3_012_trained_model.pt')


if __name__ == "__main__":
    main()
