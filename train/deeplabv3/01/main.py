"""
Main training script for the DeepLabV3 model on '01' type datasets.

This script handles:
- Setting up the DeepLabV3 model (optionally pre-trained).
- Loading and preparing the dataset using `datahandler.py`.
- Defining the loss function (MSELoss) and optimizer (Adam).
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
from model import create_deep_lab_v3 # Corrected import name
from trainer import train_model
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
import torch.nn as nn

import matplotlib.pyplot as plt 

@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.",
              default=
            #   '/home/fisher/Peoples/hseung/NUBchi/Training'
              '/home/fisher/Peoples/hseung/NEW/Train'
            )
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.",
              default = '/home/fisher/Peoples/hseung/NEW/1st_Trial/new_optim')
@click.option(
    "--epochs",
    default=30,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              # For Adam, batch size was 4
              type=int,
              help="Specify the batch size for the dataloader.")
def main(data_directory, exp_directory, epochs, batch_size):
    """
    Trains the DeepLabV3 model on a specified dataset.

    DATA_DIRECTORY: Path to the root of the training dataset. 
                    This directory should contain subfolders for images and masks.
    EXP_DIRECTORY: Path to the directory where trained models and experiment
                   artifacts will be saved. This directory will be created if it
                   doesn't exist.
    EPOCHS: Number of complete passes through the training dataset.
    BATCH_SIZE: Number of training examples utilized in one iteration.
    """
    
    # Create DeepLabV3 model with ResNet101 backbone, pre-trained on COCO.
    model = create_deep_lab_v3() 
    # model = nn.DataParallel(model, device_ids = [1,2]) # Optional: For multi-GPU training
    model.train() # Set the model to training mode

    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    # Using Mean Squared Error Loss for this segmentation task.
    # Note: CrossEntropyLoss or DiceLoss are more common for segmentation.
    criterion = torch.nn.MSELoss(reduction='mean') 
    
    # Specify the optimizer
    # Optimizer: Adam with a learning rate of 1e-4.
    optimizer = torch.optim.ADAM(model.parameters(), lr=1e-4)

    # Define metrics to be tracked during training.
    metrics = {'f1_score': f1_score ,
               'jaccard': JaccardIndex, # Using torchmetrics JaccardIndex
               'dice': Dice,             # Using torchmetrics Dice
               'accuracy': BinaryAccuracy # Using torchmetrics BinaryAccuracy
        }

    # Get data loaders for training and validation sets.
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    
    # Start the training process.
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    base_path=exp_directory, # Directory to save checkpoints and logs
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the final trained model.
    torch.save(model, exp_directory / 'adam_deeplabv3_3mask_batch_16.pt') # Consider a more descriptive name


if __name__ == "__main__":
    main()
