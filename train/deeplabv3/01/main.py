


from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
from torch.utils import data

import datahandler
from model import createDeepLabv3
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
              # adam에는 batch size 4였음
              type=int,
              help="Specify the batch size for the dataloader.")

def main(data_directory, exp_directory, epochs, batch_size):
    
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    # model = nn.DataParallel(model, device_ids = [1,2])
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function

    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.ADAM(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score ,

           'jaccard': JaccardIndex,
           'dice':Dice,
           'accuracy':BinaryAccuracy

        }

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'adam_deeplabv3_3mask_batch_16.pt')


if __name__ == "__main__":
    main()
