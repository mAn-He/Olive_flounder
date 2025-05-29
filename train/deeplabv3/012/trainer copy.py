import copy
import csv
import os
import time
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
# from torch import randint, tensor # Unused
# import torch.nn.functional as F # Unused
import warnings
import itertools

def train_model(model, criterion, dataloaders, optimizer, metrics, base_path,
                num_epochs, num_classes=3): # Default num_classes to 3 for '012' datasets
    """
    Trains and validates a PyTorch model, suited for '012' type datasets.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        dataloaders (dict): A dictionary containing 'Train' and 'Test' DataLoaders.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        metrics (dict): A dictionary of evaluation metrics. Keys are metric names (str)
                        and values are callable metric functions.
                        For sklearn metrics, they should accept (y_true, y_pred).
                        For torchmetrics, they should be initialized instances configured
                        for multi-class if appropriate.
        base_path (pathlib.Path): The base directory to save logs and model checkpoints.
        num_epochs (int): The total number of epochs to train for.
        num_classes (int, optional): The number of classes for multi-class metrics.
                                     Defaults to 3 for '012' type datasets.

    Returns:
        torch.nn.Module: The model with the best weights loaded.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    warnings.filterwarnings(action='ignore') 

    best_loss = 1e10  # Initialize best_loss to a very high value
    
    # Determine device and move model
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # Consider making device configurable
    model.to(device)
    
    # Prepare fieldnames for CSV logging
    metric_fieldnames = []
    for phase in ['Train', 'Test']:
        for name in metrics.keys():
            metric_fieldnames.append(f'{phase}_{name}')

    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + metric_fieldnames
                      
    # Initialize the log file
    log_filename = os.path.join(base_path, 'training_log_deeplabv3_012.csv') # Descriptive name for '012'
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        
        batch_summary = {key: [] for key in fieldnames} 

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            phase_metrics = {name: [] for name in metrics.keys()} # To collect metric values for averaging (for sklearn)

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase]), desc=f"{phase} Epoch {epoch}"):
                inputs = sample['image'].to(device, dtype=torch.float32)
                # For CrossEntropyLoss, masks should be LongTensor and shape [N, H, W]
                masks = sample["mask"].to(device, dtype=torch.long) 

                optimizer.zero_grad() 

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs) # Expected output shape [N, C, H, W] where C is num_classes
                    output_for_loss = outputs['out']
                    
                    # Ensure mask has the correct shape for CrossEntropyLoss if it's [N, 1, H, W]
                    if len(masks.shape) == 4 and masks.shape[1] == 1:
                        masks_for_loss = masks.squeeze(1)
                    else:
                        masks_for_loss = masks
                    
                    loss = criterion(output_for_loss, masks_for_loss)

                    # For metrics, predictions often need to be class indices [N, H, W]
                    y_pred_indices = torch.argmax(output_for_loss.detach(), dim=1).cpu()
                    y_true_indices = masks_for_loss.detach().cpu()

                    for name, metric_fn in metrics.items():
                        if name == 'f1_score': # sklearn metric
                            phase_metrics[name].append(metric_fn(y_true_indices.numpy().ravel(), 
                                                                 y_pred_indices.numpy().ravel(), 
                                                                 average='macro', zero_division=0, labels=list(range(num_classes))))
                        elif isinstance(metric_fn, (Dice, JaccardIndex)): # torchmetrics for multi-class
                            metric_instance = metric_fn.to(device) 
                            metric_instance.update(y_pred_indices.to(device), y_true_indices.to(device))
                            # Aggregated at epoch end
                        elif isinstance(metric_fn, BinaryAccuracy) and num_classes > 2: # Handle BinaryAccuracy for multi-class case
                            # This interpretation might need adjustment based on desired behavior for "accuracy" in multi-class
                            # For overall pixel accuracy, consider torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                            print(f"Warning: Using BinaryAccuracy for multi-class task ({num_classes} classes). This might not be the intended behavior.")
                            metric_instance = metric_fn.to(device)
                            # One-hot encode or pick a primary class for binary accuracy if that's intended.
                            # Here, we'll compare class 1 presence as an example, if applicable.
                            # This part is tricky for generic multi-class with BinaryAccuracy.
                            # For simplicity, if it's BinaryAccuracy, it will likely error or give misleading results for num_classes > 2
                            # unless the metric_fn itself is pre-configured (e.g. with a specific class index).
                            # The original code for '01' dataset used >0.5 threshold, which is for binary.
                            # For '012' (multi-class), directly using BinaryAccuracy is problematic.
                            # We will assume it's pre-configured if it's BinaryAccuracy.
                            metric_instance.update(y_pred_indices.to(device), y_true_indices.to(device))


                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            batch_summary[f'{phase}_loss'] = epoch_loss
            print(f'{phase} Loss: {epoch_loss:.4f}')

            for name, metric_fn in metrics.items():
                if isinstance(metric_fn, (Dice, JaccardIndex, BinaryAccuracy)): # torchmetrics
                    metric_instance = metric_fn.to(device)
                    try:
                        computed_metric = metric_instance.compute().item()
                    except Exception as e: # Catch potential errors if metric state is weird (e.g. BinaryAccuracy on multi-class without proper config)
                        print(f"Could not compute torchmetric {name}: {e}")
                        computed_metric = 0.0 # Default value
                    batch_summary[f'{phase}_{name}'] = computed_metric
                    metric_instance.reset() 
                elif name == 'f1_score': 
                     if phase_metrics[name]:
                        batch_summary[f'{phase}_{name}'] = np.mean(phase_metrics[name])
                     else:
                        batch_summary[f'{phase}_{name}'] = 0.0

            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(base_path, f'best_model_epoch_{epoch}_012.pt')) # More descriptive
                print(f"Best model for '012' saved at epoch {epoch} with validation loss: {best_loss:.4f}")

        batch_summary['epoch'] = epoch
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({k: batch_summary.get(k, '') for k in fieldnames})

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Lowest Validation Loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)
    return model
