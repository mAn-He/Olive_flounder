import copy
import csv
import os
import time
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
# from torch import randint, tensor # randint, tensor not used
import torch.nn.functional as F # F not used
import warnings
import itertools

def train_model(model, criterion, dataloaders, optimizer, metrics, base_path,
                num_epochs, num_classes=1): # Added num_classes for Jaccard/Dice
    """
    Trains and validates a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): The loss function.
        dataloaders (dict): A dictionary containing 'Train' and 'Test' DataLoaders.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        metrics (dict): A dictionary of evaluation metrics. Keys are metric names (str)
                        and values are callable metric functions.
                        For sklearn metrics, they should accept (y_true, y_pred).
                        For torchmetrics, they should be initialized instances.
        base_path (pathlib.Path): The base directory to save logs and model checkpoints.
        num_epochs (int): The total number of epochs to train for.
        num_classes (int, optional): The number of classes for multi-class metrics.
                                     Defaults to 1 (binary case often).

    Returns:
        torch.nn.Module: The model with the best weights loaded.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    warnings.filterwarnings(action='ignore') # Consider moving to the main script if it's a global setting

    best_loss = 1e10  # Initialize best_loss to a very high value
    
    # Determine device and move model
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare fieldnames for CSV logging
    # Dynamically create fieldnames for each class/metric if metrics can be multi-output
    # Assuming metrics return a single value or an array that will be averaged.
    # For simplicity, keeping the original structure of _0 suffix if only one value per metric.
    # If metrics return per-class values, this part might need adjustment based on metric output.
    # The original code had hardcoded range(0,1), implying single value metrics or averaging.
    # For 'jaccard', 'dice', 'accuracy' from torchmetrics, they are usually single values after .compute()
    # or can be configured for per-class.
    
    # Simplified fieldnames, assuming metrics will be averaged if they produce multiple values
    metric_fieldnames = []
    for phase in ['Train', 'Test']:
        for name in metrics.keys():
            metric_fieldnames.append(f'{phase}_{name}') # Assuming single value per metric after processing

    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + metric_fieldnames
                      
    # Initialize the log file
    log_filename = os.path.join(base_path, 'training_log_deeplabv3_01.csv') # More descriptive name
    with open(log_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        
        # Initialize dictionary to store batch summaries
        batch_summary = {key: [] for key in fieldnames} # Use list to append metric values

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Initialize lists for metric accumulation for the current phase
            phase_metrics = {name: [] for name in metrics.keys()}

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase]), desc=f"{phase} Epoch {epoch}"):
                inputs = sample['image'].to(device, dtype=torch.float32) # Ensure float for model
                masks = sample["mask"].to(device, dtype=torch.float32)   # Ensure float for criterion like MSE

                optimizer.zero_grad() # Zero the parameter gradients

                # Forward pass
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    # Ensure outputs['out'] matches mask dimension and type for loss calculation
                    # For MSELoss with single channel output, might need to squeeze channel dim if present
                    output_for_loss = outputs['out']
                    if output_for_loss.shape != masks.shape:
                         # This could happen if model outputs [N, C, H, W] and mask is [N, H, W] or [N, 1, H, W]
                         # Adjusting for common case: if mask is [N, H, W], unsqueeze it.
                         if len(masks.shape) == 3 and masks.shape[0] == output_for_loss.shape[0] and \
                            masks.shape[1] == output_for_loss.shape[2] and masks.shape[2] == output_for_loss.shape[3]:
                             masks_for_loss = masks.unsqueeze(1)
                         else: # If shapes still don't match, print warning, but proceed. This might lead to errors.
                             print(f"Warning: Output shape {output_for_loss.shape} and Mask shape {masks.shape} mismatch. Trying to proceed.")
                             masks_for_loss = masks
                    else:
                        masks_for_loss = masks

                    loss = criterion(output_for_loss, masks_for_loss)

                    # Prepare predictions and ground truth for metric calculation
                    # For binary segmentation, typically sigmoid is applied then threshold.
                    # If criterion is MSELoss, outputs['out'] might be direct values.
                    # For metrics, usually a binary prediction (0 or 1) is needed.
                    # Assuming output_for_loss contains logits or probabilities for the positive class.
                    # The original code used >0.8 or >0.1 for different metrics, which is unusual.
                    # Standardizing to >0.5 after sigmoid (if logits) or direct if probabilities.
                    # If model output is logits, apply sigmoid. If probabilities, use directly.
                    # For MSE, outputs are likely direct values. Thresholding at 0.5 for binary.
                    
                    # For metric calculation, ensure binary format [N, H, W] or [N, 1, H, W]
                    # Assuming outputs['out'] is [N, C, H, W] where C=1 for binary segmentation with MSE.
                    # If C > 1, this needs adjustment for multi-class.
                    # For '01' dataset, C is likely 1.
                    
                    y_pred_binary = (outputs['out'].detach() > 0.5).cpu() # Thresholding for binary metrics
                    y_true_binary = (masks_for_loss.detach() > 0.5).cpu()    # Assuming masks are 0 or 1

                    # Calculate metrics
                    for name, metric_fn in metrics.items():
                        if name == 'f1_score': # sklearn metric
                            # Flatten for sklearn's f1_score
                            phase_metrics[name].append(metric_fn(y_true_binary.numpy().ravel(), 
                                                                 y_pred_binary.numpy().ravel(), 
                                                                 average='binary', zero_division=0))
                        elif isinstance(metric_fn, (Dice, JaccardIndex, BinaryAccuracy)): # torchmetrics
                            metric_instance = metric_fn.to(device) # Ensure metric is on the same device
                            # Ensure y_pred_binary and y_true_binary are suitable for torchmetrics
                            # Typically [N, C, H, W] or [N, H, W] for binary tasks.
                            # Torchmetrics classes will handle aggregation.
                            metric_instance.update(y_pred_binary.to(device), y_true_binary.to(device))
                            # Metric values will be aggregated and computed at the end of epoch for torchmetrics

                    # Backward pass + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            batch_summary[f'{phase}_loss'] = epoch_loss
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Compute and store epoch metrics for the phase
            for name, metric_fn in metrics.items():
                if isinstance(metric_fn, (Dice, JaccardIndex, BinaryAccuracy)): # torchmetrics
                    metric_instance = metric_fn.to(device) # Get the instance
                    batch_summary[f'{phase}_{name}'] = metric_instance.compute().item()
                    metric_instance.reset() # Reset for next epoch/phase
                elif name == 'f1_score': # For sklearn metrics, average collected scores
                     if phase_metrics[name]: # Ensure list is not empty
                        batch_summary[f'{phase}_{name}'] = np.mean(phase_metrics[name])
                     else:
                        batch_summary[f'{phase}_{name}'] = 0.0 # Or NaN, or handle as appropriate

            # Deep copy the model if it's the best encountered so far (based on validation loss)
            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(base_path, f'best_model_epoch_{epoch}.pt'))
                print(f"Best model saved at epoch {epoch} with validation loss: {best_loss:.4f}")

        # Log epoch summary
        batch_summary['epoch'] = epoch
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({k: batch_summary.get(k, '') for k in fieldnames}) # Handle potentially missing metric keys

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Lowest Validation Loss: {best_loss:4f}')

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model
