import copy
import csv
import os
import time
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
from torch import randint, tensor
import torch.nn.functional as F
import warnings
import itertools

def train_model(model, criterion, dataloaders, optimizer, metrics, base_path,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    warnings.filterwarnings(action='ignore')

    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    # model = nn.DataParallel(model, device_ids = [0, 1])
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}_{n}' for m,n in itertools.product(metrics.keys(),range(0,1))] + \
        [f'Test_{m}_{n}' for m,n in itertools.product(metrics.keys(),range(0,1))]
    with open(os.path.join(base_path, 'sgd_deeplabv3_3mask_batch_16.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):

                inputs = sample['image'].to(device)
                masks = sample["mask"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
      
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)

                    loss = criterion(outputs['out'], masks)

                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    
    
                    for name, metric in metrics.items():
                        if name == 'f1_score':

                            # 4 is total
                            batchsummary[f'{phase}_{name}_0'].append(
                                metric(y_true > 0, y_pred > 0.8,average='binary'))
                            # 5 is the mean value I manually created
                        elif name =='jaccard':
      

                            jaccard = metric(task='binary').to(device)
                            
                            pred = outputs['out']>0.8
                            answer = masks>0
                  
                            jaccard_total = jaccard(answer, pred).cpu().numpy().ravel()
                            batchsummary[f'{phase}_{name}_0'].append(jaccard_total[n]
                                )                           

                        elif name =='dice':            
                        # else: /
                            dice= metric(average='micro'
                                        #  ,num_classes=3,multiclass=True
                                         ).to(device)
          
                            pred = outputs['out']>0.1
                            answer = masks>0
                            
                                      
                            dice_total = dice(answer, pred).cpu().numpy().ravel()
                            batchsummary[f'{phase}_{name}_0'].append(dice_total[n]
                                )                           
                                     
                        else: 
                            accuracy= metric(
                                        #  ,num_classes=3,multiclass=True
                                         ).to(device)
                     
                            pred = outputs['out']>0.1
                            answer = masks>0
                            
                     
                            accuracy_total = accuracy(answer, pred).cpu().numpy().ravel()
                            batchsummary[f'{phase}_{name}_0'].append(accuracy_total[n]
                                )                           
                   
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        # print(batchsummary)
        with open(os.path.join(base_path, 'sgd_deeplabv3_3mask_batch_16.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
