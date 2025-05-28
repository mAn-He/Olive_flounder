import segmentation_models_pytorch as smp
import numpy as np
import torch
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from torch.nn.functional import threshold, normalize
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from early_stopping import EarlyStopping

from tqdm import tqdm

global labels

device = "cuda:1"

# load data
label_data_path = "/home/fisher/DATA/GMISSION/annotations/annotation_v3.pkl"
train_data_path = "/home/fisher/Peoples/hseung/NUBchi/Training/img/"
mask_data_path = "/home/fisher/Peoples/hseung/NUBchi/Training/mask/"
model_path = "/home/fisher/Peoples/suyeon/Paper/Unet/Save_model/"
log_path = "/home/fisher/Peoples/suyeon/Paper/Unet/log/"

data_size = len(os.listdir(train_data_path))

class UnetDatahandler(Dataset):
    def __init__(self,
                 label: list,
                 train_data_path: str,
                 mask_data_path: str,
                 batch_size: int,
                 model
                 ):
        
        self.train_data_path = train_data_path
        self.train_data_filenames = sorted(os.listdir(self.train_data_path))
        self.filename_list = [t.split('.jpg')[0] for t in self.train_data_filenames if True]
        self.mask_data_path = mask_data_path
        
        self.label_list = list(label.values())
        
        self.batch_size = batch_size
        self.model = model
        self.train_dataset = []
    
    def __getitem__(self, idx):
            
        train_data = {}
        
        # Import image
        # filename = self.label_list[idx]['filename']
        image = cv2.imread(self.train_data_path + self.filename_list[idx] + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_numpy_HWC_to_CHW = image.transpose((2, 0, 1))
        
        # Convert NumPy array to PyTorch tensor
        image_tensor = torch.from_numpy(image_numpy_HWC_to_CHW).float()

        # Normalize the image (optional)
        image_tensor /= 255.0
        
        # Create batched_input
        train_data['original_image'] = image
        train_data['image'] = image_tensor
        train_data['original_size'] = image.shape[:2]
        
        # Create batched mask input data
        mask = np.load(self.mask_data_path + self.filename_list[idx] + '.npy')
        mask = mask[:, :, 0]
        
        train_data['mask'] = mask
        train_data['mask_size'] = mask.shape[:2]

        return train_data
    
    def __len__(self):
        print(f'Batched input length: {self.batch_size}')
        print(f'dataset size: {len(os.listdir(self.train_data_path))}')
        
        return len(os.listdir(self.train_data_path))

early_stopping = EarlyStopping(patience = 3, verbose = True)

model = smp.Unet(encoder_weights = 'imagenet',
                 in_channels=3,            # Number of input image channels (e.g., 3 for RGB image)
                 classes=1)

# Load saved model
model.load_state_dict(torch.load("/home/fisher/Peoples/suyeon/Paper/Unet/Save_model/epoch_0.pth"))

# Move model parameters to device
model.to(device)

with open(label_data_path,"rb") as fr:
            labels = pickle.load(fr)

label_list = list(labels.values())        

batch_size = 32

dataset = UnetDatahandler(label=labels, 
                          train_data_path = train_data_path, 
                          mask_data_path = mask_data_path,
                          batch_size = batch_size, 
                          model=model)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
validation_size = dataset_size - train_size

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(validation_dataset)}")

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=False)

validation_dataloader = DataLoader(dataset=validation_dataset,
                          batch_size=batch_size,
                          shuffle=False)

for param in model.parameters():
    param.requires_grad = True


threshold = 0.5
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
best_loss = 99999999
epochs = 50

for epoch in tqdm(range(epochs), desc='outer', position=0):
    # Train
    model.train()
    epoch_loss = 0
    
    for idx, batched_inputs in tqdm(enumerate(train_loader), desc='inner', position=1, leave=False):
        input_image = batched_inputs['image']
        input_mask = batched_inputs['mask']
        
        input_image = input_image.to(torch.float)
        input_mask = input_mask.to(torch.float)
        
        input_image = input_image.to(device)
        input_mask = input_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_image)
        outputs = outputs.squeeze()
                
        loss = criterion(outputs, input_mask)
        log_file = open(f"{log_path}log.txt", 'w')
        log_file.write(str(loss)+'\n')
        log_file.close()
        epoch_loss = epoch_loss + loss
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for idx, batched_inputs in enumerate(validation_dataloader):
            input_image = batched_inputs['image']
            input_mask = batched_inputs['mask']
            
            input_image = input_image.to(torch.float)
            input_mask = input_mask.to(torch.float)
            
            input_image = input_image.to(device)
            input_mask = input_mask.to(device)
            
            outputs = model(input_image)
            outputs = outputs.squeeze() 
            
            loss = criterion(outputs, input_mask)
            val_loss += loss.item()
    
    ### Part to check for early stopping ###
    early_stopping(val_loss, model) # Track current overfitting status
    
    if early_stopping.early_stop: # Early stop if condition is met
        break
    
    epoch_mean_loss = epoch_loss/(data_size/batch_size)
    print(f"Epoch {epoch} Mean Loss = {epoch_mean_loss}")
    
    if best_loss > epoch_mean_loss or epoch == epochs-1:
        best_loss = epoch_mean_loss
        torch.save(model, model_path + "epoch_" + str(epoch+1)+".pth")

