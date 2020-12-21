from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from linformer import Linformer
from PIL import Image
#from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#from vit_pytorch.efficient import ViT
from vit_pytorch import ViT

import dataset_zip

print(f"Torch: {torch.__version__}")

import datetime
#LOG_DIR = f"./tflog/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
#os.makedirs(LOG_DIR, True)

# Training settings
batch_size = 1
#batch_size = 64
#batch_size = 256
#batch_size = 512
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'
#device = 'cpu'

#os.makedirs('data', exist_ok=True)

#train_dir = 'data/train'
#test_dir = 'data/test'

#with zipfile.ZipFile('train.zip') as train_zip:
#    train_zip.extractall('data')
#    
#with zipfile.ZipFile('test.zip') as test_zip:
#    test_zip.extractall('data')

#train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
#test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

#print(f"Train Data: {len(train_list)}")
#print(f"Test Data: {len(test_list)}")

#labels = [path.split('/')[-1].split('.')[0] for path in train_list]

#train_list, valid_list = train_test_split(train_list, 
#                                          test_size=0.2,
#                                          stratify=labels,
#                                          random_state=seed)
#train_list, valid_list = torch.utils.data.random_split(train_list, [int(len(train_list) * 0.8), len(train_list)-int(len(train_list) * 0.8)])

#print(f"Train Data: {len(train_list)}")
#print(f"Validation Data: {len(valid_list)}")
#print(f"Test Data: {len(test_list)}")

#train_transforms = transforms.Compose(
#    [
#        transforms.Resize((224, 224)),
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#    ]
#)

#val_transforms = transforms.Compose(
#    [
#        transforms.Resize((224, 224)),
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#    ]
#)


#test_transforms = transforms.Compose(
#    [
#        transforms.Resize((224, 224)),
#        transforms.RandomResizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#    ]
#)

#class CatsDogsDataset(Dataset):
#    def __init__(self, file_list, transform=None):
#        self.file_list = file_list
#        self.transform = transform
#
#    def __len__(self):
#        self.filelength = len(self.file_list)
#        return self.filelength
#
#    def __getitem__(self, idx):
#        img_path = self.file_list[idx]
#        img = Image.open(img_path)
#        img_transformed = self.transform(img)
#
#        label = img_path.split("/")[-1].split(".")[0]
#        label = 1 if label == "dog" else 0
#
#        return img_transformed, label

import torchvision

trans = torchvision.transforms.ToTensor()
dataset = dataset_zip.DatasetZip('./data-zip/dataset-zip.csv', trans)

#train_data = CatsDogsDataset(train_list, transform=train_transforms)
#valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
#test_data = CatsDogsDataset(test_list, transform=test_transforms)

total_size = len(dataset)
labels = dataset.get_labels()
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=labels)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=True)

print(len(train_dataset), len(train_loader))

print(len(val_dataset), len(valid_loader))

#efficient_transformer = Linformer(
##    dim=128,
#    dim=768,
##    seq_len=49+1,  # 7x7 patches + 1 cls-token
#    seq_len=4+1,  # 2x2 patches + 1 cls-token
##    depth=1,
#    depth=12,
##    heads=8,
#    heads=12,
#    k=256
#)

# ViT-B
model = ViT(
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    image_size=32,
    patch_size=4,
    num_classes=8,
    channels=3,
).to(device)

## ViT-L
#model = ViT(
#    dim = 1024,
#    depth = 24,
#    heads = 16,
#    mlp_dim = 4096,
#    image_size=32,
#    patch_size=4,
#    num_classes=8,
#    channels=3,
#).to(device)

## ViT-H
#model = ViT(
#    dim = 1280,
#    depth = 32,
#    heads = 16,
#    mlp_dim = 5120,
#    image_size=32,
#    patch_size=4,
#    num_classes=8,
#    channels=3,
#).to(device)

#from torchsummary import summary
#summary(model, (3,32,32))

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#import os
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter(log_dir=LOG_DIR)

os.makedirs('./models', exist_ok=True)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    torch.save(model.to('cpu'), f'./models/model-H-{epoch:08}.pth')
    model = model.to(device)
#    writer.add_scalar("epoch_loss", epoch_loss, epoch)
#    writer.add_scalar("epoch_accuracy", epoch_accuracy, epoch)
#    writer.add_scalar("epoch_val_loss", epoch_loss, epoch)
#    writer.add_scalar("epoch_val_accuracy", epoch_accuracy, epoch)
