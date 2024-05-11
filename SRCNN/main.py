import pathlib
import pickle

import matplotlib.pyplot as plt
import tqdm
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from super_resolution.src.sen2venus_dataset import (
    create_train_validation_test_split,
    default_patch_transform,
)

from super_resolution.src.visualization import plot_gallery

DATA_DIR = pathlib.Path("C:/Users/skouf/Documents/2024/STAT3007/stat3007-project/SRCNN/data")
SITES_DIR = DATA_DIR / "sites"
PREPROCESSING_DIR = DATA_DIR / "preprocessing"
RESULTS_DIR = DATA_DIR / "results"

train_patches, val_patches, test_patches = create_train_validation_test_split(
    str(SITES_DIR) + "\\", sites={"FGMANAUS"}
)


train_loader = DataLoader(train_patches, batch_size=2)
for i, (low_res, high_res) in enumerate(train_loader):
    if i == 2:
        break

low_res_example = low_res[0]
high_res_example = high_res[0]

low_res_example.shape

f1 = 9
n1 = 64
f2 = 1
n2 = 32
f3 = 5
c = 3
    


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale = nn.Sequential(nn.Conv2d(c, n1, f1),
                                     nn.ReLU(),
                                     nn.Conv2d(c,n2,f2),
                                     nn.Conv2d(c,n2,f3))
    def forward(self,x):
        return self.upscale(x)


CNN = SRCNN()


x_train, y_train = [], []

for i,(x,y) in enumerate(train_patches):
    x_train.append(x)
    y_train.append(y)

x_train = torch.cat(x_train,dim=0)
y_train = torch.cat(y_train,dim=0)

x_train.shape

nn.Conv2d(c, n1, f1)(x_train)