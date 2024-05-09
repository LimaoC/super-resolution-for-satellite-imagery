import pathlib
import pickle

import matplotlib.pyplot as plt
import tqdm
import torch
import py7zr
import os
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from super_resolution.src.sen2venus_dataset import (
    create_train_validation_test_split,
    default_patch_transform,
)
from super_resolution.src.visualization import plot_gallery

data = []
path = "C:/Users/skouf/Documents/2024/STAT3007/stat3007-project/SRCNN/FGMANAUS"
for file in os.listdir(path):
    data.append(torch.load(path + "/" + file))

data = torch.cat(data, dim = 0)

