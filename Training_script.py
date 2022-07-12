import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
import albumentations as A
import sys
import time

from Simple_Unet import *
from Data import PipeDataset, get_names
from Training_functions import *

'''
This script is intended to train a model on the Pipes dataset.
Script arguments:
-IMG_PATH - path to the folder with images;
-MASK_PATH - path to the folder with masks;
-BATCH_SZ - size of minibatches for model training;
-LR - model learning rate;
-N_EPOCHS - number of training epochs;
-DICT_PATH - path where model's state dict will be saved.
Attention! Images and corresponding masks must have the same names, e. g. 123.png and 123.npy
'''


if __name__ == '__main__':
    IMG_PATH = 'data/images/'
    MASK_PATH = 'data/masks/'
    N_CLASSES = 7
    BATCH_SZ = 1
    LR = 0.001
    N_EPOCHS = 1
    DICT_PATH = 'test_dict.pth'
    df = get_names(IMG_PATH)
    model = UNet(num_class=N_CLASSES)
    device = torch.device('cpu')
    train,test = train_test_split(df['id'].values,test_size = 0.2, random_state = 1337)
    train_transform = A.Compose([A.OneOf([A.HorizontalFlip(),A.VerticalFlip(),A.RandomRotate90()],p=0.8),
                                 A.Perspective(p=0.7,scale=(0.07,0.12)),A.Blur(p=0.5,blur_limit=6),
                                 A.RandomBrightnessContrast((0,0.5),(0,0.5)),A.GaussNoise()])
    test_transform = A.Compose([A.OneOf([A.HorizontalFlip(),A.VerticalFlip(),A.RandomRotate90()],p=0.8),
                               A.GaussNoise()])
    train_set = PipeDataset(IMG_PATH,MASK_PATH,train,train_transform)
    test_set = PipeDataset(IMG_PATH,MASK_PATH,test,test_transform)
    train_loader = DataLoader(train_set, batch_size = BATCH_SZ, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = BATCH_SZ, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    start = time.perf_counter()
    history,best_model_dict = fit(N_EPOCHS, model, N_CLASSES, train_loader, test_loader, criterion, optimizer, device)
    end = time.perf_counter()
    print(f'Time elapsed: {end-start}')
    torch.save(best_model_dict,DICT_PATH)
