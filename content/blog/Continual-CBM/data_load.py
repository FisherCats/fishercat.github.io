import torch
import os 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets, transforms, models
def get_dataset(dataset,preprocess):
    if dataset == "cifar100":
        data = datasets.CIFAR100(root=os.path.expanduser("~/my_data"), download=True, train=False, transform=preprocess)
    return data
def get_dataloder(dataset):
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)
    return data_loader
def get_cls_feat(cls):
    pass
def get_text_feat(path):
    pass

def main():
    pass