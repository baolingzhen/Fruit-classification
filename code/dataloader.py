from glob import glob
import numpy as np
import os
import torch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms


mean = torch.tensor([0.68397546, 0.5785919, 0.50372267])
std = torch.tensor([0.3033392, 0.3598722, 0.39136496])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}


def getDataLoader(args):
    dataset_dir = args.dataset
    batch_size = args.batch_size
    if args.subcommand == 'train':
        classes_num = len(os.listdir(os.path.join(dataset_dir)))
        split_rate = args.val_rate
        train_data = datasets.ImageFolder(dataset_dir, transform=data_transforms['train'])
        train_size = len(train_data)
        val_size = int(split_rate * train_size)
        train_ds, val_ds = random_split(train_data, [train_size - val_size, val_size])
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)

        return (train_dl, val_dl), (train_size, val_size, classes_num)

    if args.subcommand == 'eval':
        classes_num = len(os.listdir(os.path.join(dataset_dir)))
        test_ds = datasets.ImageFolder(dataset_dir, transform=data_transforms['test'])
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=3, pin_memory=True, shuffle=True)
        return test_ds, test_dl, classes_num
