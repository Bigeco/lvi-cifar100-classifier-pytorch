import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np

class CifarDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.data = torchvision.datasets.CIFAR100(root='./data', 
                                                  train=train, 
                                                  download=False,
                                                  transform=transform)

    def __len__(self):
        return len(self.data)

    def __getclass__(self):
        return self.data.classes
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    def __showimg__(self, idx):
        npimg = self.data.data[idx]
        plt.imshow(npimg)
        plt.show() 


def load_data(batch_size, num_workers=2, train_ratio=0.8, root='./data'):
    """
    Load and prepare CIFAR100 dataset.
    
    Args:
        batch_size (int): Size of each batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train_ratio (float): Ratio of training set to full dataset.
        root (str): Root directory of dataset.
    
    Returns:
        tuple: (train_loader, valid_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = CifarDataset(root='./data', transform=transform, train=True)
    
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    valid_size = dataset_size - train_size
    
    trainset, validset = random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CifarDataset(root='./data', transform=transform, train=True)
    testset = CifarDataset(root='./data', transform=transform, train=False)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)