import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CifarDataset(root='./data', transform=transform, train=True)
testset = CifarDataset(root='./data', transform=transform, train=False)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)