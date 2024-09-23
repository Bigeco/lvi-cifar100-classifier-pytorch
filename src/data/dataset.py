"""
이 dataset.py 는 데이터를 불러오는 것과 관련된 코드이다. 
주요 기능:
- [Class] CifarDataset: 이 클래스는 단순히 학습데이터와 테스트데이터 '객체'를 정의하기 위한 것이다. 
- [Function] load_data: 이 함수는 모델 학습에 필요한 데이터를 불러오는 함수이다. 
                        CifarDataset 객체를 정의하고 Train, Valid, Test 데이터셋을 리턴한다.

마지막 수정: 2023-09-23
"""

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


def load_data(batch_size, transform, num_workers=2, train_ratio=0.8, root='./data'): 
    """
    Load and prepare CIFAR100 dataset.
    
    Args:
        batch_size (int): Size of each batch.
        transform (tuple): A tuple of transformation functions to be applied to the data. 
                           (for data preprocessing or augmentation).
        num_workers (int): Number of subprocesses to use for data loading.
        train_ratio (float): Ratio of training set to full dataset.
        root (str): Root directory of dataset.
    
    Returns:
        tuple: (train_loader, valid_loader)
    """
    train_transform, test_transform = transform  # 입력받은 transform 에는 학습 데이터셋과 테스트 데이터셋에 대한 transform 이 존재한다.
    
    full_dataset = CifarDataset(root='./data', transform=train_transform, train=True) # 학습 데이터셋(50000개)에 대하여 객체를 생성한다. 
    
    dataset_size = len(full_dataset) # 길이를 출력하므로 50000 이 나온다.
    train_size = int(train_ratio * dataset_size) # 학습 데이터셋에서 Train 과 Valid 을 나누기 위하여 train_ratio(기본 0.2) 를 곱하고 정수형으로 변환한다. 결과는 40000.
    valid_size = dataset_size - train_size # 결과는 10000. 즉, Valid 용 데이터셋이 10000개 라는 뜻이다.
    
    trainset, validset = random_split(full_dataset, [train_size, valid_size]) # random_split 함수는 데이터셋을 나누긴 나누되 랜덤으로 분할하는 것이다. 
    testset = CifarDataset(root='./data', transform=test_transform, train=False) # 마찬가지로 테스트 데이터셋에 대한 객체를 정의한다.

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # 모델 학습을 위한 dataloader 를 정의하는 부분이다.
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2) # 모델 테스트를 위해서 dataloader 를 정의하는 부분이다.

    return train_loader, valid_loader, testloader # Train, Valid, Test 용에 대해서 각각 dataloader 를 리턴한다.
