import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


class CifarDataset(Dataset):
    def __init__(self, root='./data', transform=None, train=True):
        self.data = torchvision.datasets.CIFAR100(root=root,
                                                  train=train,
                                                  download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getclass__(self):
        return self.data.classes

    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        image = self.transform(image)
        return image, label

    def __showimg__(self, idx):
        npimg = self.data.data[idx]
        plt.imshow(npimg)
        plt.show()


def split_data(dataset, train_ratio=0.8):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset


def get_transform(select_transform=None):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transforms = []
    if select_transform:
        if 'RandomCrop' in select_transform:
            train_transforms.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
        if 'RandomHorizontalFlip' in select_transform:
            train_transforms.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in select_transform:
            train_transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if 'RandomRotation' in select_transform:
            train_transforms.append(transforms.RandomRotation(15))
        if 'RandomVerticalFlip' in select_transform:
            train_transforms.append(transforms.RandomVerticalFlip())
        if 'AutoAugment' in select_transform:
            train_transforms.append(transforms.AutoAugment())

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if select_transform:
        if 'Cutout' in select_transform:
            train_transforms.append(Cutout(n_holes=1, length=16))

    train_transform = transforms.Compose(train_transforms)

    test_transform = \
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return train_transform, test_transform


def get_datasets(root, select_transform, train_ratio, split=True):
    train_transform, test_transform = get_transform(select_transform)
    dataset = CifarDataset(root=root, transform=train_transform, train=True)
    test_dataset = CifarDataset(root=root, transform=test_transform, train=False)
    if split:
        train_dataset, valid_dataset = split_data(dataset, train_ratio=train_ratio)
        valid_dataset.transform = test_transform
        return train_dataset, valid_dataset, test_dataset
    else:
        return dataset, test_dataset


def get_dataloaders(root, select_transform, train_ratio, batch_size, num_workers, prefetch_factor, split=True):
    trainset, validset, testset = get_datasets(root, select_transform, train_ratio, split)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              prefetch_factor=prefetch_factor)
    test_loader = DataLoader(testset,
                             batch_size=batch_size * 2,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             prefetch_factor=prefetch_factor)
    if split:
        valid_loader = DataLoader(validset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  prefetch_factor=prefetch_factor)
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, None, test_loader


class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam