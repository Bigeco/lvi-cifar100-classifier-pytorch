import torch
import torch.nn as nn


class BaseConfig:
    def __init__(self,
                 epochs=2,
                 batch_size=64,
                 num_workers=2,
                 base_lr=0.001,
                 max_lr=0.001,
                 train_ratio=0.8,
                 grad_clip=0.01,
                 weight_decay=0.001,
                 dropout_rate=0.2,
                 transform=('RandomCrop',
                            'RandomHorizontalFlip'),
                 criterion=nn.CrossEntropyLoss(),
                 optimizer=torch.optim.Adam,
                 scheduler=None,
                 patience=10):
        self.data_root = '../data'
        self.epochs = epochs
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.train_ratio = train_ratio
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.transform = transform
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience


class ResNet9Config(BaseConfig):
    def __init__(self, epoch_ratio=(4, 4, 8, 8, 4)):
        super().__init__(batch_size=128)
        self.epoch_ratio = epoch_ratio


class ResNet18Config(BaseConfig):
    def __init__(self):
        super().__init__(batch_size=128)
