import torch

CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'num_epochs': 25,
    'dropout_rate': 0.2,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers': 2,
    'train_ratio': 0.8,
    'data_root': './data'
}