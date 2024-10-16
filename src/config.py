from models.ResNet import *
from models.ResNext import *
from models.EfficientNet import *
from models.DenseNet import *
from models.ViT import *
from models.WideResNet import *
from models.Swin import *
from models.ShakePyramidNet import *

from losses import *
import torch.optim as optim
from optimizers import *
from schedulers import *

# model 정의
MODEL_DICT = {
    "resnet9": resnet9,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50": resnext50,
    "resnext101": resnext101,
    "resnext152": resnext152,
    "wide_resnet28_10": wide_resnet28_10,
    "shake_pyramidnet_110": shake_pyramidnet_110,  # Best Top-1 accuracy
    "swin1": swin1,
    "swin2": swin2,
    "swin3": swin3,
    "swin4": swin4,
    "swin5": swin5,
    "swin6": swin6,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
    "densenet201": densenet201,
    "vit": vit
}

CRITERION_DICT = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,  # Best Top-1 accuracy
    "FocalLoss": FocalLoss,
    "LabelSmoothingLoss": LabelSmoothingLoss
}

# optimizer 정의
OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SAM": SAMSGD
}

SCHEDULER_DICT = {
    "LambdaLR": lambda optimizer: lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.95 ** epoch  # 예시 함수, 필요에 따라 수정
    ),
    "MultiplicativeLR": lambda optimizer: lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda epoch: 0.95  # 예시 함수, 필요에 따라 수정
    ),
    "StepLR": lambda optimizer: lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    ),
    "MultiStepLR": lambda optimizer, epochs: lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epochs // 2, epochs * 3 // 4],
        gamma=0.1
    ),
    "ConstantLR": lambda optimizer: lr_scheduler.ConstantLR(
        optimizer,
        factor=0.5,
        total_iters=5
    ),
    "LinearLR": lambda optimizer: lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=5
    ),
    "ExponentialLR": lambda optimizer: lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.9
    ),
    "PolynomialLR": lambda optimizer: lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=100,
        power=1.0
    ),
    "CosineAnnealingLR": lambda optimizer: lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    ),
    "ChainedScheduler": lambda optimizer: lr_scheduler.ChainedScheduler(
        [
            lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
            lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        ]
    ),
    "SequentialLR": lambda optimizer: lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
            lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        ],
        milestones=[60]
    ),
    "ReduceLROnPlateau": lambda optimizer: lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10
    ),
    "CyclicLR": lambda optimizer: lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.001,
        max_lr=0.1,
        step_size_up=2000
    ),
    "OneCycleLR": lambda optimizer: lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=100
    ),
    "CosineAnnealingWarmRestarts": lambda optimizer: lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    ),
    # Best Top-1 accuracy
    "CombinedScheduler": lambda optimizer, epochs: CombinedScheduler(
        optimizer,
        milestones=[epochs // 2, epochs * 3 // 4],
        mode='min',
        factor=0.1,
        patience=10
    )
}