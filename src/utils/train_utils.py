"""
이 train_utils.py 는 모델을 학습하기 위해 직접적으로 필요한 utils 를 정의하는 코드이다.
주요 기능:
- [Function] train_one_epoch: 이 함수는 한 epoch 에 대해서 모델 학습을 진행한다. 
- [Function] process_batch: 이 함수는 한 batch_size 에 대해서 모델 학습을 진행한다.

마지막 수정: 2023-09-23
"""

import torch
from tqdm import tqdm
from .metrics import top_k_accuracy

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (callable): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        device (torch.device): Device to train on.
    
    Returns:
        tuple: (epoch_loss, epoch_metrics)
    """
    model.train()
    epoch_loss = 0.0            # epoch_loss 와 epoch_metrics 는 한 epoch 에 대한 결과값을 저장하기 위한 변수이다.
    epoch_metrics = {
        'top_1_accuracy': 0.0,
        'top_5_accuracy': 0.0,
        'top_1_super_accuracy': 0.0
    }
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")): # dataloader 변수에서 배치수만큼 데이터 이미지와 라벨을 불러온다.
        loss, accuracies = process_batch(model, images, labels, criterion, optimizer, device) # process_batch 함수를 이용해서 배치수만큼 학습된 모델 결과를 불러온다.
        
        epoch_loss += loss # 해당 loss 는 배치수에 대한 손실값이므로 epoch_loss 에 더해준다.
        for key in epoch_metrics: # epoch_metrics 정확도 결과값도 마찬가지로 배치수에 대한 값이므로 더해준다.
            epoch_metrics[key] += accuracies[key]
        
        # if batch_idx % 100 == 0:
        #     print(f"Batch: {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")
    
    num_batches = len(dataloader)
    epoch_loss /= num_batches                # 한 epoch 에 대한 loss 와 accuracy 에 배치수를 나눠준다. 그 이유는 평균값을 계산하기 위함이다. 
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics # 한 epoch 에 대한 결과값을 리턴한다.

def process_batch(model, images, labels, criterion, optimizer, device):
    """
    Process a single batch of data.
    
    Args:
        model (torch.nn.Module): The neural network model.
        images (Tensor): Batch of input images.
        labels (Tensor): Batch of target labels.
        criterion (callable): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        device (torch.device): Device to process on.
        mode (str): Either "train" or "eval".
    
    Returns:
        tuple: (loss, accuracies)
    """
    images = images.to(device) # 이미지 행렬을 gpu에 할당한다. 
    labels = labels.to(device) # 라벨을 gpu 에 할당한다.
    
    optimizer.zero_grad()
    
    outputs = model(images) # 모델에다가 이미지 행렬을 인풋으로 넣어준다. 그러면 모델 예측 결과인 outputs 이 나온다.
    loss = criterion(outputs, labels) # loss 값을 계산한다. 이때 loss 는 배치사이즈(기본 64)에 대해서 처리했을 때의 Loss 이다.
    
    loss.backward()
    optimizer.step()

    accuracies = { # 한 batch 에 대한 accuracy 를 구한 것이다.
        'top_1_accuracy': top_k_accuracy(labels, outputs, k=1, super=False),
        'top_5_accuracy': top_k_accuracy(labels, outputs, k=5, super=False),
        'top_1_super_accuracy': top_k_accuracy(labels, outputs, k=1, super=True)
    }
    
    return loss.item(), accuracies