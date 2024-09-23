"""
이 eval_utils.py 는 학습된 모델을 테스트 하기 위해 필요한 utils 를 정의하는 코드이다.
주요 기능:
- [Function] evaluate_one_epoch: 이 함수는 모델 학습 한 epoch 가 끝날 때마다 Validset 에 대해서 정확도가 얼마 나오는지 확인하는 용도이다.
                                 또한, Testset 에 대해서도 정확도가 얼마 나오는지 구할 수 있다. 

마지막 수정: 2023-09-23
"""

import torch
from tqdm import tqdm
from .metrics import top_k_accuracy

def evaluate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluate the model for one epoch.
    
    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for evaluation data.
        criterion (callable): Loss function.
        device (torch.device): Device to evaluate on.
    
    Returns:
        tuple: (epoch_loss, epoch_metrics)
    """
    model.eval()
    epoch_loss = 0.0            # epoch_loss 와 epoch_metrics 는 한 epoch 에 대한 결과값을 저장하기 위한 변수이다.
    epoch_metrics = {
        'top_1_accuracy': 0.0,
        'top_5_accuracy': 0.0,
        'top_1_super_accuracy': 0.0
    }

    with torch.no_grad():  # no_grad 로 선언하게 되면 학습을 진행하지 않는다. 
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")): # dataloader 변수에서 배치수만큼 데이터 이미지와 라벨을 불러온다.
            images = images.to(device) # 이미지 행렬을 gpu에 할당한다. 
            labels = labels.to(device) # 라벨을 gpu 에 할당한다. 
            
            outputs = model(images) # 모델에다가 이미지 행렬을 인풋으로 넣어준다. 그러면 모델 예측 결과인 outputs 이 나온다.
            loss = criterion(outputs, labels) # loss 값을 계산한다. 이때 loss 는 배치사이즈(기본 64)에 대해서 처리했을 때의 Loss 이다.

            epoch_loss += loss.item() # epoch loss 는 한 epoch 에 대한 loss 를 말한다. 따라서 변수 loss 값을 더해주어야 한다.
            epoch_metrics['top_1_accuracy'] += top_k_accuracy(labels, outputs, k=1, super=False)      # 계산한 정확도는 epoch_metrics 에 값을 더한다. 
            epoch_metrics['top_5_accuracy'] += top_k_accuracy(labels, outputs, k=5, super=False)        # epoch_metrics 값도 마찬가지로 한 epoch 에 대한 결과이다.
            epoch_metrics['top_1_super_accuracy'] += top_k_accuracy(labels, outputs, k=1, super=True)   # 따라서 이때 top_k_accuracy 함수를 통해 구한 것은 배치 사이즈(기본 64)에 대해서 처리했을 때 결과이므로 더해야 한다.
            
            # if batch_idx % 100 == 0:
            #     print(f"Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        num_batches = len(dataloader) # dataloader 의 length 이므로 배치수가 나오게 된다.
        epoch_loss /= num_batches # 배치수만큼 나누어주어 loss 값의 평균값을 계산한다. 
        for key in epoch_metrics: # epoch_metrics 결과값도 마찬가지로 평균값으로 계산되도록 나누어준다.
            epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics # 결과값인 loss 와 정확도 epoch_metrics 를 리턴한다.