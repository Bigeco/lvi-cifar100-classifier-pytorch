"""
이 train.py 는 모델 학습을 전체적으로 진행하는 함수를 모아둔 코드이다.
주요 기능:
- [Function] objective: 이 함수는 epoch 수만큼 학습을 진행하는 함수이다.

마지막 수정: 2023-09-23
"""

import torch
import torch.nn as nn
from src.data.dataset import load_data
# from src.models.resnet import ResNet56
from src.utils.train_utils import train_one_epoch
from src.utils.eval_utils import evaluate_one_epoch
from src.config import CONFIG

def objective(config, transform, model):
    """
    Main training function.
    
    Args:
        config (dict): Configuration dictionary.
        model (torch.nn.Module): The neural network model.
    
    Returns:
        tuple: (best_model, [best_valid_loss, best_top1_acc, best_top5_acc, best_top1_super_acc])
    """
    # Load data
    train_loader, valid_loader, test_loader = load_data(
        config['batch_size'], 
        transform,
        config['num_workers'], 
        config['train_ratio'], 
        config['data_root']
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize Best Loss and Accuracy 
    # 다음 변수들을 정의한 이유는 여러 epoch 를 진행하면서 무조건 모델을 업데이트하는게 아니라 좋은 정확도가 나온 것에 대해 모델을 판별하는 조건에 필요한 변수이다.
    best_model = None
    best_valid_loss = float('inf')
    best_top1_acc = 0 
    best_top5_acc = 0
    best_top1_super_acc = 0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"Epoch: {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config['device'])
        valid_loss, valid_metrics = evaluate_one_epoch(model, valid_loader, criterion, config['device'])

        print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        print(f"Train Accuracy: {train_metrics['top_1_accuracy']*100:.2f}%, {train_metrics['top_5_accuracy']*100:.2f}%, {train_metrics['top_1_super_accuracy']*100:.2f}%")
        print(f"Valid Accuracy: {valid_metrics['top_1_accuracy']*100:.2f}%, {valid_metrics['top_5_accuracy']*100:.2f}%, {valid_metrics['top_1_super_accuracy']*100:.2f}%")

        # Save Best Model, Loss and Accuracy
        if (valid_loss < best_valid_loss and # 이전 epoch 중에서 best 수치인 valid loss 값과 현재 epoch 에서 구한 valid loss 값을 비교한다. 
            valid_metrics['top_1_accuracy'] >= best_top1_acc and # 이후로도 마찬가지로 이전 epoch 중에서 최고인 값과 현재 epoch 값을 비교한다.
            valid_metrics['top_5_accuracy'] >= best_top5_acc and 
            valid_metrics['top_1_super_accuracy'] >= best_top1_super_acc):
            
            best_valid_loss = valid_loss # 비교했을 때 만약 현재 epoch 결과값이 최고라면 값을 변경해준다. 
            best_top1_acc = valid_metrics['top_1_accuracy']
            best_top5_acc = valid_metrics['top_5_accuracy']
            best_top1_super_acc = valid_metrics['top_1_super_accuracy']

            best_model = model
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved!")

    _, test_epoch_metrics = \
        evaluate_one_epoch(best_model, test_loader, criterion, CONFIG['device'])

    return best_model, [best_valid_loss, best_top1_acc, best_top5_acc, best_top1_super_acc], test_epoch_metrics

if __name__ == "__main__":
    # model = ResNet56().to(CONFIG['device'])
    # best_model, [best_valid_loss, best_top1_acc, best_top5_acc, best_top1_super_acc], test_epoch_metrics = objective(CONFIG, (TRAIN_TRANSFORM, TEST_TRANSFORM), model)

    # print(f'Best Validation Loss: {best_valid_loss:.4f}')
    # print(f'Best Top-1 Accuracy: {best_top1_acc * 100:.2f}%')
    # print(f'Best Top-5 Accuracy: {best_top5_acc * 100:.2f}%')
    # print(f'Best Top-1 Super Accuracy: {best_top1_super_acc * 100:.2f}%')

    # print(f"Test Top-1 Accuracy: {test_epoch_metrics['top_1_accuracy'] * 100:.2f}%")
    # print(f"Test Top-5 Accuracy: {test_epoch_metrics['top_5_accuracy'] * 100:.2f}%")
    # print(f"Test Top-1 Super Accuracy: {test_epoch_metrics['top_1_super_accuracy'] * 100:.2f}%")
    pass