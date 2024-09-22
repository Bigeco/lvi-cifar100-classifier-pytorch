import torch
import torch.nn as nn
from src.data.dataset import load_data
# from src.models.resnet import ResNet56
from src.utils.train_utils import train_one_epoch
from src.utils.eval_utils import evaluate_one_epoch
from src.config import CONFIG

def objective(config, model):
    """
    Main training function.
    
    Args:
        config (dict): Configuration dictionary.
        model (torch.nn.Module): The neural network model.
    
    Returns:
        tuple: (best_model, [best_valid_loss, best_top1_acc, best_top5_acc, best_top1_super_acc])
    """
    # Load data
    train_loader, valid_loader = load_data(
        config['batch_size'], 
        config['num_workers'], 
        config['train_ratio'], 
        config['data_root']
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize Best Loss and Accuracy
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
        if (valid_loss < best_valid_loss and 
            valid_metrics['top_1_accuracy'] >= best_top1_acc and 
            valid_metrics['top_5_accuracy'] >= best_top5_acc and 
            valid_metrics['top_1_super_accuracy'] >= best_top1_super_acc):
            best_valid_loss = valid_loss
            best_top1_acc = valid_metrics['top_1_accuracy']
            best_top5_acc = valid_metrics['top_5_accuracy']
            best_top1_super_acc = valid_metrics['top_1_super_accuracy']
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved!")

    return model, [best_valid_loss, best_top1_acc, best_top5_acc, best_top1_super_acc]

if __name__ == "__main__":
    # model = ResNet56().to(CONFIG['device'])
    # best_model, [best_valid_loss, best_top1_acc, best_top5_acc, best_top1_super_acc] = objective(CONFIG, model)

    # print(f'Best Validation Loss: {best_valid_loss:.4f}')
    # print(f'Best Top-1 Accuracy: {best_top1_acc * 100:.2f}%')
    # print(f'Best Top-5 Accuracy: {best_top5_acc * 100:.2f}%')
    # print(f'Best Top-1 Super Accuracy: {best_top1_super_acc * 100:.2f}%')
    pass