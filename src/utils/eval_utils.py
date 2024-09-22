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
    epoch_loss = 0.0
    epoch_metrics = {
        'top_1_accuracy': 0.0,
        'top_5_accuracy': 0.0,
        'top_1_super_accuracy': 0.0
    }

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            epoch_metrics['top_1_accuracy'] += top_k_accuracy(labels, outputs, k=1, super=False)
            epoch_metrics['top_5_accuracy'] += top_k_accuracy(labels, outputs, k=5, super=False)
            epoch_metrics['top_1_super_accuracy'] += top_k_accuracy(labels, outputs, k=1, super=True)
            
            if batch_idx % 100 == 0:
                print(f"Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        num_batches = len(dataloader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics