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
    epoch_loss = 0.0
    epoch_metrics = {
        'top_1_accuracy': 0.0,
        'top_5_accuracy': 0.0,
        'top_1_super_accuracy': 0.0
    }
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        loss, accuracies = process_batch(model, images, labels, criterion, optimizer, device, mode="train")
        
        epoch_loss += loss
        for key in epoch_metrics:
            epoch_metrics[key] += accuracies[key]
        
        if batch_idx % 100 == 0:
            print(f"Batch: {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}")
    
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics

def process_batch(model, images, labels, criterion, optimizer, device, mode="train"):
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
    images = images.to(device)
    labels = labels.to(device)
    
    if mode == "train":
        optimizer.zero_grad()
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    if mode == "train":
        loss.backward()
        optimizer.step()

    accuracies = {
        'top_1_accuracy': top_k_accuracy(labels, outputs, k=1, super=False),
        'top_5_accuracy': top_k_accuracy(labels, outputs, k=5, super=False),
        'top_1_super_accuracy': top_k_accuracy(labels, outputs, k=1, super=True)
    }
    
    return loss.item(), accuracies