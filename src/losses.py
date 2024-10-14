import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        num_classes = predictions.size(1)

        # Create one-hot encoded target with smoothing
        one_hot = torch.zeros_like(predictions).scatter_(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / num_classes)

        # Compute the loss
        log_probs = F.log_softmax(predictions, dim=-1)
        loss = -torch.sum(one_hot * log_probs, dim=-1).mean()

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()