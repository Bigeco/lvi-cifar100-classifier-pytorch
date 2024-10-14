import torch.nn as nn

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