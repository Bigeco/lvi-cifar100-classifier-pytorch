import torch.optim.lr_scheduler as lr_scheduler

# PyramidNet with Shake-drop 전용 scheduler
class CombinedScheduler:
    def __init__(self, optimizer, milestones, mode, factor, patience):
        self.multistep = lr_scheduler.MultiStepLR(optimizer, milestones)
        self.plateau = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

    def step(self, metrics=None):
        self.multistep.step()
        if metrics is not None:
            self.plateau.step(metrics)