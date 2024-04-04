import torch

from typing import Any

Optimizer = torch.optim.Optimizer

class OneCycleScheduler(torch.optim.lr_scheduler.OneCycleLR):

    def __init__(self, 
                 optimizer: Optimizer, 
                 max_lr: float, 
                 steps_per_epoch: int, 
                 epochs: int, 
                 last_epoch: int=-1
    ) -> None:

        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        super().__init__(optimizer, max_lr, 
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         last_epoch=last_epoch)
    
    def export_checkpoint(self) -> dict:

        checkpoint = {
            'max_lr': self.max_lr,
            'steps_per_epoch': self.steps_per_epoch,
            'epochs': self.epochs,
            'state_dict': self.state_dict(),
        }
        return checkpoint

    @classmethod
    def import_checkpoint(cls, checkpoint: dict, optimizer: Any, epoch: int) -> Any:

        last_epoch = checkpoint['steps_per_epoch'] * epoch
        scheduler = OneCycleScheduler(optimizer,
                                      checkpoint['max_lr'],
                                      checkpoint['steps_per_epoch'],
                                      checkpoint['epochs'],
                                      last_epoch=last_epoch)
        scheduler.load_state_dict(checkpoint['state_dict'])
        return scheduler
        