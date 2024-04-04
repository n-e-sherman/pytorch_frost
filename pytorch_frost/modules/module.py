import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple, Union
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import CosineAnnealingLR
from .optimizer import WeightDecayAdamW

Tensor = torch.Tensor
Scheduler = torch.optim.lr_scheduler.LRScheduler

class MaskedEBHModel(pl.LightningModule):
    
    def __init__(self, 
                 embedding: nn.Module, body: nn.Module, head: nn.Module, loss: nn.Module, 
                 dropout: float=0.0, 
                 # optimizer parameters
                 weight_decay: float=1e-1,
                 lr: float=6e-4,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 use_scheduler: bool=True, # If false, constant learning rate
                 # max_lr: float=5e-3, # not used
                 # validation metric
                 val_metric: Optional[Metric]=None,
    ) -> None:
        super().__init__()
        
        # network
        self.embedding = embedding
        self.body = body
        self.head = head
        self.drop = nn.Dropout(dropout)
        
        # loss
        self.loss = loss
        
        # optimizer
        self.weight_decay = weight_decay
        self.lr = lr
        self.betas = betas
        self.use_scheduler = use_scheduler
        # self.max_lr = max_lr
        
        self.val_metric = val_metric
        
    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        
        
        # TODO: do we need to handle masking? Maybe a cleaner way to do this.
        
        x = self.embedding(data) # (B, N, C)
        x = self.drop(x)
        
        mask = data.get('<padding_mask>', None)
        if mask is not None:
            mask = ~mask.bool()
        x = self.body(x, mask=mask)
        x = self.head(x)
        return x
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        res = {'loss' : loss}
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return res
    
    def validation_step(self, batch, batch_idx):
        
        if self.val_metric is None:
            self.log('val_loss', 0, on_step=False, on_epoch=True, prog_bar=True)
            return None
        
        input, target = batch
        output = self(input)
        
        val_loss = self.val_metric.update(output, target)
        
        self.log('val_loss', self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) -> None:
        
        #TODO: Include a scheduler as well and custom optimizer
        res = {}
        res['optimizer'] = WeightDecayAdamW(self.named_parameters(), self.device, 
                                            weight_decay=self.weight_decay,
                                            lr=self.lr,
                                            betas=self.betas)
        
        if self.use_scheduler:
            _scheduler = CosineAnnealingLR(res['optimizer'], T_max=self.trainer.max_epochs)
            scheduler = {
                'scheduler': _scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
            res['lr_scheduler'] = scheduler
            
        return res
            
        