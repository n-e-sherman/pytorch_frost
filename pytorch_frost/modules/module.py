import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, Union
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from .optimizer import WeightDecayAdamW

Tensor = torch.Tensor
Scheduler = torch.optim.lr_scheduler.LRScheduler

class MaskedEBHModel(pl.LightningModule):
    
    def __init__(self, 
                 embedding: nn.Module, body: nn.Module, head: nn.Module, loss: nn.Module, 
                 dropout: float=0.0, 
                 weight_decay: float=1e-1,
                 lr: float=5e-8,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 max_lr: float=1e-5,
                 scheduler: str='one-cycle',
                 use_scheduler: bool=True, # If false, constant learning rate
                 val_metric: Optional[Metric]=None,
                 embedding_pad_value: float=np.nan,
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
        self.max_lr = max_lr
        self.scheduler=scheduler
        
        self.val_metric = val_metric
        
        self.embedding_pad_value = embedding_pad_value
        
        self.apply(self._init_weights)
        
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6))
        
    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        
        
        # TODO: do we need to handle masking? Maybe a cleaner way to do this.
        x = self.embedding(data) # (B, N, C)
        x = self.drop(x)
        
        mask = data.get('<padding_mask>', None)
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
            if self.scheduler == 'cosine':
                _scheduler = CosineAnnealingLR(res['optimizer'], T_max=self.trainer.max_epochs)
                scheduler = {
                    'scheduler': _scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            elif self.scheduler == 'one-cycle':
                _scheduler = OneCycleLR(res['optimizer'], 
                                        max_lr=self.max_lr, 
                                        total_steps=self.trainer.estimated_stepping_batches)
                scheduler = {
                    'scheduler': _scheduler,
                    'interval': 'step',  # Specifies to update the scheduler on every optimizer step
                    'frequency': 1,
                    'name': 'learning_rate',
                }
            res['lr_scheduler'] = scheduler
            
        return res
    
    def get_num_params(self) -> None:

        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            module.weight.data[module.padding_idx] = self.embedding_pad_value
