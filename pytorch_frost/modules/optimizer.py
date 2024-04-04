import torch
import torch.nn as nn
import inspect
from typing import Tuple, Any, Optional, Union

class WeightDecayAdamW(torch.optim.AdamW):

    def __init__(self, 
                 parameters: dict,
                 device: str,
                 weight_decay: float=1e-1, 
                 lr: float=6e-4, 
                 betas: Tuple[float, float]=(0.9, 0.95)
    ) -> None:
        
        self.device = device
        self.weight_decay = weight_decay
        self.lr = lr
        self.betas = betas

        optim_groups = self._get_optim_groups(parameters)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        super().__init__(optim_groups, lr=lr, betas=betas, **extra_args)
        
    def _get_optim_groups(self, parameters) -> Tuple[list, dict]:

        param_dict = {pn: p for pn, p in parameters}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        return optim_groups
    
    def export_checkpoint(self) -> dict:

        checkpoint = {
            'device': self.device,
            'weight_decay': self.weight_decay,
            'lr': self.lr,
            'betas': self.betas,
            'state_dict': self.state_dict(),
        }
        return checkpoint

    @classmethod
    def import_checkpoint(cls, checkpoint: dict, 
                          model: nn.Module, 
                          device: Optional[Union[str, torch.device]] = None
    ) -> Any:

        if device is not None:
            checkpoint['device'] = device

        optimizer = WeightDecayAdamW(model.named_parameters(),
                                   checkpoint['device'], 
                                   weight_decay=checkpoint['weight_decay'],
                                   lr=checkpoint['lr'],
                                   betas=checkpoint['betas'])
        optimizer.load_state_dict(checkpoint['state_dict'])
        return optimizer
        