import torch
import torch.nn as nn
from typing import Callable, Dict

Tensor = torch.Tensor

class CallableLoss(nn.Module):
    
    def __init__(self, function: Callable) -> None:
        super().__init__()
        
        self._f = function

    def forward(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        
        device = next(iter(output.values())).device
        res = torch.zeros(1, device=device)
        for key in output:
            res += self._f(output[key], target[key])
            
        return res
        
class MSELoss(CallableLoss):
    
    def __init__(self):
        super().__init__(nn.functional.mse_loss)
        
class BCELoss(CallableLoss):
    
    def __init__(self):
        super().__init__(nn.functional.binary_cross_entropy)
        