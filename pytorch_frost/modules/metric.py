import torch
import torch.nn as nn
from torchmetrics.metric import Metric
from typing import Callable, Any, Dict

Tensor = torch.Tensor

class CallableMetric(Metric):
    
    def __init__(self, function: Callable, 
                 function_kwargs: Dict[str, Any]={},
                 dist_reduce_fx: str='sum',
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        self._f = function
        self._f_kwargs = function_kwargs
        self.add_state("value", default=torch.zeros(1), dist_reduce_fx=dist_reduce_fx)
        self.add_state("count", default=torch.zeros(1), dist_reduce_fx=dist_reduce_fx)
        
    def update(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> None:
        
        # this assumes that every tensor in the dictionary has the same number of batches
        # device = next(iter(output.values())).device
        # val = val.to(device)
        for key in output:
            val = self._f(output[key], target[key], **self._f_kwargs)
            self.value += val.detach()  # Ensure detached value is used for accumulation
            self.count += output[key].nelement() / output[key].shape[1]  # Adjust for the number of elements in a batch
        
    def compute(self) -> Tensor:
        
        return self.value / self.count

class BCEMetric(CallableMetric):
    
    def __init__(self, 
                 dist_reduce_fx: str='sum', 
                 **kwargs
    ) -> None:
        super().__init__(nn.functional.binary_cross_entropy, 
                         function_kwargs={'reduction': 'sum'}, 
                         dist_reduce_fx=dist_reduce_fx, 
                         **kwargs)
        
class MSEMetric(CallableMetric):
    
    def __init__(self, 
                 dist_reduce_fx: str='sum', 
                 **kwargs
    ) -> None:
        super().__init__(nn.functional.mse_loss, 
                         function_kwargs={'reduction': 'sum'}, 
                         dist_reduce_fx=dist_reduce_fx, 
                         **kwargs)
                