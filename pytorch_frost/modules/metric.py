import torch
import torch.nn as nn
from torchmetrics.metric import Metric
from typing import Callable, Any, Dict

Tensor = torch.Tensor

class CallableMetric(Metric):
    
    def __init__(self, function: Callable, 
                 dist_reduce_fx: str='sum',
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        self._f = function
        self.add_state("value", default=torch.zeros(1), dist_reduce_fx=dist_reduce_fx)
        self.add_state("count", default=torch.zeros(1), dist_reduce_fx=dist_reduce_fx)
        
    def update(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> None:
        
        # this assumes that every tensor in the dictionary has the same number of batches
        for key in output:
            count = output[key].shape[0]
            self.value += self._f(output[key], target[key])
        self.count += count
        
    def compute(self) -> Tensor:
        
        return self.value / self.count

class BCEMetric(CallableMetric):
    
    def __init__(self, 
                 dist_reduce_fx: str='sum', 
                 **kwargs
    ) -> None:
        super().__init__(nn.functional.binary_cross_entropy, dist_reduce_fx=dist_reduce_fx, **kwargs)
        
class MSEMetric(CallableMetric):
    
    def __init__(self, 
                 dist_reduce_fx: str='sum', 
                 **kwargs
    ) -> None:
        super().__init__(nn.functional.mse_loss, dist_reduce_fx=dist_reduce_fx, **kwargs)
                