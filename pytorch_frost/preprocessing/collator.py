import torch
from torch.utils.data._utils.collate import default_collate
from typing import List, Any
from .transformer import DataTransformer


Tensor = torch.Tensor

class PadCollator:
    
    def __init__(self, transformer: DataTransformer) -> None:
        
        self.pad_values = transformer.encoded_pad_values
    
    def __call__(self, batch: list) -> list:
        
        inputs = {}
        targets = {}
        for input, target in batch:
            
            for key, tensor in input.items():
                if key in inputs:
                    inputs[key].append(tensor)
                else:
                    inputs[key] = [tensor]
        
            for key, tensor in target.items():
                if key in targets:
                    targets[key].append(tensor)
                else:
                    targets[key] = [tensor]
                
        padding_mask = None
        for key, tensors in inputs.items():
            value, mask = self._batch_pad_tensors(tensors, self.pad_values[key])
            inputs[key] = value
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if padding_mask is None:
                padding_mask = mask
            else:
                padding_mask = padding_mask & mask
                
        for key, tensors in targets.items():
            value, _= self._batch_pad_tensors(tensors, self.pad_values[key])
            targets[key] = value
        
        inputs['<padding_mask>'] = ~padding_mask # flip for transformer: True means pad
        
        return inputs, targets
            
    def _batch_pad_tensors(self, tensors: List[Tensor], pad_value: Any) -> Tensor:

        # Example list of tensors of potentially different shapes
        max_sizes = [max(t.size(i) for t in tensors) for i in range(tensors[0].dim())]
        new_shape = [len(tensors)] + max_sizes
        
        batched_tensor = torch.full(new_shape, pad_value, dtype=tensors[0].dtype)
        mask = torch.zeros(new_shape, dtype=torch.bool)
        
        # Copy each tensor into the new batched tensor
        for i, t in enumerate(tensors):
            slices = tuple(slice(0, s) for s in t.shape)
            batched_tensor[i][slices] = t
            mask[i][slices] = 1
            
        return batched_tensor, mask