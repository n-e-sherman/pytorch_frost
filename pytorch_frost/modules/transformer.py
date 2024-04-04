import torch
import torch.nn as nn
from typing import Callable, Optional, Union

Tensor = torch.Tensor

class TransformerBodyLayer(nn.Module):
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int=768, # usually I want this to be 3*d_model
                 dropout: float=0.0,
                 activation: Union[str, Callable[[Tensor], Tensor]]=nn.functional.gelu,
                 layer_norm_eps: float=1e-05,
                 batch_first: bool=True,
                 norm_first: bool=True,
                 device: Optional[Union[str, torch.device]]=None,
                 dtype: Optional[torch.dtype]=None,
                 enable_nested_tensor: bool=False,
                 mask_check: bool=True,
                 norm=None,
                 ) -> None:
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   layer_norm_eps=layer_norm_eps,
                                                   batch_first=batch_first,
                                                   norm_first=norm_first,
                                                   #bias=bias,
                                                   device=device,
                                                   dtype=dtype)
        
        self.body = nn.TransformerEncoder(encoder_layer, num_layers,
                                          norm=norm,
                                          enable_nested_tensor=enable_nested_tensor,
                                          mask_check=mask_check)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        
        return self.body(x, src_key_padding_mask=mask)