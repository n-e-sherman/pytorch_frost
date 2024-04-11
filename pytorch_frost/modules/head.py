import torch
import torch.nn as nn
from typing import Callable, Optional
from .utils import GlobalAttentionPool1d, GlobalProjectedAttentionPool1d, SumPool1d, MeanPool1d, NoPool1d, IdentityOperation

Tensor = torch.Tensor

def IdentityOperation(x):
    return x

class PooledHeadLayer(nn.Module):
    
    def __init__(self, d_model: int, targets: list, 
                 bias: bool=False,
                 activation: Callable=IdentityOperation, # nn.functional.sigmoid is a valid argument here.
                 pooling: Optional[str]='mean', # choices: attention, projected-attention, mean, sum, None
                 pooling_kwargs: dict={}
    ) -> None:
        
        '''
            targets is a list of column names
        '''
        
        super().__init__()
        
        self.targets = targets
        self.head = nn.Linear(d_model, len(targets), bias=bias)
        self.activation = activation

        # define pooling
        if pooling is None:
            self.pool = NoPool1d()
        elif pooling == 'attention':
            self.pool = GlobalAttentionPool1d(d_model)
        elif pooling == 'projected-attention':
            self.pool = GlobalProjectedAttentionPool1d(d_model, **pooling_kwargs)
        elif pooling == 'sum':
            self.pool = SumPool1d()
        elif pooling == 'mean':
            self.pool = MeanPool1d()
        else:
            raise NotImplementedError(f"pooling choice {pooling} is not implemented.")   
        
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        
        # x: (B, N, d)
        
        mask = mask if mask is not None else torch.zeros_like(x, dtype=torch.bool).all(dim=-1, keepdim=False)
        
        x = self.pool(x, mask.unsqueeze(-2)) # (B, d) -- (B, N, d) if no pooling
        x = self.head(x) # (B, T) -- (B, N, T) if no pooling
        x = self.activation(x)
        res = {}
        for i,key in enumerate(self.targets):
            res[key] = x[...,[i]] #(B, 1) -- (B, N, 1) if no pooling
        return res
        
    
    