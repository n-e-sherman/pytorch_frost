import torch
import torch.nn as nn
from typing import Dict
from .utils import GlobalAttentionPool1d, GlobalProjectedAttentionPool1d, SumPool1d, MeanPool1d, NoPool1d

Tensor = torch.Tensor

class PooledEmbeddingLayer(nn.Module):


    def __init__(self, d_model: int, embedding_info: dict, 
                     bias: bool=True,
                     pooling: str='attention', # choices: attention, projected-attention, sum, mean(not recommended).
                     pooling_kwargs: dict={},
    ) -> None:
        super().__init__()
        
        ''' - embedding_info is a dictionary with  the following structure:
                {'key' : {'type': <str>, 'dim': <int>, 'padding_idx': <int>}}
                
            - 'type' can either be 'numerical' or 'categorical'
            - for numerical features, 'dim' is the input dimension of 
                    the numerical variable: often 'dim'=1.
            - for categorical features, 'dim' is the number of unique ids.
            - 'padding_idx' is only needed for categorical variables.
            
            - key must match the key for the data in the forward function
        '''
        
        self.d_model = d_model
        
        # define embeddings
        embeddings = {}
        for k,v in embedding_info.items():
            if v['type'] == 'numerical':
                embeddings[k] = nn.Linear(v['dim'], d_model, bias)
            elif v['type'] == 'categorical':
                embeddings[k] = nn.Embedding(v['dim'], d_model, padding_idx=v.get('padding_idx', None))
            else:
                raise NotImplementedError(f"embedding choice {v[type]} is not implemented for key {k}")
        self.embeddings = nn.ModuleDict(embeddings)
        
        # define pooling
        if pooling == 'attention':
            self.pool = GlobalAttentionPool1d(d_model)
        elif pooling == 'projected-attention':
            self.pool = GlobalProjectedAttentionPool1d(d_model, **pooling_kwargs)
        elif pooling == 'sum':
            self.pool = SumPool1d()
        elif pooling == 'mean':
            self.pool = MeanPool1d()
        else:
            raise NotImplementedError(f"pooling choice {pooling} is not implemented.")    
        
        
    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        
        ''' data is a dictionary of tensors
            We assume that all numerical data has shape (B, N, C) and categorical has shape (B, N)
              - B must be the same for every data field
              - N must either be the same, or have dim 1, but it must be present
              - C can be different for every tensor, and will match after embedding
        '''

        #TODO: maybe have better checking? Not clear what is a problem if there is a key mismatch between data and self.embeddings
        for key in self.embeddings:
            if key not in data:
                raise KeyError(f"key: {key} is given for an embedding, but is not included in data.")

        # NOTE: will ignore keys provided in data that are not in self.embeddings
        embedded_vectors = [] # [F](B, N, d)
        embedded_masks = []
        for k,v in self.embeddings.items():
            
            x = data[k]
            pre_mask = ~torch.isnan(x).view(*x.shape[0:2], 1) # (B, N, 1) -- True means keep
            x = torch.nan_to_num(x, nan=0.0)
            x_E = v(x)
            post_mask = ~torch.isnan(x_E).all(dim=-1, keepdim=True)
            mask = pre_mask & post_mask
            x_E = torch.nan_to_num(x_E, nan=0.0)
            embedded_vectors.append(x_E)
            embedded_masks.append(mask)
            
        res_full = torch.stack(embedded_vectors, dim=-2) # (B, N, F, d)
        mask_full = torch.stack(embedded_masks, dim=-1) # (B, N, 1, F)
        res = self.pool(res_full, ~mask_full)

        return res # (B, N, C)
                