import torch
import torch.nn as nn

from typing import Optional

Tensor = torch.Tensor

class GlobalAttentionPool1d(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, d_model))
        torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """
        x: Tensor of shape (..., F, d)
        """
    
        mask = mask if mask is not None else torch.zeros_like(x, dtype=torch.bool).all(dim=-1, keepdim=True).transpose(-1, -2)
    
        q = self.query
        
        attention_scores = torch.matmul(x, q.transpose(-2, -1)).transpose(-2, -1) # (..., 1, F)
        attention_scores = torch.where(mask, torch.tensor(float('-inf'), device=attention_scores.device), attention_scores)
    
        attention_weights = nn.functional.softmax(attention_scores, dim=-1) # (..., 1, F)
        attention_weights = torch.where(mask, torch.tensor(float(0.0), device=attention_weights.device), attention_weights)
        
        attention_output = torch.matmul(attention_weights, x) # (..., 1, d)
        res = attention_output.squeeze(-2)
        
        return res

class GlobalProjectedAttentionPool1d(GlobalAttentionPool1d):
    def __init__(self, d_model: int, bias: bool=False) -> None:
        super().__init__(d_model)

        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """
        x: Tensor of shape (..., F, d)
        """
        
        mask = mask if mask is not None else torch.zeros_like(x, dtype=torch.bool).all(dim=-1, keepdim=True).transpose(-1, -2)

        q = self.query
        K = self.k_proj(x)
        V = self.v_proj(x)

        attention_scores = torch.matmul(x, q.transpose(-2, -1)).transpose(-2, -1) # (..., 1, F)
        attention_scores = torch.where(mask, torch.tensor(float('-inf'), device=attention_scores.device), attention_scores)
    
        attention_weights = nn.functional.softmax(attention_scores, dim=-1) # (..., 1, F)
        attention_weights = torch.where(mask, torch.tensor(float(0.0), device=attention_weights.device), attention_weights)
        
        attention_output = torch.matmul(attention_weights, torch.nan_to_num(V, nan=0.0)) # (..., 1, d)
        res = attention_output.squeeze(-2)
        return self.out_proj(res)
    
class MeanPool1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        
        mask = mask if mask is not None else torch.zeros_like(x, dtype=torch.bool).all(dim=-1, keepdim=True).transpose(-1, -2)
        # (B, N, 1, F) and x is (B, N, F, d)
        
        x = torch.where(mask.transpose(-1, -2), torch.tensor(float(0.0), device=x.device), x)
        
        return self.pool(x.transpose(-1, -2)).transpose(-1, -2).squeeze(1)
    
class SumPool1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        
        mask = mask if mask is not None else torch.zeros_like(x, dtype=torch.bool).all(dim=-1, keepdim=True).transpose(-1, -2)
        # (B, N, 1, F) and x is (B, N, F, d)
        
        x = torch.where(mask.transpose(-1, -2), torch.tensor(float(0.0), device=x.device), x)
        return torch.sum(x, dim=-2)
    
class NoPool1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        
        return x
    
def IdentityOperation(x):
    return x