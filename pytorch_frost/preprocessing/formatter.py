import warnings
import torch
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Callable, Dict

Tensor = torch.Tensor
PandasDataFrame = pd.DataFrame

class DataFormatter:
    
    def __init__(self, 
                 numerical_columns: list = [],
                 categorical_columns: list = [],
                 targets: list = [],
                 sort_cols: list = [],
                 targets_pooled: Union[list, bool]=True,
                 targets_type: Union[list, type]=float,
                 formatter: Dict[str, Callable]={}
    ) -> None:
        
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns)
        self.targets = list(targets)
        self.sort_cols = sort_cols
        
        self.targets_type = targets_type
        if not isinstance(self.targets_type, list):
            self.targets_type = [targets_type for _ in self.targets]
            
        self.targets_pooled = targets_pooled
        if not isinstance(self.targets_pooled, list):
            self.targets_pooled = [targets_pooled for _ in self.targets]
        
        # make sure target is removed from other columns
        for target in targets:
            if target in self.numerical_columns:
                self.numerical_columns.remove(target)
                warnings.warn(f"column {target} was provided in both numerical_columns and targets. Removing from input.")
            if target in self.categorical_columns:
                self.categorical_columns.remove(target)
                warnings.warn(f"column {target} was provided in both categorical_columns and targets. Removing from input.")
        self.inputs = self.numerical_columns + self.categorical_columns
                
        # define format options
        self.formatter = {}
        for col in self.numerical_columns:
            self.formatter[col] = self.to_float_tensor
        for col in self.categorical_columns:
            self.formatter[col] = self.to_int_tensor
        for col, pool, _type in zip(self.targets, self.targets_pooled, self.targets_type):
            if _type == float:
                if pool:
                    self.formatter[col] = self.to_float_pooled_tensor
                else:
                    self.formatter[col] = self.to_float_tensor
            elif _type == int:
                if pool:
                    self.formatter[col] = self.to_int_pooled_tensor
                else:
                    self.formatter[col] = self.to_int_tensor
            else:
                raise NotImplementedError(f'target_type {_type} for column {col} is not implemented.')
        self.formatter.update(formatter) # override formatter with input
                    
    def format_sample(self, pdf: PandasDataFrame) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        pdf = pdf.sort_values(by=self.sort_cols)
        
        input = {}
        for key in self.inputs:
            input[key] = self.formatter[key](pdf[key])
            
        target = {}
        for key in self.targets:
            target[key] = self.formatter[key](pdf[key])
        
        return input, target
    
    def format_output(self, input: Dict[str, Tensor], output: Dict[str, Tensor]) -> PandasDataFrame:

        res = output.copy()
        res.update(input)

        # Remove mask if it does not exist
        if '<padding_mask>' in res:
            if res['<padding_mask>'] is None:
                res.pop('<padding_mask>')
        
        # Get tensor shape info
        B, N = 0, 0
        for k,v in res.items():
            B = np.max([B, v.shape[0]])
            N = np.max([N, v.shape[1]])
            if len(v.shape) == 2:
                res[k] = v.unsqueeze(-1)

        # Flatten all tensors to form df
        zeros = torch.zeros(B, N, 1)
        for k,v in res.items():
            _v = v + zeros
            res[k] = _v.detach().view(-1).numpy()

        # Add a sample index
        res['sample'] = np.repeat(np.arange(B), N)

        df_res = pd.DataFrame(res)
        if '<padding_mask>' in df_res.columns:
            df_res = df_res.loc[df_res['<padding_mask>'] == 1]
            df_res = df_res.drop('<padding_mask>', axis=1)
        df_res = df_res.drop_duplicates()
        return df_res
    
    def fit(self, *args, **kwargs):
        pass
            
    @staticmethod
    def to_float_tensor(x: pd.Series) -> Tensor:
            return torch.tensor(x.astype(float).values).reshape(-1 ,1).float()
        
    @staticmethod
    def to_int_tensor(x: pd.Series) -> Tensor:
        return torch.tensor(x.astype(int).values).reshape(-1).int()
    
    @staticmethod
    def to_float_pooled_tensor(x: pd.Series) -> Tensor:
        return torch.tensor(x.astype(float).unique()[0]).reshape(1).float()
    
    @staticmethod
    def to_int_pooled_tensor(x: pd.Series) -> Tensor:
        return torch.tensor(x.astype(int).unique()[0]).reshape(1).int()
    