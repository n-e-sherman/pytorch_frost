
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Tuple
from .transformer import DataTransformer


PandasDataFrame = pd.DataFrame
Tensor = torch.Tensor

class PDFDataset(Dataset):
    
    def __init__(self, pdf: PandasDataFrame, transformer: DataTransformer, 
                 id_column: str='game_id',
                 store_samples=True, 
                 encoded=True,
                 **kwargs
    ) -> None:
        
        self.pdf = pdf
        self.transformer = transformer
        self.store_samples = store_samples
        
        if not encoded:
            self.pdf = transformer.encode(pdf)
                
        self.idx_col = 'idx'
        while True:
            if self.idx_col in self.pdf.columns:
                self.idx_col += '_'
            else:
                self.pdf[self.idx_col] = self.pdf[id_column].astype('category').cat.codes
                break
        
        self._len = len(self.pdf[self.idx_col].unique())
        self.samples = {}
        
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        if idx in self.samples:
            return self.samples[idx]
        
        sample = self.pdf.loc[self.pdf[self.idx_col] == idx]
        input, target = self.transformer.format(sample)
        
        if self.store_samples:
            self.samples[idx] = (input, target)
            
        return input, target
    
    def __len__(self) -> int:
        
        return self._len