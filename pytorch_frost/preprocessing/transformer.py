import torch
import pandas as pd
import numpy as np

from typing import Optional, Dict, Tuple
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from .encoder import DataEncoder
from .formatter import DataFormatter

PandasDataFrame = pd.DataFrame
Tensor = torch.Tensor

class DataTransformer:
    
    def __init__(self, encoder: DataEncoder, formatter: DataFormatter) -> None:
        
        '''
            Formatter has the knowledge of what is and is not relevant for the model.
            formatter.numerical_columns and formatter.categorical_columns are strictly inputs
            formatter.targets are strictly the target for the model.
        '''
        
        self.encoder = encoder
        self.formatter = formatter
        
        self._embedding_info = self._get_embedding_info()
        
    def fit(self, pdf_train: PandasDataFrame, pdf_full: Optional[PandasDataFrame]=None) -> None:
        
        self.encoder.fit(pdf_train, pdf_full)
        self.formatter.fit(pdf_train, pdf_full)
        
    def encode(self, pdf: PandasDataFrame) -> PandasDataFrame:
        
        return self.encoder.encode(pdf)
    
    def decode(self, pdf: PandasDataFrame) -> PandasDataFrame:
        
        return self.encoder.encode(pdf)
    
    def format(self, pdf: PandasDataFrame) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        return self.formatter.format_sample(pdf)
    
    def inverse_format(self, input: Dict[str, Tensor], output: Dict[str, Tensor]) -> PandasDataFrame:
        
        return self.formatter(input, output)
    
    def transform(self, pdf: PandasDataFrame) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        pdf_enc = self.encoder.encode(pdf)
        return self.formatter.format_sample(pdf_enc)
    
    def inverse_transform(self, input: Dict[str, Tensor], output: Dict[str, Tensor]) -> PandasDataFrame:
    
        res = self.formatter.format_output(input, output)
        return self.encoder.decode(res)
        
    def create_model_signature(self) -> ModelSignature:
        
        _input_schema = []
        for key in self.formatter.numerical_columns:
            _input_schema.append(TensorSpec(np.dtype(np.float32), (-1, -1, -1), name=key))
        for key in self.formatter.categorical_columns:
            _input_schema.append(TensorSpec(np.dtype(np.int32), (-1, -1), name=key))
        input_schema = Schema(_input_schema)
        
        # output assumes a float output, but this may not always be true.
        _output_schema = []
        for key, _type, pool in zip(self.formatter.targets, self.formatter.targets_type, self.formatter.targets_pooled):
            
            if _type == int:
                np_type = np.int32
            elif _type == float:
                np_type = np.float32
            else:
                raise RuntimeError(f'Creating model schema, but target type was {_type} which is not valid.')    
                
            if pool:
                shape = (-1, -1)
            else:
                shape = (-1, -1, -1)
                
            _output_schema.append(TensorSpec(np.dtype(np_type), shape, name=key))
        output_schema = Schema(_output_schema)

        return ModelSignature(inputs=input_schema, outputs=output_schema)   
        
    @property
    def embedding_info(self):
        
        return self._embedding_info
    
    @property
    def targets(self):
        
        return self.formatter.targets
    
    def _get_embedding_info(self) -> dict:
        
        res = {}
        for col in self.formatter.numerical_columns:
            res[col] = {'type': 'numerical', 'dim': 1}

        for col in self.formatter.categorical_columns:
            dim = len(self.encoder.transformer.named_transformers_[col]['encoder'].categories_[0])
            res[col] = {'type': 'categorical', 'dim': dim, 'padding_idx': self.encoder.encoded_pad_values[col]}
            
        return res    
