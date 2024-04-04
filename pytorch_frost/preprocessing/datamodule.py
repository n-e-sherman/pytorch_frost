import warnings
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Type, Any, Callable
from .transformer import DataTransformer
from .dataset import PDFDataset

PandasDataFrame = pd.DataFrame

class PDFDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 train: Optional[PandasDataFrame]=None,
                 val: Optional[PandasDataFrame]=None,
                 test: Optional[PandasDataFrame]=None,
                 predict: Optional[PandasDataFrame]=None,
                 transformer: Optional[DataTransformer]=None,
                 dataset_class: Type[Dataset]=PDFDataset,
                 batch_size: int=64,
                 num_workers: int=4,
                 shuffle: bool=True,
                 pin_memory: bool=True,
                 collate_fn: Optional[Callable]=None,
                 **dataset_kwargs # used to pass arguments into the dataset_class
    ) -> None:
        super().__init__()
        
        self._data = {}
        
        self.df_train = train
        self.df_val = val
        self.df_test = test
        self.df_predict = predict
        self.transformer = transformer
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None
        
        self.train_dataloader_kwargs = dict(batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            shuffle=shuffle, 
                                            pin_memory=pin_memory, 
                                            collate_fn=collate_fn)
        
        self.val_dataloader_kwargs = dict(batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            shuffle=shuffle, 
                                            pin_memory=pin_memory, 
                                            collate_fn=collate_fn)
        
        self.test_dataloader_kwargs = dict(batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            shuffle=shuffle, 
                                            pin_memory=pin_memory, 
                                            collate_fn=collate_fn)
        
        self.predict_dataloader_kwargs = dict(batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            shuffle=shuffle, 
                                            pin_memory=pin_memory, 
                                            collate_fn=collate_fn)
        
        
    def prepare_data(self) -> None:
        super().prepare_data()
        
    def setup(self, stage: Optional[str]=None) -> None:
        
        if stage is None:
            for _stage in ['fit', 'validate', 'test', 'predict']:
                try: 
                    self.setup(_stage)
                except Exception as e:
                    print(f'{e} -- skipping')
            
        elif stage == 'fit':
            if self.df_train is None:
                raise RuntimeError('Setting up for fit stage, but no training dataset is provided.')
            self.train_set = self.dataset_class(self.df_train, self.transformer, **self.dataset_kwargs)
            if self.df_val is None:
                warnings.warn('No validation set is provided, skipping validation step during training.')
            else:
                self.val_set = self.dataset_class(self.df_val, self.transformer, **self.dataset_kwargs)
            
        elif stage == 'validate':
            if self.df_val is None:
                raise RuntimeError('Setting up for validation stage, but no validation dataset is provided.')
            self.val_set = self.dataset_class(self.df_val, self.transformer, **self.dataset_kwargs)
                
        elif stage in ('test', None):
            if self.df_test is None:
                raise RuntimeError('Setting up for test stage, but no test dataset is provided.')
            self.test_set = self.dataset_class(self.df_test, self.transformer, **self.dataset_kwargs)
                
        elif stage in ('predict', None):
            if self.df_predict is None:
                raise RuntimeError('Setting up for predict stage, but no predict dataset is provided.')
            self.predict_set = self.dataset_class(self.df_predict, self.transformer, **self.dataset_kwargs)
        
        else:
            raise RuntimeError(f'setup called with stage {stage} used which is invalid.')
        
    def update(self, **kwargs) -> None:
        
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
                    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.train_dataloader_kwargs) if self.train_set is not None else None

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, **self.val_dataloader_kwargs) if self.val_set is not None else None

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, **self.test_dataloader_kwargs) if self.test_set is not None else None
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_set, **self.predict_dataloader_kwargs) if self.predict_set is not None else None