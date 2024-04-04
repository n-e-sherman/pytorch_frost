from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Callable
from .utils import CategoricalToStringImputer
import numpy as np
import pandas as pd
import warnings

PandasDataFrame = pd.DataFrame

class DataEncoder:
    
    def __init__(self, 
                 numerical_columns: list=[],
                 categorical_columns: list=[],
                 passthrough_columns: list=[], # passthrough columns should not be part of the input to the model directly.
                 custom_columns: dict={}, # not implemented yet
                 numerical_transform: str='normalize', # other option is standardize
                 numerical_pad: float=np.nan,
                 categorical_pad: str='<PAD>',
                 passthrough_pad: float=np.nan,
                 force_min: Optional[dict]=None,
                 force_max: Optional[dict]=None,
    ) -> None:
        
        '''
            custom_columns allows for treating any column in a specific way.
            format should be a dictionary of the form {'key' : custom_column}
            and custom_column is a dictionary with structure
            {
                'transformer': <transformer>,
                'pad' : <pad_value>,
            }
            If 'pad' is not given, it will be taken to be the numerical_pad.
            If custom_column is just a transformer and not a dictionary, then
            the pad value will be taken to be the numerical_pad.
            
        '''

        num_and_cat = set(numerical_columns).intersection(set(categorical_columns))
        assert len(num_and_cat) == 0, f"{num_and_cat} were provided in both the numerical and categorical columns."

        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns)
        self.passthrough_columns = list(passthrough_columns)
        self.custom_columns = dict(custom_columns)
            
        if numerical_transform == 'normalize':
            numerical_transform = MinMaxScaler
        elif numerical_transform == 'standardize':
            numerical_transform = StandardScaler
        else:
            raise NotImplementedError(f'numerical_transform {numerical_transform} is not implemented.')
        
        for col, val in custom_columns.items():
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                warnings.warn(f"column {col} was provided in both numerical_columns and custom_columns. Removing from numerical_columns.")
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                warnings.warn(f"column {col} was provided in both categorical_columns and custom_columns. Removing from categorical_columns.")
            if col in self.passthrough_columns:
                self.passthrough_columns.remove(col)
                warnings.warn(f"column {col} was provided in both passthrough_columns and custom_columns. Removing from passthrough_columns.")
                
            if isinstance(val, dict):
                pad = val.get('pad', numerical_pad)
                transformer = val.get('transformer', numerical_transform())
                self.custom_columns[col] = {'transformer': transformer, 'pad': pad}
            else:
                self.custom_columns[col] = {'transformer': val, 'pad': numerical_pad}
        
        for col in passthrough_columns:
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                warnings.warn(f"column {col} was provided in both numerical_columns and passthrough_columns. Removing from numerical_columns.")
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                warnings.warn(f"column {col} was provided in both categorical_columns and passthrough_columns. Removing from categorical_columns.")
                
        self.force_min = force_min if force_min is not None else {}
        self.force_max = force_max if force_max is not None else {}
                    
        # Define transformers
        transformers = []
        self.pad_values = {}
        self._column_order = []
        if self.numerical_columns:
            transformers.append(('numerical', numerical_transform(), self.numerical_columns))
            self._column_order += self.numerical_columns
            for col in numerical_columns:
                self.pad_values[col] = numerical_pad
                    
        for col in self.categorical_columns:
            categorical_preprocessing = Pipeline(steps=[
                ('imputer', CategoricalToStringImputer(column=col, fill_value=str(categorical_pad))),
                ('encoder', OrdinalEncoder())
            ])
            transformers.append((col, categorical_preprocessing, [col]))
            self.pad_values[col] = categorical_pad
            self._column_order.append(col)
            
        if self.passthrough_columns:
            self._column_order += self.passthrough_columns
            transformers.append(('passthrough', 'passthrough', self.passthrough_columns))
            for col in self.passthrough_columns:
                self.pad_values[col] = passthrough_pad
                
        for col, val in self.custom_columns.items():
            self._column_order.append(col)
            transformers.append((col, val['transformer'], [col]))
            self.pad_values[col] = val['pad']
                
        self._fitted = False
        self.transformer = ColumnTransformer(transformers)

    def fit(self, pdf_train: PandasDataFrame, pdf_full: Optional[PandasDataFrame]=None) -> None:

        '''
            Train the ColumnTransformer
            Set embedding info
            Set pad values
            Set column_order?
        '''
        
        # add categorical variables to the encoder list to give them an index
        pdf_add = pd.DataFrame()
        if pdf_full is not None:
            add_columns = {}
            N_col = 0
            for col in self.categorical_columns:
                unique_full = pdf_full[col].unique()
                unique_train = pdf_train[col].unique()
                new_column = [x for x in unique_full if x not in unique_train]
                N_col = np.max([N_col, len(new_column)])
                add_columns[col] = new_column
            
            # need to fill pdf_add with new categorical variables.
            if N_col > 0:
                for col, val in add_columns.items():
                    N_val = len(val)
                    add_columns[col] = list(val) + [self.pad_values[col]]*(N_col - N_val)
                pdf_add = pd.DataFrame(add_columns)
                
        pdf_pad = pd.DataFrame([self.pad_values])
        self.transformer.fit(pd.concat([pdf_pad, pdf_train, pdf_add])) # put pad first so padding_idx=0.
        
        self._set_pad_values()
        self._fitted = True
        
    def encode(self, pdf_in: PandasDataFrame) -> PandasDataFrame:

        pdf = pdf_in.copy()
        columns_fill = self._fill_pdf(pdf)
        res = self.transformer.transform(pdf[self.column_order])
        res = pd.DataFrame(res, columns=self.column_order)
        res = res.drop(columns_fill, axis=1)
        for col in self.numerical_columns:
            res[col] = res[col].astype(float)
        for col in self.categorical_columns:
            res[col] = res[col].astype(int)
        return res
        
    def decode(self, pdf_in: pd.DataFrame, keep_columns: bool=True) -> pd.DataFrame:
        
        '''
            Note: all categorical columns will be strings after using decode.
        '''

        pdf = pdf_in.copy()
        columns_fill = self._fill_pdf(pdf)
        data = []
        columns = []
        
        # Inverse transform for numerical columns
        if self.numerical_columns:
            data.append(self.transformer.named_transformers_['numerical'].inverse_transform(
                pdf[self.numerical_columns]))
            columns += self.numerical_columns

        # Inverse transform for categorical columns
        for col in self.categorical_columns:
            _data = self.transformer.named_transformers_[col]['encoder'].inverse_transform(
                pdf[[col]])
            _data = self.transformer.named_transformers_[col]['imputer'].inverse_transform(_data)
            data.append(_data)
        columns += self.categorical_columns
        
        for col in self.custom_columns:
            data.append(self.transformer.named_transformers_[col].inverse_transform(pdf[[col]]))
            columns.append(col)

        # Inverse transform for passthrough columns
        if self.passthrough_columns:
            data.append(pdf[self.passthrough_columns].values)
            columns += self.passthrough_columns

        # passthrough any extra columns untransformed
        used_columns = list(set(columns + columns_fill))
        extra_columns = [col for col in pdf.columns if col not in used_columns]
        if keep_columns:
            data.append(pdf[extra_columns].values)
            columns += extra_columns

        # Combine the inversely transformed columns back into a DataFrame
        res = pd.DataFrame(np.column_stack(data), columns=columns).infer_objects()
        return res.drop(columns_fill, axis=1)
    
    @property
    def column_order(self):
        return self._column_order
    
    @property
    def encoded_pad_values(self):
        return self._encoded_pad_values
        
    def _set_pad_values(self) -> None:
    
        pdf = pd.DataFrame([self.pad_values])
        pdf = pdf[self.column_order]
        _pdf = self.encode(pdf)
        self._encoded_pad_values = _pdf.to_dict('records')[0]
        for col in self.categorical_columns:
            self._encoded_pad_values[col] = int(self._encoded_pad_values[col])
            
    def _fill_pdf(self, pdf: PandasDataFrame) -> list:

        ''' If initial columns don't include all required, fill with pads
            If initial columns include extra, trim them
        '''
        
        columns_in = pdf.columns
        columns_fill = [col for col in self.column_order if col not in columns_in]
        for col in columns_fill:
            pdf.loc[:, col] = self.pad_values[col]
        return columns_fill