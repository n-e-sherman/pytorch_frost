import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalToStringImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, fill_value='missing'):
        self.fill_value = fill_value
        self.column = column
    
    def fit(self, X, y=None):
        # No fitting process needed for this transformer
        return self
    
    def transform(self, X):
        # Convert all columns to string, handling missing values
        X_transformed = X.astype(str).fillna(self.fill_value)
        return X_transformed
    
    def inverse_transform(self, X):
        # Replace the custom fill_value with np.nan
        # Note: The inverse operation assumes X is a DataFrame for simplicity
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[self.column])
        
        X_reverted = X.replace(self.fill_value, np.nan)
        return X_reverted