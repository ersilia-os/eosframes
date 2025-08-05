import pandas as pd
from data_frames.scalarizer import make_scalarizer
import numpy as np

class Scale:
    def __init__(
            self, 
            robust_scalar: bool = False, 
            power_transform: bool=False):
        self.robust_scalar = robust_scalar
        self.power_transform = power_transform
        self.pipeline_ = make_scalarizer(
            power_transfrom=self.power_transform,
            robust_scaler=self.robust_scalar
            )
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        # Ensure only numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols == 0):
            raise ValueError("No numeric columnds to transform.")
        
        X = self.pipeline_.fit_transform(df[numeric_cols].to_numpy())
        self._is_fitted = True

        return pd.DataFrame(
            X,
            infex=df.index,
            columns=numeric_cols
        )

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with the already-fitted pipeline
        """
        if not self._is_fitted:
            raise RuntimeError("You must call .fit() before .inference()")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columnds to transform.")
       
        X = self.pipeline_transform(df[numeric_cols].to_numpy())
        return pd.DataFrame(
            X,
            infex=df.index,
            columns=numeric_cols
        )

    