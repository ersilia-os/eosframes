import pandas as pd
from data_frames.scalarizer import make_scalarizer

class Scale:
    def __init__(
            self, 
            robust_scalar: bool = False, 
            power_transform: bool=False
    ):
        self.pipeline_ = make_scalarizer(
            power_transfrom=power_transform,
            robust_scaler=robust_scalar
            )
        self._is_fitted = False
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        
        # Ensure only numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columnds to transform.")
        
        X_trans = self.pipeline_.fit_transform(df[numeric_cols])
        self._is_fitted = True
        self.feature_cols = list(numeric_cols)

        return pd.DataFrame(
            X_trans,
            index=df.index,
            columns=self.feature_cols
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
       
        X_new = self.pipeline_.transform(df[numeric_cols].to_numpy())
        return pd.DataFrame(
            X_new ,
            index=df.index,
            columns=self.feature_cols
        )

    