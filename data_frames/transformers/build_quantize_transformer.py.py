import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer



class BinaryToExtremes(BaseEstimator, TransformerMixin):
    """
    Map binary-ish inputs to {-127, +127}.
    Threshold at 0.5 so it’s robust to float dtype.
    """
    #this is required by scikit-learn api
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        #convert data into np array
        #if values >= .5 --> 127
        #if values <= .5 --> -127
        return np.where(np.asarray(X, float) >= 0.5, 127, -127).astype(np.int16)

#zimin needs to read through this 
class EvenlySpacedDiscreteMapper(BaseEstimator, TransformerMixin):
    """
    For small-cardinality integer columns, map each unique value to an
    evenly spaced integer in [-127, 127], preserving order of uniques.
    """
    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel())
        s = s.dropna()
        self.uniques_ = np.sort(s.unique())
        k = len(self.uniques_)
        # Protect against degenerate column (all NaN)
        if k == 0:
            self.map_ = {}
        else:
            # Evenly space k codes across [-127, 127]
            codes = np.linspace(-127, 127, num=k, endpoint=True)
            self.map_ = {u: int(np.rint(c)) for u, c in zip(self.uniques_, codes)}
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        mapped = s.map(self.map_)
        # Handle unseen values by clipping to nearest trained unique
        if mapped.isna().any() and len(self.map_) > 0:
            uniq = self.uniques_
            def nearest_code(v):
                i = np.clip(np.searchsorted(uniq, v), 0, len(uniq)-1)
                return self.map_[uniq[i]]
            mapped = mapped.where(mapped.notna(), s.apply(nearest_code))
        return mapped.to_numpy(dtype=np.int16).reshape(-1, 1)

def build_quantizer(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Binary (0/1) columns
    bin_cols = [c for c in numeric_cols
                if df[c].dropna().isin([0,1]).all()]
    
    # Small-cardinality integers (<=10 unique, integer dtype)
    # small_int_cols = [c for c in numeric_cols
    #                   if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= 10 and c not in bin_cols]

    # # Count-like (non-negative integers with wider range)
    # count_cols = [c for c in numeric_cols
    #               if pd.api.types.is_integer_dtype(df[c])
    #               and c not in bin_cols + small_int_cols
    #               and df[c].min() >= 0]

    # # Bounded ratios in [0,1]
    # bounded_cols = [c for c in numeric_cols
    #                 if df[c].min() >= 0 and df[c].max() <= 1 and c not in bin_cols]

    # The rest of continuous numerics (floats etc.)
    continuous_cols = list(
        set(numeric_cols)
         - set(bin_cols))
        #  - set(small_int_cols) 
        #  - set(count_cols) - set(bounded_cols))

    
    binary_pipe = BinaryToExtremes()
    continuous_pipe = _quantile_uniform_then_int(len(df))


    preproc = ColumnTransformer(
        transformers=[
            ("bin_to_extremes",   bin_pipe,          bin_cols),
            ("small_int_even",    small_int_pipe,    small_int_cols),
            ("continuous_quant",  continuous_pipe,   continuous_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # so names are just original column names
    )

    return preproc
    