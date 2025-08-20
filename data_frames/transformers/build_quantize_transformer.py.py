import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline



def build_quantizer(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Binary (0/1) columns
    bin_cols = [c for c in numeric_cols
                if df[c].dropna().isin([0,1]).all()]
    
    # Small-cardinality integers (<=10 unique, integer dtype)
    small_int_cols = [c for c in numeric_cols
                      if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= 10 and c not in bin_cols]

    # Count-like (non-negative integers with wider range)
    count_cols = [c for c in numeric_cols
                  if pd.api.types.is_integer_dtype(df[c])
                  and c not in bin_cols + small_int_cols
                  and df[c].min() >= 0]

    # Bounded ratios in [0,1]
    bounded_cols = [c for c in numeric_cols
                    if df[c].min() >= 0 and df[c].max() <= 1 and c not in bin_cols]

    # The rest of continuous numerics (floats etc.)
    continuous_cols = list(set(numeric_cols) - set(bin_cols) - set(small_int_cols) - set(count_cols) - set(bounded_cols))

    return df, kbd? 