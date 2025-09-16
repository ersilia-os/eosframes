import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------- Helper functions (no lambdas, so joblib can pickle) ----------
def log1p_transform(x):
    return np.log1p(x)

def identity(x):
    return x  # useful for passthrough cases if needed

# -------------------------------------------------------------------------

def build_typed_transformer(df: pd.DataFrame):

    # Replace constant columns with 0
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.copy()
    df[constant_cols] = 0
    
    # Heuristics to group columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    # Binary (0/1) columns
    bin_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1]).all()]


    # Count-like (non-negative integers with wider range)
    count_cols = [
        c for c in numeric_cols
        if pd.api.types.is_integer_dtype(df[c])
        and c not in bin_cols 
        and df[c].min() >= 0
    ]

    # Bounded ratios in [0,1]
    bounded_cols = [
        c for c in numeric_cols
        if df[c].min() >= 0 and df[c].max() <= 1 and c not in bin_cols
    ]

    # Continuous numerics
    continuous_cols = list(
        set(numeric_cols)
        - set(bin_cols)
        - set(count_cols)
        - set(bounded_cols)
    )

    # Transformers
   
    minmax_scaler = Pipeline([
    ("mm", MinMaxScaler(feature_range=(0, 1)))
    ])
    
    robust_scaler = Pipeline([
    ("rs", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)))
    ])

    quantile_normal = Pipeline([
        ("qt", QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, max(10, len(df) // 3))
        ))
    ])

    # Build ColumnTransformer
    preproc = ColumnTransformer(
        transformers=[
            ("bin_passthrough", "passthrough", bin_cols),
            ("count_logscale", minmax_scaler, count_cols),
            ("bounded_quantile", quantile_normal, bounded_cols),
            ("continuous_rs", robust_scaler, continuous_cols),
        ],
        remainder="drop"
    )

    return preproc