import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, QuantileTransformer, PowerTransformer
from sklearn.pipeline import Pipeline

def build_typed_transformer(df: pd.DataFrame):
    # Heuristics to group columns
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

    # Transformers
    log1p_then_scale = Pipeline([
        ("log1p", FunctionTransformer(lambda x: np.log1p(x))),
        ("scale", StandardScaler(with_mean=True))
    ])

    yeojohnson_then_scale = Pipeline([
        ("yj", PowerTransformer(method="yeo-johnson", standardize=True))
    ])

    # For bounded, a quantile→normal often works well for linear models
    quantile_normal = Pipeline([
        ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, max(10, len(df)//3))))
    ])

    passthrough = "passthrough"

    preproc = ColumnTransformer(
        transformers=[
            ("bin_passthrough", passthrough, bin_cols),
            ("smallint_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), small_int_cols),
            ("counts_log1p_scale", log1p_then_scale, count_cols),
            ("bounded_quantile", quantile_normal, bounded_cols),
            ("continuous_yj", yeojohnson_then_scale, continuous_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preproc, {
        "binary": bin_cols,
        "small_int": small_int_cols,
        "counts": count_cols,
        "bounded": bounded_cols,
        "continuous": continuous_cols
    }