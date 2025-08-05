import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline

def make_scalarizer(
        power_transfrom: bool = False,
        robust_scaler: bool = False
) -> Pipeline:
    """
    Build a sklearn Pipeline which:
        (a) median-imputes missing values on numeric columns
        (b) applies Yeo-Johnson PowerTransformer (if specified)
        (c) applies RobustScaler (if specified)
    The caller is responsible for selecting numeric columns beforehand.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if power_transfrom:
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=True)))
    if power_transfrom:
        steps.append(("scale", RobustScaler()))
    
    return Pipeline(steps)