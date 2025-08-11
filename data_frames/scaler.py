import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline

def make_scaler(
        power_transform: bool = False,
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
    if power_transform:
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=True)))
    if robust_scaler:
        steps.append(("standard", RobustScaler()))
    
    return Pipeline(steps)