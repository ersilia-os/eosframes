import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler


def bin(
        df: pd.DataFrame, 
        robust_scalar: bool = True, 
        power_transform: bool = False
        ) -> pd.DataFrame: 
    """
    Quantizes the numerical features of a DataFrame into discrete bins.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to quantize.
    
    Returns:
    pd.DataFrame: DataFrame with quantized numerical features.
    """
    """
    Scales the numerical features of a DataFrame using RobustScaler or .
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to scale.
    
    Returns:
    pd.DataFrame: DataFrame with scaled numerical features.
    """
    #SCALING
    
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
       
    # Ensure only numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:  # Fixed the bug
        raise ValueError("No numeric columns to transform.")
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df[numeric_cols])

    if power_transform:      
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        X = pt.fit_transform(X)

    if robust_scalar:
        rs = RobustScaler()
        X = rs.fit_transform(X)

    df_scaled = pd.DataFrame(
        X,
        infex=df.index,
        columns=numeric_cols
    )
    
    n_bins = 256

    # Choose 'uniform' or 'quantile' strategy
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    
    return kbd