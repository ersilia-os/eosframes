#!pip install scikit-learn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler


def scalerize(
        df: pd.DataFrame, 
        robust_scaler: bool = False, 
        power_transform: bool = False
        ) -> pd.DataFrame:
    """
    Scales the numerical features of a DataFrame using RobustScaler or Power Transformer.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to scale.
    
    Returns:
    pd.DataFrame: DataFrame with scaled numerical features.
    """
    
    # Ensure only numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns to transform.")
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df[numeric_cols])

    if power_transform:      
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        X = pt.fit_transform(X)

    if robust_scaler:
        rs = RobustScaler()
        X = rs.fit_transform(X)

    return pd.DataFrame(X, index=df.index, columns=numeric_cols)