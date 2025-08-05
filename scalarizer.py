#!pip install scikit-learn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler


def scalarize(
        df: pd.DataFrame, 
        robust_scalar: bool = False, 
        power_transform: bool = False
        ) -> pd.DataFrame:
    """
    Scales the numerical features of a DataFrame using RobustScaler or .
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to scale.
    
    Returns:
    pd.DataFrame: DataFrame with scaled numerical features.
    """
    
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
       
    # Ensure only numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols == 0):
        raise ValueError("No numeric columnds to transform.")

    # Ensure that the DataFrame contains only numeric columns
    if not all(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        raise ValueError("DataFrame must contain only numeric columns.")
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df[numeric_cols])

    if power_transform:      
        # Initialize the transformer
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        # Box-Cox only supports positive values
        # Yeo-Johnson supports both positive and negative values

        # Fit and transform the data
        X = pt.fit_transform(X)

    if robust_scalar:
        rs = RobustScaler()
        X = rs.fit_transform(X)

    df_scaled = pd.DataFrame(
        X,
        infex=df.index,
        columns=numeric_cols
    )

    return df_scaled