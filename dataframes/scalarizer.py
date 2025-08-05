#!pip install scikit-learn
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer


def scalarize(df, robust_scalar, power_transform):
    """
    Scales the numerical features of a DataFrame using RobustScaler or a PowerTransformer.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to scale.
    
    Returns:
    Tuple[scaler, df_scaled]: The fitted transformer and the transformed DataFrame
    """
    
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
       
    # Ensure only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_cols]

    # Ensure that the DataFrame contains only numeric columns
    if not all(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        raise ValueError("DataFrame must contain only numeric columns.")
    
    #Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)

    df_scaled = df_imputed.copy() 

    if robust_scalar:
        scaler = RobustScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])

    if power_transform:      
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        #Box-Cox only supports positive values
        # Yeo-Johnson supports both positive and negative values
        df_scaled[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])


    return scaler, df_scaled