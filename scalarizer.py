#!pip install scikit-learn
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer


def scalarize(df, power_transform, robust_scalar):
    """
    Scales the numerical features of a DataFrame using RobustScaler or .
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to scale.
    
    Returns:
    pd.DataFrame: DataFrame with scaled numerical features.
    """
    
    # Check if the DataFrame is empty
    if df.empty:
        return df
    
    # Ensure that the DataFrame contains only numeric columns
    if not all(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        raise ValueError("DataFrame must contain only numeric columns.")
    
    #Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df) 

    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if robust_scalar:
        s = RobustScaler()
        df_scaled = df.copy() #need to define df_scaled variable first 
        df_scaled[numeric_cols] = s.fit_transform(df[numeric_cols])

    if power_transform:      
        # Initialize the transformer
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        #Box-Cox only supports positive values
        # Yeo-Johnson supports both positive and negative values

        # Fit and transform the data
        df_scaled = pd.DataFrame(pt.fit_transform(df), numeric_cols)

    return df_scaled