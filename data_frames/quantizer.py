from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd


def quantize(df) :
    """
    Quantizes the numerical features of a DataFrame into discrete bins.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical features to quantize.
    
    Returns:
    pd.DataFrame: DataFrame with quantized numerical features.
    """
    
    # Check if the DataFrame is empty
    if df.empty:
        return df

    numeric_df = df.select_dtypes(include=['number'])

    X = numeric_df.to_numpy()
    n_bins = 256

    # Choose 'uniform' or 'quantile' strategy
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_binned = kbd.fit_transform(X)

    X_binned_centered = X_binned.astype(int) - (n_bins // 2 - 1)

    #convert to data frame
    X_binned_centered_df = pd.DataFrame(
    X_binned_centered,
    columns=numeric_df.columns,
    index=numeric_df.index
    )
    
    return X_binned_centered_df