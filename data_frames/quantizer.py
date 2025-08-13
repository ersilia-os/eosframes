import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler


def bin() -> KBinsDiscretizer: 
    """
    Creates a KBinsDiscretizer for quantizing numerical features.
    
    Returns:
    KBinsDiscretizer to be used for inference. 
    """
    
    n_bins = 256
    bin_edges = np.linspace(-128, 127, n_bins + 1)

    # Choose 'uniform' or 'quantile' strategy
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
    kbd.fit(bin_edges.reshape(-1, 1))
    
    return kbd