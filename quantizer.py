from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

numeric_df = df_scaled.select_dtypes(include=['number'])

X = numeric_df.to_numpy()
n_bins = 256

# Choose 'uniform' or 'quantile' strategy
kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_binned = kbd.fit_transform(X)

print(X_binned.astype(int))