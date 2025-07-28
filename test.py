import sys
sys.path.append("/Users/ziminqian/Desktop/ersilia")

import pandas as pd
from transform import Transform

# Load CSV
df = pd.read_csv("output_five.csv")
df_numeric = df.select_dtypes(include=["number"]) # Drop any rows with non-numeric values

# Initialize transformer
transformer = Transform(False, True, 256)

# Run the transformation
df_transformed = transformer.run(df_numeric)

# Assertions on the transformed DataFrame
assert isinstance(df_transformed, pd.DataFrame)
assert df_transformed.shape == df.shape
assert df_transformed.min().min() >= -128
assert df_transformed.max().max() <= 127

print(df_transformed.head())