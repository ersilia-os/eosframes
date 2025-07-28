import sys
sys.path.append("/Users/ziminqian/Desktop/ersilia")

import pandas as pd
from transform import Transform

# Load CSV
df = pd.read_csv("output_five.csv", skiprows=1) #fix the label(non numerical row)
df = df.drop(df.columns[[0, 1]], axis=1) #drop the first and second column
# df_numeric = df.select_dtypes(include=["number"]) # Drop any rows with non-numeric values
print(df.head(10))
# # Initialize transformer
transformer = Transform(False, True)

# # Run the transformation
df_transformed = transformer.run(df)

# # Assertions on the transformed DataFrame
assert isinstance(df_transformed, pd.DataFrame)
assert df_transformed.shape == df.shape
# assert df_transformed.min().min() >= -128
# assert df_transformed.max().max() <= 127

df_transformed.to_csv('Scaled_Quantized_output.csv', index=False)