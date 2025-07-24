import sys
sys.path.append("/Users/ziminqian/Desktop/ersilia")
from ersilia.api import ErsiliaModel
import pandas as pd
from transform import Transform

#cd /path/to/ersilia
#pip install -e .

# df = csv to data frame

transformer = Transform(n_bins=256)
df_transformed = transformer.fit_transform(df)

# Simple assertions
assert isinstance(df_transformed, pd.DataFrame)
assert df_transformed.shape == df.shape
assert df_transformed.min().min() >= -128
assert df_transformed.max().max() <= 127

print(df_transformed)