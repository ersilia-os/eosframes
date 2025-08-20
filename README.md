# data-frames
data-frames is a Python package for fitting data transformation pipelines and applying them to new datasets. It is built for reproducible preprocessing of numeric data—including median imputation, robust scaling, Yeo-Johnson power transforms, and quantization.

 ## Installation

 To get started, create a Conda environment:

```bash
conda create -n my_env python=3.12
conda activate my_env
```
Then install the package using pip:

```bash
pip install git+https://github.com/ersilia-os/data-frames.git
```

## Usage

Here’s an example of training and saving a typed transformer on your dataset.

```python
import pandas as pd
from data_frames.transformers.scale import Scale

# Load your dataset
df = pd.read_csv("drugbank_output.csv")

# Select only feature columns (skip identifiers if present)
feature_cols = [c for c in df.columns if c not in ("key", "input")]
df = df[feature_cols]

# Initialize Scale with a model ID (e.g. eos78ao)
scaler = Scale(model_id="eos78ao")

# Fit the transformer
df_scaled = scaler.fit(df)
print("Scaled shape:", df_scaled.shape)

# Save the model (locally and to S3 bucket 'ersilia-dataframes')
scaler.save()
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)