import pandas as pd
from data_frames.transformers.scale import Scale

mordred = Scale.load(model_id="eos78ao", bucket_name="ersilia-dataframes")

# Run inference on new data
df_new = pd.read_csv("esol_subset_200_output.csv")
df_scaled = mordred.inference(df_new)

import matplotlib.pyplot as plt
import numpy as np

# Step 1: reuse same grouping logic on raw data
numeric_cols = df_new.select_dtypes(include="number").columns.tolist()

bin_cols = [c for c in numeric_cols if df_new[c].dropna().isin([0, 1]).all()]

small_int_cols = [
    c for c in numeric_cols
    if pd.api.types.is_integer_dtype(df_new[c])
    and df_new[c].nunique(dropna=True) <= 10
    and c not in bin_cols
]

count_cols = [
    c for c in numeric_cols
    if pd.api.types.is_integer_dtype(df_new[c])
    and c not in bin_cols + small_int_cols
    and df_new[c].min() >= 0
]

bounded_cols = [
    c for c in numeric_cols
    if df_new[c].min() >= 0 and df_new[c].max() <= 1 and c not in bin_cols
]

continuous_cols = list(
    set(numeric_cols)
    - set(bin_cols)
    - set(small_int_cols)
    - set(count_cols)
    - set(bounded_cols)
)

groups = {
    "Binary": bin_cols,
    "Small Int": small_int_cols,
    "Count-like": count_cols,
    "Bounded [0,1]": bounded_cols,
    "Continuous": continuous_cols,
}

# Step 2: sample 3–5 from each group and plot
for group_name, cols in groups.items():
    if not cols:
        continue
    
    sample_size = min(len(cols), 5)  # up to 5 features
    sampled_cols = np.random.choice(cols, size=sample_size, replace=False)
    
    plt.figure(figsize=(10, 6))
    df_scaled[sampled_cols].boxplot(rot=45)
    plt.title(f"Boxplots of {group_name} Features (Scaled) - {sample_size} samples",
              fontsize=14, weight="bold")
    plt.ylabel("Scaled Values")
    plt.tight_layout()
    plt.show()

