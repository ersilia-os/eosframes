from data_frames.scale import Scale
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Fit on "training" DataFrame
scaler = Scale(robust_scaler=True, power_transform=False)
df_train = pd.read_csv("first_500_output.csv")

# 2. pick only the descriptor columns (everything except “key” and “input”)
feature_cols = [c for c in df_train.columns if c not in ("key","input")]
# print(df_train[feature_cols])
df_train_scaled = scaler.fit(df_train[feature_cols])

# 2. Save the entire pipeline for later
joblib.dump(scaler, "mordred.pkl")

scaler2 = joblib.load("mordred.pkl")
df_new = pd.read_csv("new_compounds_output.csv")
df_new_scaled = scaler2.inference(df_new)

# 4) Helper to plot a 2×3 grid of histograms
def plot_grid(df, title):
    cols = df.columns[:6]   # first six features
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col], bins=30)
        ax.set_title(f"{title}: {col}")
        ax.set_xlabel("scaled value")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.show()

# Plot train distributions
plot_grid(df_train_scaled, "Train")

# Plot inference distributions
plot_grid(df_new_scaled, "Inference")