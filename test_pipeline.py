from data_frames.scale import Scale
import pandas as pd
import joblib

# 1. Fit on "training" DataFrame
scaler = Scale(robust_scalar=True, power_transform=False)
df_train = pd.read_csv("first_500_output.csv")

# 2. pick only the descriptor columns (everything except “key” and “input”)
feature_cols = [c for c in df_train.columns if c not in ("key","input")]
print(df_train[feature_cols])
df_train_scaled = scaler.fit(df_train[feature_cols])

# 2. Save the entire pipeline for later
joblib.dump(scaler.pipeline_, "mordred.pkl")

loaded_pipeline = joblib.load("mordred.pkl")
df_new = pd.read_csv("new_compounds_output.csv")
df_new_scaled = loaded_pipeline.transform(df_new.select_dtypes("number"))