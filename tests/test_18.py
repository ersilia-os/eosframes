import pandas as pd
from data_frames.transformers.scale import Scale

df = pd.read_csv("test_missing_cols.csv")
feature_cols = [c for c in df.columns if c not in ("key", "input")]
df = df[feature_cols]

# === Train & save to S3 ===
s = Scale(model_id="eos78ao")
df_scaled = s.fit(df)
# print("Training complete. Scaled shape:", df_scaled.shape)

# s.save("eos78ao", True)
# # print("Saved eos78ao to S3 (ersilia-dataframes).")

# # === Load back from S3 ===
# loaded = Scale.load(model_id="eos78ao", bucket_name="ersilia-dataframes")
# print("Loaded eos78ao back from S3.")

# === Inference ===
df_sample = pd.read_csv("test_missing_cols.csv")
df_inferred = s.transform(df_sample)
df_inferred.to_csv("result_missing")