import pandas as pd
from data_frames.transformers.scale import Scale
from data_frames.transformers.build_typed_transformer import build_typed_transformer

df = pd.read_csv("output_all.csv")
feature_cols = [c for c in df.columns if c not in ("key", "input")]
df = df[feature_cols]

# === Train & save to S3 ===
scaler = Scale(model_id="eos78ao")
df_scaled = scaler.fit(df)
print("Training complete. Scaled shape:", df_scaled.shape)

# scaler.save()
# print("Saved eos78ao to S3 (ersilia-dataframes).")

# === Load back from S3 ===
loaded = Scale.load(model_id="eos78ao", bucket_name="ersilia-dataframes")
print("Loaded eos78ao back from S3.")

# === Inference ===
df_sample = df.head(5)
df_inferred = loaded.inference(df_sample)
print("Inference result on 5 rows:")
print(df_inferred)