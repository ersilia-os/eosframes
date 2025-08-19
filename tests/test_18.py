import pandas as pd
from data_frames.scale import Scale

# Load the saved transformer from S3
mordred = Scale.load(
    model_id="eos78ao",
    transformer_type="robust_scaler",
    bucket_name="ersilia-dataframes"
)

# Example: run inference on new data
# (make sure df_new has the exact feature columns used in training)
df_new = pd.read_csv("drugbank_outputc5.csv")
df_scaled = mordred.inference(df_new)

print(df_scaled.head())
