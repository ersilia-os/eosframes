from data_frames.transformers.quantize import Quantize
import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("output_all.csv")
df_feature1 = df.iloc[:, [2]] #column 2 is the feature "abc"

kbd = Quantize(robust_scaler=True, power_transform=False)

fitted_df = kbd.fit(df_feature1) 
# Save the model
kbd.save('modred_dir')


# Later, load the model
loaded_kbd = kbd.load('modred_dir')
# Use the loaded model for inference
df_inference = pd.read_csv("ten_inf_numeric.csv")
df_inference1 = df_inference.iloc[:, [3]] #column 2 is the feature "abc"


result = loaded_kbd.inference(df_inference1)
result.to_csv('inf_five.csv', index=False)
