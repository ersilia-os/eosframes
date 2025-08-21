from data_frames.transformers.quantize import Quantize
import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("test_binary_constant_fit.csv")

kbd = Quantize(robust_scaler=True, power_transform=False)

fitted_df = kbd.fit(df) 
# Save the model
kbd.save('modred_dir')

# Later, load the model
loaded_kbd = kbd.load('modred_dir')
# Use the loaded model for inference
df_inference = pd.read_csv("test_binary_constant.csv")

result = loaded_kbd.inference(df_inference)
result.to_csv('binary-def-contin.csv', index=False)
