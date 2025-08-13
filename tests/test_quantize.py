from data_frames.quantize import Quantize
import pandas as pd
import joblib
import matplotlib.pyplot as plt




df = pd.read_csv('output_all.csv')
df_feature1 = df.iloc[:, [2]] #column 2 is the feature "abc"

kbd = Quantize(robust_scaler=True, power_transform=False)

fitted_df = kbd.fit(df_feature1) 
fitted_df.to_csv('quantized_all.csv')

# Save the model
kbd.save("model_directory")


# Later, load the model
loaded_kbd = kbd.load("model_directory")
# Use the loaded model for inference
df_inference = pd.read_csv('ten_inf_numeric.csv')
df_inference1 = df_inference.iloc[:, [3]] #column 2 is the feature "abc"


result = loaded_kbd.inference(df_inference1)
result.to_csv('new_inference_quantized.csv', index=False)
