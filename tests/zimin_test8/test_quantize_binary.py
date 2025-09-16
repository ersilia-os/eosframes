from data_frames.transformers.quantize import Quantize
import pandas as pd

df = pd.read_csv("test_binary_constant_fit.csv")

kbd = Quantize(model_id="eos78ao")

fitted_df = kbd.fit(df) 
fitted_df.to_csv("test_fitted_binary.csv")
# Save the model
kbd.save('modred_dir')

# Later, load the model
loaded_kbd = kbd.load('modred_dir')
# Use the loaded model for inference
df_inference = pd.read_csv("test_binary_constant.csv")

result = loaded_kbd.transform(df_inference)
result.to_csv('binary-def-contin.csv', index=False)
