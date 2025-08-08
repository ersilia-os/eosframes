from data_frames.scale import Scale
import pandas as pd
import joblib
import matplotlib.pyplot as plt


scale = Scale(robust_scalar=True, power_transform=True)
fitted_df = scale.fit("outputall.csv")

# Save the model
scale.save_model("my_model_directory")

# Later, load the model
loaded_scale = Scale.load_model("my_model_directory")

# Use the loaded model for inference
result = loaded_scale.inference()
result.to_csv('new_inference_scaled.csv', index=False)

# # Load CSV
# df = pd.read_csv("output_all.csv", skiprows=1) #fix the label(non numerical row)


# #data formating
# extra_cols = df.iloc[:, [0, 1]] #keep columns to add back later
# df_numeric = df.drop(df.columns[[0, 1]], axis=1) #drop the first and second column

# #instantiate the Scale class
# scaler = Scale(robust_scalar=True, power_transform=False)

# # Initialize transformer
# df_train_scaled =  scaler.fit(df_numeric)#using robust scalar and no power transformer

# #FITTING
# joblib.dump(scaler.pipeline_, 'Fit_Modred_Z')

# # #plotting data
# # assert isinstance(df_numeric, pd.DataFrame)
# # #format the data correctly 
# # df_final = pd.concat([extra_cols.reset_index(drop=True), df_scaled], axis=1)
# # df_final.to_csv('new_scaled.csv', index=False)

# #PREDICTION OR INFERENCE TIME

# transformer = joblib.load('Fit_Modred_Z') #transformer is a robust scalar

# new_df = pd.read_csv("ten_inference.csv")
# new_x = transformer.transform(new_df) #transform is built-in function of scikit-learn transformer for robust scalar

# new_x.to_csv('new_inference_scaled.csv', index=False)
