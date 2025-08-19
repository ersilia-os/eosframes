from data_frames.scale import Scale

modred = Scale.load(model_id="eos78ao", transformer_type="robust_scaler", bucket_name="ersilia-dataframes")

mordred.inference()