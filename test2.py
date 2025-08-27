from data_frames.transformers.scale import Scale
import pandas as pd

mordred = Scale(model_id="eos78ao")
df = pd.read_csv("drugbank_output.csv")
mordred.fit(df)
mordred.save(dir_name="eos78ao_2")