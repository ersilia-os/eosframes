import pandas as pd
from data_frames.transformers.scale import Scale
from data_frames.transformers.build_typed_transformer import build_typed_transformer

df_train = pd.read_csv("files/first_500_output.csv")
X = df_train.drop(columns=["key","input"], errors="ignore")
preproc, groups = build_typed_transformer(X)
preproc.fit(X)    # persist with joblib.dump(preproc, "eos78ao_typed_preproc.joblib")
X_trans = preproc.transform(X)
