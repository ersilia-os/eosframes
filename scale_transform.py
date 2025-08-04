from scalarizer import scalarize

class Scale:
    def __init__(self, robust_scalar, power_transformer):
        self.robust_scalar = robust_scalar
        self.power_transformer = power_transformer

    def fit(self, df):
        df_scaled = scalarize(df, self.power_transformer, self.robust_scalar) # Using RobustScaler
        return df_scaled
        #return transformer?

#from scalarizer import scalarize

# class Scale:
#     def __init__(self, robust_scalar, power_transformer):
#         self.robust_scalar = robust_scalar
#         self.power_transformer = power_transformer
#         self.scaler_ = None  # Will store fitted transformer

#     def fit(self, df):
#         self.scaler_ = scalarize(df, self.power_transformer, self.robust_scalar)
#         return self

#     def transform(self, df):
#         if self.scaler_ is None:
#             raise ValueError("You must call fit() before transform()")
#         return self.scaler_.transform(df)

#     def fit_transform(self, df):
#         return self.fit(df).transform(df)