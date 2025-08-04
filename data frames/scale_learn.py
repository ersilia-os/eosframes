from scalarizer import scalarize

class Scale:
    def __init__(self, robust_scalar, power_transformer):
        self.robust_scalar = robust_scalar
        self.power_transformer = power_transformer
        self.transform = False

    def fit(self, df):
        df_scaled = scalarize(df, self.power_transformer, self.robust_scalar) # Using RobustScaler
        self.transform = True
        return df_scaled
        #return transformer?

    def inference(self, df):
        if self.transform:
            return self.scaler_.transform(df)
        raise ValueError("You must call fit() before inference()")
    