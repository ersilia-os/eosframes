from quantizer import quantize

class Quantize:
    def __init__(self, robust_scalar, power_transformer):
        self.robust_scalar = robust_scalar
        self.power_transformer = power_transformer

    def fit(self, df):
        df_scaled = scalarize(df, self.power_transformer, self.robust_scalar) # Using RobustScaler
        return df_scaled