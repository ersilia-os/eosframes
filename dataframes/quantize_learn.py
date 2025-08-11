from quantizer import quantize

class Quantize:
    def __init__(self, robust_scaler, power_transformer):
        self.robust_scaler = robust_scaler
        self.power_transformer = power_transformer

    def fit(self, df):
        df_scaled = scalarize(df, self.power_transformer, self.robust_scaler) # Using RobustScaler
        return df_scaled