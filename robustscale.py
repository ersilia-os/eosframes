from scalarizer import scalarize
from quantizer import quantize

class Transform:
    def __init__(self, robust_scalar, power_transformer):
        self.robust_scalar = robust_scalar
        self.power_transformer = power_transformer

    def run(self, df):
        df_scaled = scalarize(df, self.power_transformer, self.robust_scalar) # Using RobustScaler
        df_quantized = quantize(df_scaled)
        return df_quantized