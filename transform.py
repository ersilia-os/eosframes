from scalarizer import scalarize
from quantizer import quantize

class Transform:
    def __init__(self, robust_scalar, power_transformer, quantization_bins=250):
        self.robust_scalar = robust_scalar
        self.power_transformer = power_transformer
        self.quantization_bins = quantization_bins

    def run(self, df):
        df_scaled = scalarize(df, self.power_transformer, self.robust_scalar) # Using RobustScaler
        df_quantized = quantize(df_scaled, n_bins=self.quantization_bins)
        return df_quantized