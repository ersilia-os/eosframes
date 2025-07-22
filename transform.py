from .scalarizer import scalarize
from .quantizer import quantize

class Transform:
    def __init__(self, scaling_strategy='standard', quantization_bins=250):
        self.scaling_strategy = scaling_strategy
        self.quantization_bins = quantization_bins

    def run(self, df):
        df_scaled = scalarize(df, strategy=self.scaling_strategy)
        df_quantized = quantize(df_scaled, n_bins=self.quantization_bins)
        return df_quantized