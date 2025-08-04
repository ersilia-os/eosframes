from scalarizer import scalarize

class Scale:
    def __init__(self, robust_scalar, power_transformer):
        self.robust_scalar = robust_scalar
        self.power_transformer = power_transformer
        self.scaler = None  # will hold fitted transformer

    def train(self, df):
        self.scaler, df_scaled = scalarize(df, self.power_transformer, self.robust_scalar) # Using RobustScaler
        return self.scaler, df_scaled

    def inference(self, df):
         if self.scaler_ is None:
            raise ValueError("You must call fit() before inference()")
        return self.scaler.transform(df) #scikit-learn transformer has function .transform()

    