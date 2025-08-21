import pandas as pd
import joblib
import json
import os
from datetime import datetime
from data_frames.transformers.scalerizer import scalerize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from data_frames.transformers.build_quantize_transformer import build_quantizer
from data_frames.transformers.build_typed_transformer import build_typed_transformer
# from data_frames.quantizer import bin


class Quantize:
    def __init__(
            self, 
            robust_scaler: bool = False, 
            power_transform: bool=False
    ):
        # Store the original parameters for saving/loading
        self.kbd = None
        self.robust_scaler = robust_scaler
        self.power_transform = power_transform
        self.pipeline_ = None
        # self.transformer = bin()
        self._is_fitted = False
        self.feature_cols: list[str] = []
        self.num_rows = 0

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if the DataFrame is empty
        if df.empty:
             raise ValueError("Input DataFrame is empty.")
        
        # Set the number of rows
        self.num_rows = len(df)
        
        # Capture the timestamp when fit was called
        self.fit_timestamp = datetime.now()

        # impute missing values 
        imputer = SimpleImputer(strategy="median")
        df = pd.DataFrame(imputer.fit_transform(df), index=df.index, columns=df.columns)
        
        # Ensure only numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns to transform.")
        
        # scale data
        scaler = build_typed_transformer(df)
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data)

        #new code
        self.pipeline_ = build_quantizer(scaled_df)
        X_bin = self.pipeline_.fit_transform(scaled_df)

        #old code
        ###
        # self.kbd, X_bin  = build_quantizer(scaled_df)
        # # Fit the KBinsDiscretizer
        # n_bins = 256
        # self.kbd = KBinsDiscretizer(
        #     n_bins=n_bins, 
        #     encode='ordinal', 
        #     strategy='quantile', 
        #     quantile_method='averaged_inverted_cdf'
        # )
        
        # # Fit and transform the data
        # X_bin = self.kbd.fit_transform(scaled_df)
        
        # # Convert to integer and shift to center around 0
        # # This ensures we get values from -127 to 128 (256 bins total)
        # X_bin = X_bin.astype(int) - (n_bins // 2 - 1)
        ###
        
        self._is_fitted = True
        self.feature_cols = list(numeric_cols)

        return pd.DataFrame(
            X_bin,
            index=df.index,
            columns=self.feature_cols
        )
    
    def save(self, model_dir: str):
        """
        Save the fitted pipeline and related metadata to a directory.
        
        Args:
            model_dir (str): Directory path where the model files will be saved.
                            If the directory doesn't exist, it will be created.
        
        Raises:
            ValueError: If the model hasn't been fitted yet.
        """
        # Check if the model has been fitted before attempting to save
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise ValueError("Model not fitted. Call `fit` first.")

        # Create the model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the fitted pipeline to a joblib file
        # This serializes the entire pipeline object including all fitted transformers
        pipeline_path = os.path.join(model_dir, "pipeline.joblib") #creates a file path 
        joblib.dump(self.kbd, pipeline_path) #converts pipeline object into binary data to that file with median, etc

        # Create metadata dictionary containing all the important attributes
        # This includes the configuration parameters and fitted state information
        metadata = {
            "robust_scaler": self.robust_scaler,  # Use the stored robust_scaler
            "power_transform": self.power_transform,  # Use the stored power_transform
            "feature_cols": self.feature_cols,  # Save the feature column names that were used during fitting
            "fit_date": self.fit_timestamp.strftime("%Y-%m-%d") if hasattr(self, 'fit_timestamp') else None,
            "fit_time": self.fit_timestamp.strftime("%H:%M:%S") if hasattr(self, 'fit_timestamp') else None,
            "fit_timestamp": self.fit_timestamp.isoformat() if hasattr(self, 'fit_timestamp') else None,
            "num_rows": self.num_rows
        }
        
        # Save the metadata as a JSON file for easy reading and debugging
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)  # Use indent=2 for pretty formatting

    @classmethod
    def load(cls, model_dir: str):
        """
        Load a previously saved model from a directory.
        
        Args:
            model_dir (str): Directory path where the model files are stored.
        
        Returns:
            Scale: A new Scale instance with the loaded pipeline and metadata.
        
        Raises:
            FileNotFoundError: If the model directory or required files don't exist.
        """
        # Check if the model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        
        # Check if the pipeline file exists and load it
        pipeline_path = os.path.join(model_dir, "pipeline.joblib")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file {pipeline_path} not found.")
        pipeline = joblib.load(pipeline_path)  # Load the fitted pipeline object
        
        # Check if the metadata file exists and load it
        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        
        # Load the metadata from JSON file
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Create a new instance of the Scale class with the original parameters
        # We need to extract the original parameters from metadata or use defaults
        obj = cls(
            robust_scaler=metadata.get("robust_scaler", False),  # Get robust_scaler from metadata or default to False
            power_transform=metadata.get("power_transform", False),  # Get power_transform from metadata or default to False
        )
        
        # Restore the fitted pipeline and other attributes
        obj.kbd = pipeline  # Restore the fitted transformer
        obj.feature_cols = metadata.get("feature_cols", [])  # Restore the feature column names
        obj._is_fitted = metadata.get("_is_fitted", False)  # Restore the fitted state
        
        # Restore timestamp if available
        if "fit_timestamp" in metadata and metadata["fit_timestamp"]:
            obj.fit_timestamp = datetime.fromisoformat(metadata["fit_timestamp"])

        return obj

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with the already-fitted pipeline
        """
        # if not self._is_fitted:
        #     raise RuntimeError("You must call .fit() before .inference()")
        
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns to transform.")
       
        # Apply the same preprocessing that was used during fitting
        # scale data
        scaler = build_typed_transformer(df)
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data)

        #new code
        X_new = self.pipeline_.transform(scaled_df)
        
        return pd.DataFrame(
            X_new,
            index=df.index,
            columns=self.feature_cols
        )

