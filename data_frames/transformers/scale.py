import pandas as pd
import joblib
import json
import os
import tempfile
import boto3
from datetime import datetime
from data_frames.transformers.build_typed_transformer import build_typed_transformer
from data_frames.transformers.save_to_s3 import save_to_s3
from sklearn.impute import SimpleImputer


class Scale():
    def __init__(
        self, model_id: str):
        # Store the original parameters for saving/loading
        self.model_id = model_id
        self.pipeline_ = None
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

        # Ensure only numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columnds to transform.")
        
        #added line to scale fit 
        imputer = SimpleImputer(strategy="median")
        df = pd.DataFrame(imputer.fit_transform(df), index=df.index, columns=df.columns)

        self.feature_cols = list(numeric_cols)
        self.pipeline_ = build_typed_transformer(df)
        # self.pipeline_ = result[0]
        transformed = self.pipeline_.fit_transform(df)

        return pd.DataFrame(transformed)

    # Zimin save/load changes

    def save(self, dir_name=None, local=False):
        """
        Save the fitted pipeline and related metadata to a directory.

        Args:
            model_dir (str): Directory path where the model files will be saved.
                            If the directory doesn't exist, it will be created.

        Raises:
            ValueError: If the model hasn't been fitted yet.
        """
        save_dir = dir_name or self.model_id

        # Create metadata dictionary containing all the important attributes
        # This includes the configuration parameters and fitted state information
        metadata = {
            "feature_cols": self.feature_cols,  # Save the feature column names that were used during fitting
            "fit_date": self.fit_timestamp.strftime("%Y-%m-%d")
            if hasattr(self, "fit_timestamp")
            else None,
            "fit_time": self.fit_timestamp.strftime("%H:%M:%S")
            if hasattr(self, "fit_timestamp")
            else None,
            "fit_timestamp": self.fit_timestamp.isoformat()
            if hasattr(self, "fit_timestamp")
            else None,
            "num_rows": self.num_rows,
        }
        if local:
            # Create the model directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Save the fitted pipeline to a joblib file
            # This serializes the entire pipeline object including all fitted transformers
            pipeline_path = os.path.join(save_dir, "pipeline.joblib")  
            joblib.dump(self.pipeline_, pipeline_path)

            # Save the metadata as a JSON file for easy reading and debugging
            meta_path = os.path.join(save_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        save_to_s3(
            dir_name=save_dir,
            metadata=metadata,
            pipeline=self.pipeline_,
            bucket_name="ersilia-dataframes",
        )

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        bucket_name: str | None = None,
        model_dir: str | None = None,
    ):
        """
        Load a previously saved transformer for a given model and transformer type.

        Args:
            model_id: The model identifier used in S3 path prefix.
            transformer_type: One of {"robust_scaler", "power_transform", "none"}.
            bucket_name: If provided, files are downloaded from
                         s3://<bucket>/<model_id>/<transformer_type>/
            model_dir: If provided (and bucket_name is None), load from this local dir.

        Returns:
            Scale: instance with pipeline and metadata restored.
        """
        # Resolve source of metadata/pipeline files
        if bucket_name:
            # Download from S3 into a temp directory
            s3 = boto3.client("s3")
            tmpdir = tempfile.mkdtemp(prefix=f"{model_id}")
            pipeline_path = os.path.join(tmpdir, "pipeline.joblib")
            meta_path = os.path.join(tmpdir, "metadata.json")

            prefix = f"{model_id}"
            s3.download_file(bucket_name, f"{prefix}/pipeline.joblib", pipeline_path)
            s3.download_file(bucket_name, f"{prefix}/metadata.json", meta_path)

        elif model_dir:
            pipeline_path = os.path.join(model_dir, "pipeline.joblib")
            meta_path = os.path.join(model_dir, "metadata.json")
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(f"Pipeline file {pipeline_path} not found.")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        else:
            raise ValueError(
                "Provide either bucket_name (for S3) or model_dir (for local)."
            )

        # Load files
        pipeline = joblib.load(pipeline_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Instantiate and restore
        obj = cls(
            model_id=model_id,
        )
        obj.pipeline_ = pipeline
        obj.feature_cols = metadata.get("feature_cols", [])
        obj.num_rows = metadata.get("num_rows", 0)

        ts = metadata.get("fit_timestamp")
        if ts:
            obj.fit_timestamp = datetime.fromisoformat(ts)

        return obj

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with the already-fitted pipeline
        """
        if not self.feature_cols:
            raise RuntimeError(
                "Trained feature_cols are empty; was the model saved correctly?"
            )

        # Check for missing trained columns
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Inference data is missing trained columns: {missing}. "
                f"Expected exactly these columns (in order): {self.feature_cols}"
            )

        # Build input with the exact schema used for training
        X = df.reindex(columns=self.feature_cols)
        X = X.apply(pd.to_numeric, errors="coerce")

        X_new = self.pipeline_.transform(X)
        return pd.DataFrame(X_new, index=df.index, columns=self.feature_cols)
