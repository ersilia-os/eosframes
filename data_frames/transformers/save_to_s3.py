import boto3
import json
import joblib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def save_to_s3(
    dir_name,
    metadata,
    pipeline,
):
    # Read bucket name and AWS config from environment
    bucket_name = os.getenv("S3_BUCKET_NAME")
    region = os.getenv("AWS_DEFAULT_REGION")

    # Build S3 folder path: model_id/transform_type/
    s3_prefix = f"{dir_name}"

    # Save locally first
    metadata_path = "metadata.json"
    pipeline_path = "pipeline.joblib"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(pipeline, pipeline_path)

    # Upload to S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region,
    )

    s3.upload_file(metadata_path, bucket_name, os.path.join(s3_prefix, metadata_path))
    s3.upload_file(pipeline_path, bucket_name, os.path.join(s3_prefix, pipeline_path))

    print(
        f"✅ Saved {metadata_path} and {pipeline_path} to s3://{bucket_name}/{s3_prefix}"
    )
