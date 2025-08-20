import boto3
import json
import joblib
import os


def save_to_s3(
    model_id,
    metadata,
    pipeline,
    bucket_name="your-bucket-name",
):
    # Build S3 folder path: model_id/transform_type/
    s3_prefix = f"{model_id}"

    # Save locally first
    metadata_path = "metadata.json"
    pipeline_path = "pipeline.joblib"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(pipeline, pipeline_path)

    # Upload to S3
    s3 = boto3.client("s3")

    s3.upload_file(metadata_path, bucket_name, os.path.join(s3_prefix, metadata_path))
    s3.upload_file(pipeline_path, bucket_name, os.path.join(s3_prefix, pipeline_path))

    print(
        f"✅ Saved {metadata_path} and {pipeline_path} to s3://{bucket_name}/{s3_prefix}"
    )
