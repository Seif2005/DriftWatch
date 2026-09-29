from pathlib import Path
from io import StringIO

import boto3
import pandas as pd


def read_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Reads a CSV object from S3 into a pandas DataFrame."""
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response["Body"])


def write_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Writes a pandas DataFrame as a CSV object to S3."""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


def download_bucket(bucket: str, destination: Path) -> None:
    """Downloads every object in an S3 bucket, preserving its key path."""
    s3_client = boto3.client("s3")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for object_info in page.get("Contents", []):
            key = object_info["Key"]
            if key.endswith("/"):
                continue

            local_path = (destination / key).resolve()

            if destination not in local_path.parents:
                raise ValueError(f"Refusing to write outside destination: {key}")

            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, key, str(local_path))
            print(f"Downloaded s3://{bucket}/{key} -> {local_path}")

def main():
    bucket = "driftwatch-bucket"
    driftwatch_root = Path("/home/sagemaker-user/DriftWatch")
    bucket_directory = driftwatch_root / bucket
    download_bucket(bucket, bucket_directory)

if __name__ == "__main__":
    main()
