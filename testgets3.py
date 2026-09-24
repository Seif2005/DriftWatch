import boto3
from io import StringIO

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

def main():
    bucket = "telco-churn-746760141372-eu-west-1-an"
    prefix = "input/"
    key = f"{prefix}telco_churn.csv"
    output_key = "output/telco_churn.csv"

    df = read_from_s3(bucket, key)
    write_to_s3(df, bucket, output_key)
    print(df.head())

if __name__ == "__main__":
    main()
