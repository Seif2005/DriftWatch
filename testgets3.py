import boto3
import pandas as pd


def read_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Reads a CSV object from S3 into a pandas DataFrame."""
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response["Body"])

def main():
    bucket = "telco-churn-746760141372-eu-west-1-an"
    prefix = "input/"
    key = f"{prefix}telco_churn.csv"

    df = read_from_s3(bucket, key)
    print(df.head())

if __name__ == "__main__":
    main()
