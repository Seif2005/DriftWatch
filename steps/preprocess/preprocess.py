"""Data preprocessing module for DriftWatch Telco Customer Churn.

Designed for standalone execution as well as execution inside AWS SageMaker Processing
jobs (using Scikit-Learn container).
"""

import argparse
import boto3
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns to drop: identifiers, constants (zero-variance), target leakage, post-event and high-cardinality geospatial
COLUMNS_TO_DROP = [
    # Identifiers (high cardinality, leads to overfitting)
    "customerID",
    "CustomerID",
    # Zero-variance / constant features
    "Count",
    "Country",
    "State",
    # Target leakage & post-event metrics
    "Churn Label",
    "Churn Reason",
    "Churn Score",
    "CLTV",
    # Geospatial & high-cardinality features
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

# Standard column mapping from IBM Telco naming convention to canonical names
COLUMN_MAPPING = {
    "Tenure Months": "tenure",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Value": "Churn",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Senior Citizen": "SeniorCitizen",
}

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def read_from_s3(s3_uri: str, region: str) -> pd.DataFrame:
    """Read a CSV object from an S3 URI into a pandas DataFrame."""
    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != "s3" or not parsed_uri.netloc or not parsed_uri.path.strip("/"):
        raise ValueError(f"Expected an S3 URI like s3://bucket/key, got: {s3_uri}")

    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip("/")
    s3_client = boto3.client("s3", region_name=region)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response["Body"])


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataframe: drops identifiers/leakage columns, standardizes column

    names, handles whitespace and missing TotalCharges, encodes Churn label to 0/1.
    """
    df_cleaned = df.copy()

    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)

    # Standardize column names
    rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in df_cleaned.columns}
    df_cleaned = df_cleaned.rename(columns=rename_dict)

    # Handle TotalCharges if present
    if "TotalCharges" in df_cleaned.columns:
        df_cleaned["TotalCharges"] = pd.to_numeric(df_cleaned["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with 0.0 (tenure == 0 rows) or median if all tenure > 0
        if df_cleaned["TotalCharges"].isna().any():
            median_val = df_cleaned["TotalCharges"].median()
            fill_val = 0.0 if (("tenure" in df_cleaned.columns and (df_cleaned.loc[df_cleaned["TotalCharges"].isna(), "tenure"] == 0).all())) else (median_val if not pd.isna(median_val) else 0.0)
            df_cleaned["TotalCharges"] = df_cleaned["TotalCharges"].fillna(fill_val)

    # Encode Churn to integer if string ('Yes'/'No') or non-numeric
    if "Churn" in df_cleaned.columns:
        if df_cleaned["Churn"].dtype == object or str(df_cleaned["Churn"].dtype).startswith("str") or str(df_cleaned["Churn"].dtype).startswith("string"):
            df_cleaned["Churn"] = df_cleaned["Churn"].map({"Yes": 1, "No": 0})
            if df_cleaned["Churn"].isna().any():
                df_cleaned["Churn"] = df_cleaned["Churn"].fillna(0)
            df_cleaned["Churn"] = df_cleaned["Churn"].astype(int)

    return df_cleaned


def encode_features(
    df: pd.DataFrame, feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """One-hot encode features and optionally align them to training columns."""
    categorical_cols = [
        col
        for col in df.columns
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
        )
    ]
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    if feature_columns is not None:
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded
def parse_args():
    """Parses command-line arguments for SageMaker Processing or standalone run."""
    parser = argparse.ArgumentParser(description="Preprocess Telco Churn Data for SageMaker")
    parser.add_argument(
        "--input_s3_uri",
        type=str,
        required=True,
        help="S3 URI for the raw input CSV",
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="AWS region containing the S3 bucket",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Root directory for processed output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random_state = 42
    output_paths = {
        "train": Path(args.output_dir) / "train",
        "validation": Path(args.output_dir) / "validate",
        "test": Path(args.output_dir) / "test",
        "baseline": Path(args.output_dir) / "baseline",
    }
    print("=" * 60)
    print("DriftWatch Preprocessing Job Starting")
    print("=" * 60)

    # 1. Load raw input data from S3
    print(f"Loading raw dataset from: {args.input_s3_uri}")
    df = read_from_s3(args.input_s3_uri, args.region)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # 2. Clean and drop unneeded / leakage features
    df_cleaned = clean_and_preprocess(df)
    print(f"After cleaning & dropping columns: {df_cleaned.shape[1]} columns remain")

    # Ensure target column is present
    if "Churn" not in df_cleaned.columns:
        raise KeyError("Target column 'Churn' not found after preprocessing")

    # 3. Stratified Train / Validation / Test split (70% / 15% / 15%)
    X = df_cleaned.drop(columns=["Churn"])
    y = df_cleaned["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # 4. Fit one-hot encoding on training data and align the other splits to it.
    X_train = encode_features(X_train)
    training_feature_columns = list(X_train.columns)
    X_val = encode_features(X_val, training_feature_columns)
    X_test = encode_features(X_test, training_feature_columns)
    print(f"After one-hot encoding: {X_train.shape[1]} columns")

    print(f"Train split:      {len(X_train)} samples ({y_train.mean():.2%} churn)")
    print(f"Validation split: {len(X_val)} samples ({y_val.mean():.2%} churn)")
    print(f"Test split:       {len(X_test)} samples ({y_test.mean():.2%} churn)")

    # 5. Continuous feature scaling with StandardScaler (fit strictly on X_train)
    active_num_cols = [col for col in NUMERICAL_COLS if col in X_train.columns]
    if active_num_cols:
        scaler = StandardScaler()
        X_train[active_num_cols] = scaler.fit_transform(X_train[active_num_cols])
        X_val[active_num_cols] = scaler.transform(X_val[active_num_cols])
        X_test[active_num_cols] = scaler.transform(X_test[active_num_cols])
        print(f"Standardized numerical columns: {active_num_cols}")

    # 6. Format datasets
    # SageMaker XGBoost format: target as first column, NO headers, NO index
    train_sagemaker = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    val_sagemaker = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)
    test_sagemaker = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

    # 7. Output directory creation and file export
    for out_path in output_paths.values():
        out_path.mkdir(parents=True, exist_ok=True)

    train_file = output_paths["train"] / "train.csv"
    val_file = output_paths["validation"] / "validation.csv"
    test_file = output_paths["test"] / "test.csv"
    baseline_file = output_paths["baseline"] / "baseline.csv"

    train_sagemaker.to_csv(train_file, index=False, header=False)
    val_sagemaker.to_csv(val_file, index=False, header=False)
    test_sagemaker.to_csv(test_file, index=False, header=False)

    # Model Monitor Baseline: unlabeled training features WITH headers, NO index
    X_train.to_csv(baseline_file, index=False, header=True)

    print("Successfully exported datasets:")
    print(f" - Train:      {train_file} {train_sagemaker.shape}")
    print(f" - Validation: {val_file} {val_sagemaker.shape}")
    print(f" - Test:       {test_file} {test_sagemaker.shape}")
    print(f" - Baseline:   {baseline_file} {X_train.shape}")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
