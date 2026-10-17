"""Data preprocessing module for DriftWatch Telco Customer Churn.

Designed for standalone execution as well as execution inside AWS SageMaker Processing
jobs (using Scikit-Learn container).
"""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
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
    import boto3

    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != "s3" or not parsed_uri.netloc or not parsed_uri.path.strip("/"):
        raise ValueError(f"Expected an S3 URI like s3://bucket/key, got: {s3_uri}")

    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip("/")
    s3_client = boto3.client("s3", region_name=region)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(response["Body"])


SCHEMA_VERSION = 1


def normalize_features(df, rules=None):
    """Apply deterministic naming rules without learning from records."""
    rules = rules or {"drop_columns": COLUMNS_TO_DROP, "column_mapping": COLUMN_MAPPING}
    result = df.drop(columns=rules["drop_columns"], errors="ignore").rename(
        columns=rules["column_mapping"]
    ).copy()
    if result.columns.duplicated().any():
        raise ValueError("Duplicate canonical column names")
    return result


def clean_and_preprocess(df):
    """Normalize and validate labels before splitting; do not impute features."""
    result = normalize_features(df)
    if "Churn" not in result:
        raise ValueError("Target column 'Churn' is required")
    # Older container pandas cannot replace string keys in an integer Series.
    # Convert text labels individually and preserve already-numeric targets.
    labels = result["Churn"].map(
        lambda value: {"Yes": 1, "No": 0}.get(value, value)
        if isinstance(value, str) else value
    )
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Churn must contain only Yes/No or 0/1")
    result["Churn"] = labels.astype(int)
    return result


def _numeric_features(frame, state):
    result = frame.copy()
    for name, kind in state["feature_types"].items():
        if kind != "numeric":
            continue
        if name == "TotalCharges":
            values = pd.to_numeric(result[name], errors="coerce")
            zero_tenure = (pd.to_numeric(result["tenure"], errors="coerce") == 0
                           if "tenure" in result else pd.Series(False, index=result.index))
            values = values.fillna(pd.Series(
                np.where(zero_tenure, 0.0, state["imputation"]["TotalCharges"]),
                index=result.index,
            ))
        else:
            values = pd.to_numeric(result[name], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid or nonfinite numeric values in {name}")
        result[name] = values.astype(float)
    return result


def fit_preprocessing(df):
    """Learn all transformation state exclusively from training features."""
    frame = normalize_features(df).drop(columns=["Churn"], errors="ignore")
    if frame.empty or not all(isinstance(name, str) for name in frame.columns):
        raise ValueError("Training features must have rows and named columns")
    state = {
        "schema_version": SCHEMA_VERSION,
        "normalization": {"drop_columns": list(COLUMNS_TO_DROP), "column_mapping": dict(COLUMN_MAPPING)},
        "required_features": list(frame.columns),
        "feature_types": {}, "categories": {}, "reference_levels": {},
        "imputation": {}, "scaler": {}, "output_features": [],
    }
    for name in frame:
        categorical = (name not in NUMERICAL_COLS and not pd.api.types.is_numeric_dtype(frame[name]))
        state["feature_types"][name] = "categorical" if categorical else "numeric"
        if categorical:
            if frame[name].isna().any() or not frame[name].map(lambda x: isinstance(x, str)).all():
                raise ValueError(f"Missing or non-string categorical values in {name}")
            levels = sorted(frame[name].unique().tolist())
            state["categories"][name] = levels
            state["reference_levels"][name] = levels[0]
    if "TotalCharges" in frame:
        values = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        median = values.median()
        state["imputation"]["TotalCharges"] = float(median) if pd.notna(median) else 0.0
    numeric = _numeric_features(frame, state)
    active = [name for name in NUMERICAL_COLS if name in frame]
    if active:
        scaler = StandardScaler().fit(numeric[active])
        state["scaler"] = {
            name: {"mean": float(mean), "scale": float(scale)}
            for name, mean, scale in zip(active, scaler.mean_, scaler.scale_)
        }
    state["output_features"] = [name for name in frame if state["feature_types"][name] == "numeric"]
    for name, levels in state["categories"].items():
        state["output_features"].extend(f"{name}_{level}" for level in levels[1:])
    validate_state(state)
    return state


def validate_state(state):
    """Reject unsupported or internally inconsistent persisted state."""
    try:
        if type(state["schema_version"]) is not int or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported preprocessing schema version")
        required = state["required_features"]
        types = state["feature_types"]
        if (not isinstance(required, list) or not isinstance(types, dict) or not required
                or "Churn" in required or len(set(required)) != len(required) or set(types) != set(required)):
            raise ValueError("Invalid required feature schema")
        if not all(isinstance(name, str) for name in required):
            raise ValueError("Feature names must be strings")
        rules = state["normalization"]
        if (not isinstance(rules["drop_columns"], list) or not isinstance(rules["column_mapping"], dict)
                or not all(isinstance(x, str) for x in rules["drop_columns"])
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in rules["column_mapping"].items())):
            raise ValueError("Invalid normalization rules")
        categorical = [name for name in required if types[name] == "categorical"]
        if any(kind not in ("numeric", "categorical") for kind in types.values()):
            raise ValueError("Invalid feature types")
        if set(state["categories"]) != set(categorical) or set(state["reference_levels"]) != set(categorical):
            raise ValueError("Invalid categorical schema")
        expected = [name for name in required if types[name] == "numeric"]
        for name in categorical:
            levels = state["categories"][name]
            if (not isinstance(levels, list) or not levels or not all(isinstance(x, str) for x in levels)
                    or levels != sorted(set(levels)) or state["reference_levels"][name] != levels[0]):
                raise ValueError(f"Invalid categories for {name}")
            expected.extend(f"{name}_{level}" for level in levels[1:])
        if expected != state["output_features"] or len(set(expected)) != len(expected) or not expected:
            raise ValueError("Invalid or colliding output feature names")
        if "TotalCharges" in required:
            if not np.isfinite(state["imputation"]["TotalCharges"]):
                raise ValueError("Invalid imputation value")
        active = {name for name in NUMERICAL_COLS if name in required}
        if set(state["scaler"]) != active:
            raise ValueError("Invalid scaler feature schema")
        for stats in state["scaler"].values():
            if not np.isfinite([stats["mean"], stats["scale"]]).all() or stats["scale"] <= 0:
                raise ValueError("Invalid scaler statistics")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed preprocessing state") from exc
    return state


def transform_features(df, state):
    """Transform raw labeled or unlabeled records using immutable fitted state."""
    validate_state(state)
    frame = normalize_features(df, state["normalization"])
    missing = set(state["required_features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    frame = _numeric_features(frame[state["required_features"]], state)
    result = frame[[name for name in state["required_features"]
                    if state["feature_types"][name] == "numeric"]].copy()
    for name, levels in state["categories"].items():
        invalid = frame[name].isna() | ~frame[name].isin(levels)
        if invalid.any():
            raise ValueError(f"Missing or unseen categories in {name}: {frame.loc[invalid, name].tolist()}")
        for level in levels[1:]:
            result[f"{name}_{level}"] = (frame[name] == level).astype(float)
    for name, stats in state["scaler"].items():
        result[name] = (result[name] - stats["mean"]) / stats["scale"]
    result = result[state["output_features"]].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Transformation produced nonfinite features")
    return result


def save_preprocessing(state, directory):
    validate_state(state)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preprocessing.json").write_text(
        json.dumps(state, indent=2, allow_nan=False), encoding="utf-8"
    )
    source, target = Path(__file__).resolve(), directory / "preprocessing.py"
    if source != target.resolve():
        shutil.copyfile(source, target)


def load_preprocessing(directory):
    try:
        state = json.loads((Path(directory) / "preprocessing.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed preprocessing JSON") from exc
    return validate_state(state)


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
        "preprocessing": Path(args.output_dir) / "preprocessing",
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

    # 3. Stratified Train / Validation / Test split (70% / 15% / 15%)
    X = df_cleaned.drop(columns=["Churn"])
    y = df_cleaned["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # Fit exclusively on training rows, then reuse the exact state for every split.
    state = fit_preprocessing(X_train)
    X_train = transform_features(X_train, state)
    X_val = transform_features(X_val, state)
    X_test = transform_features(X_test, state)
    save_preprocessing(state, output_paths["preprocessing"])
    print(f"Transformed feature count: {X_train.shape[1]}")

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
