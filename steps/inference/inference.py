"""Load verified DriftWatch artifacts and predict raw single or batch customers.

Examples (run from the repository root)::

    python steps/inference/inference.py --input customers.json --output predictions.json
    python steps/inference/inference.py --model-dir artifacts/inference/model-ID --input customers.csv

JSON input is one customer object or a list of customer objects. CSV input must
have named raw feature columns. Imports perform no downloads or inference.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import tarfile
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_S3_URI = "s3://driftwatch-bucket/models/artifacts/bbpg1hb7zms4-Bayesian-zgA7xZdemi-005-88a4fc6d/output/model.tar.gz"
PREPROCESSING_JSON_S3_URI = "s3://driftwatch-bucket/data/processed/preprocessing/preprocessing.json"
PREPROCESSING_PY_S3_URI = "s3://driftwatch-bucket/data/processed/preprocessing/preprocessing.py"

ARTIFACT_NAMES = ("xgboost-model", "training_metadata.json", "preprocessing.json", "preprocessing.py")


def download_s3_file(client, uri, destination):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected s3://bucket/key, got {uri!r}")
    client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))


def unpack_model(archive_path, destination):
    """Copy recognized regular root files without extracting archive paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in ARTIFACT_NAMES:
                continue
            if name in seen or not member.isfile():
                raise ValueError(f"Duplicate or nonregular model artifact: {name}")
            seen.add(name)
            with archive.extractfile(member) as source, (destination / name).open("wb") as target:
                shutil.copyfileobj(source, target)
    for name in ARTIFACT_NAMES[:2]:
        if name not in seen:
            raise ValueError(f"Model archive is missing {name}")


def download_artifacts(model_uri, state_uri, code_uri, cache_dir, client=None):
    if client is None:
        import boto3

        client = boto3.client("s3")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep the completed download for reuse; clean partial downloads on failure.
    with TemporaryDirectory(prefix="download-", dir=cache_dir) as temporary:
        root = Path(temporary)
        archive = root / "model.tar.gz"
        model_dir = root / "model"
        download_s3_file(client, model_uri, archive)
        unpack_model(archive, model_dir)
        for name, uri in (("preprocessing.json", state_uri), ("preprocessing.py", code_uri)):
            separate = root / name
            download_s3_file(client, uri, separate)
            packaged = model_dir / name
            if packaged.exists() and packaged.read_bytes() != separate.read_bytes():
                raise ValueError(f"S3 {name} differs from the preprocessing bundled with the model")
            shutil.copyfile(separate, packaged)
        # Validate before retaining or importing preprocessing code.
        validate_artifacts(model_dir)
        import uuid
        retained = cache_dir / f"model-{uuid.uuid4().hex}"
        shutil.move(str(model_dir), str(retained))
    return retained

def validate_artifacts(model_dir):
    model_dir = Path(model_dir)
    for name in ARTIFACT_NAMES:
        if not (model_dir / name).is_file():
            raise ValueError(f"Missing model artifact: {name}")
    metadata = json.loads((model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    count = metadata.get("feature_count")
    best = metadata.get("best_iteration")
    threshold = metadata.get("selected_threshold")
    if type(count) is not int or count <= 0:
        raise ValueError("Invalid metadata feature_count")
    if type(best) is not int or best < 0:
        raise ValueError("Invalid metadata best_iteration")
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid metadata selected_threshold")
    provenance = metadata.get("preprocessing")
    if not isinstance(provenance, dict) or type(provenance.get("schema_version")) is not int or provenance["schema_version"] != 1:
        raise ValueError("Missing or unsupported preprocessing provenance; use a model trained with the current contract")
    files = provenance.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Malformed preprocessing hash metadata")
    for name in ("preprocessing.json", "preprocessing.py"):
        entry = files.get(name)
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        actual = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"Preprocessing integrity mismatch: {name}; obtain the original training artifacts")
    state = json.loads((model_dir / "preprocessing.json").read_text(encoding="utf-8"))
    if type(state.get("schema_version")) is not int or state["schema_version"] != provenance["schema_version"]:
        raise ValueError("Preprocessing schema versions differ")
    features = state.get("output_features")
    if (not isinstance(features, list) or not all(isinstance(name, str) for name in features)
            or len(features) != count or len(set(features)) != count):
        raise ValueError("Preprocessing output features do not match model metadata")
    return metadata


def load_inference_bundle(model_dir):
    """Load verified artifacts once; prediction performs no S3 calls."""
    model_dir = Path(model_dir)
    metadata = validate_artifacts(model_dir)
    spec = importlib.util.spec_from_file_location("driftwatch_model_preprocessing", model_dir / "preprocessing.py")
    preprocessing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing)
    state = preprocessing.load_preprocessing(model_dir)
    model = xgb.Booster()
    model.load_model(str(model_dir / "xgboost-model"))
    if model.num_features() != metadata["feature_count"]:
        raise ValueError("Booster feature count differs from metadata")
    if metadata["best_iteration"] >= model.num_boosted_rounds():
        raise ValueError("Best iteration is outside the saved model's tree range")
    configuration = json.loads(model.save_config())
    if configuration["learner"]["objective"]["name"] != "binary:logistic":
        raise ValueError("Expected a binary:logistic churn probability model")
    # Training reads headerless CSVs: feature columns are 1, 2, ..., N.
    # Some serialized formats discard names; named models must match a known schema.
    names = model.feature_names
    positional_names = [str(index) for index in range(1, metadata["feature_count"] + 1)]
    if names is not None and names != positional_names and names != state["output_features"]:
        raise ValueError("Booster feature names do not match the training or preprocessing schema")
    return {"model": model, "metadata": metadata, "state": state, "preprocessing": preprocessing}

def predict_churn(customers, bundle):
    """Predict one or many raw customers through exactly the same path."""
    if isinstance(customers, pd.DataFrame):
        frame = customers.copy()
    elif isinstance(customers, dict):
        if any(not pd.api.types.is_scalar(value) for value in customers.values()):
            raise ValueError("A customer dictionary must contain scalar feature values")
        frame = pd.DataFrame([customers])
    elif isinstance(customers, list):
        if not customers or not all(isinstance(record, dict) and record for record in customers):
            raise ValueError("A batch must be a nonempty list of customer dictionaries")
        if any(not pd.api.types.is_scalar(value) for record in customers for value in record.values()):
            raise ValueError("Customer features must contain scalar values")
        frame = pd.DataFrame(customers)
    else:
        raise TypeError("Expected a customer dictionary, a list of dictionaries, or a DataFrame")
    if frame.empty:
        raise ValueError("At least one customer with feature values is required")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input column names")
    if not all(isinstance(name, str) for name in frame.columns):
        raise ValueError("Input feature names must be strings")
    features = bundle["preprocessing"].transform_features(frame, bundle["state"])
    if features.shape[1] != bundle["metadata"]["feature_count"]:
        raise ValueError("Transformed feature count differs from model metadata")
    model = bundle["model"]
    # Preserve preprocessing order, while supplying the feature names saved by training.
    matrix = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=model.feature_names)
    probabilities = np.asarray(model.predict(
        matrix, iteration_range=(0, bundle["metadata"]["best_iteration"] + 1)
    ))
    if (probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Model returned invalid churn probabilities")
    threshold = float(bundle["metadata"]["selected_threshold"])
    return {"predictions": [
        {"churn_probability": float(probability), "churn_prediction": int(probability >= threshold)}
        for probability in probabilities
    ]}


def model_fn(model_dir):
    """Load SageMaker's local model and bundled preprocessing without S3 calls."""
    return load_inference_bundle(model_dir)


def input_fn(request_body, content_type):
    """Decode raw customer JSON or headered CSV for SageMaker serving."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in ("application/json", "text/csv"):
        raise ValueError(f"Unsupported input content type: {content_type}")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    if media_type == "application/json":
        customers = json.loads(request_body)
        if not isinstance(customers, (dict, list)):
            raise ValueError("JSON input must be a customer object or a list of customer objects")
        return customers
    return pd.read_csv(StringIO(request_body))


def predict_fn(customers, bundle):
    """Use the shared preprocessing, tree range, and classification threshold."""
    return predict_churn(customers, bundle)


def output_fn(prediction, accept):
    """Serialize the shared prediction response as JSON."""
    if accept.split(";", 1)[0].strip().lower() != "application/json":
        raise ValueError(f"Unsupported output content type: {accept}")
    return json.dumps(prediction, allow_nan=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model-dir", type=Path,
                              help="Local directory of verified model/preprocessing artifacts; skips S3.")
    model_source.add_argument("--model-s3-uri", default=MODEL_S3_URI,
                              help="Model archive S3 URI (defaults to the configured winning model).")
    parser.add_argument("--preprocessing-json-s3-uri", default=PREPROCESSING_JSON_S3_URI)
    parser.add_argument("--preprocessing-py-s3-uri", default=PREPROCESSING_PY_S3_URI)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--input", type=Path, required=True,
                        help="Raw customer JSON object/list or CSV with feature headers.")
    parser.add_argument("--input-format", choices=("json", "csv"),
                        help="Defaults to the input filename extension.")
    parser.add_argument("--output", type=Path,
                        help="JSON output file; omitted means standard output.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_format = args.input_format or args.input.suffix.lower().lstrip(".")
    if input_format == "json":
        customers = json.loads(args.input.read_text(encoding="utf-8"))
    elif input_format == "csv":
        customers = pd.read_csv(args.input)
    else:
        raise ValueError("Use a .json or .csv input file, or specify --input-format")
    # Parse input before making any AWS calls. Local model directories need no AWS access.
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = download_artifacts(
            args.model_s3_uri, args.preprocessing_json_s3_uri,
            args.preprocessing_py_s3_uri, args.cache_dir,
        )
    bundle = load_inference_bundle(model_dir)
    result = predict_churn(customers, bundle)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
