"""Validate and bundle preprocessing artifacts without running model training."""

import hashlib
import json
from pathlib import Path
import shutil


ARTIFACT_FILES = ("preprocessing.json", "preprocessing.py")


def inspect_preprocessing(directory, feature_count):
    directory = Path(directory)
    for name in ARTIFACT_FILES:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing preprocessing artifact: {directory / name}")
    try:
        state = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("Unsupported preprocessing schema version")
        features = state["output_features"]
        if (not isinstance(features, list) or not all(isinstance(x, str) for x in features)
                or len(set(features)) != len(features) or len(features) != feature_count):
            raise ValueError("Preprocessing feature count/schema does not match training CSVs")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed preprocessing artifact") from exc
    return {
        "schema_version": state["schema_version"],
        "files": {
            name: {"sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
            for name in ARTIFACT_FILES
        },
    }


def bundle_preprocessing(directory, model_dir, feature_count, expected=None):
    """Copy exact inspected bytes into the SageMaker model archive directory."""
    metadata = inspect_preprocessing(directory, feature_count)
    if expected is not None and metadata != expected:
        raise ValueError("Preprocessing artifacts changed during training")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        source, target = Path(directory) / name, model_dir / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    if inspect_preprocessing(model_dir, feature_count) != metadata:
        raise ValueError("Copied preprocessing artifacts failed integrity validation")
    return metadata
