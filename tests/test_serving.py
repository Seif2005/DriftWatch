"""Exercise serving with synthetic preprocessing artifacts and a fake booster."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMatrix:
    def __init__(self, data, feature_names=None):
        self.data = np.asarray(data)
        self.feature_names = feature_names


class FakeBooster:
    feature_names = ["1", "2"]

    def load_model(self, path):
        self.path = path

    def num_features(self):
        return 2

    def num_boosted_rounds(self):
        return 8

    def save_config(self):
        return json.dumps({"learner": {"objective": {"name": "binary:logistic"}}})

    def predict(self, matrix, iteration_range):
        self.iteration_range = iteration_range
        self.matrix = matrix
        return np.where(matrix.data[:, 0] < 0, 0.4, 0.55)


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_xgb = types.ModuleType("xgboost")
        fake_xgb.Booster = FakeBooster
        fake_xgb.DMatrix = FakeMatrix
        # No XGBoost installation, training, or AWS access is needed.
        with patch.dict(sys.modules, {"xgboost": fake_xgb, "boto3": None}):
            cls.inference = load_module("serving_under_test", ROOT / "steps/inference/inference.py")
        cls.preprocessing = load_module("preprocessing_under_test", ROOT / "steps/preprocess/preprocess.py")

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_dir = Path(temporary.name)
        self.rows = [
            {"Tenure Months": 0, "Contract": "A"},
            {"Tenure Months": 10, "Contract": "B"},
            {"Tenure Months": 20, "Contract": "A"},
        ]
        state = self.preprocessing.fit_preprocessing(pd.DataFrame(self.rows))
        self.preprocessing.save_preprocessing(state, self.model_dir)
        (self.model_dir / "xgboost-model").write_text("fake model")
        self.metadata = {
            "feature_count": 2, "best_iteration": 2, "selected_threshold": 0.55,
            "preprocessing": {
                "schema_version": 1,
                "files": {
                    name: {"sha256": hashlib.sha256((self.model_dir / name).read_bytes()).hexdigest()}
                    for name in ("preprocessing.json", "preprocessing.py")
                },
            },
        }
        self.write_metadata()
        self.no_download = patch.object(
            self.inference, "download_artifacts", side_effect=AssertionError("Serving must not download")
        )
        self.no_download.start()
        self.addCleanup(self.no_download.stop)

    def write_metadata(self):
        (self.model_dir / "training_metadata.json").write_text(json.dumps(self.metadata))

    def serve(self, body, content_type):
        bundle = self.inference.model_fn(str(self.model_dir))
        customers = self.inference.input_fn(body, content_type)
        result = self.inference.predict_fn(customers, bundle)
        response = self.inference.output_fn(result, "application/json")
        return json.loads(response), bundle

    def test_single_json_uses_bundled_model(self):
        result, bundle = self.serve(json.dumps(self.rows[0]), "application/json")
        self.assertEqual(result, {"predictions": [{"churn_probability": 0.4, "churn_prediction": 0}]})
        self.assertEqual(bundle["model"].path, str(self.model_dir / "xgboost-model"))

    def test_json_batch_csv_and_shared_predictions_agree(self):
        result, bundle = self.serve(json.dumps(self.rows).encode(), "application/json; charset=utf-8")
        csv_result, _ = self.serve(pd.DataFrame(self.rows).to_csv(index=False).encode(), "text/csv")
        self.assertEqual(csv_result, result)
        self.assertEqual(result, self.inference.predict_churn(self.rows, bundle))
        self.assertEqual([p["churn_prediction"] for p in result["predictions"]], [0, 1, 1])
        self.assertEqual(bundle["model"].iteration_range, (0, 3))
        self.assertEqual(bundle["model"].matrix.feature_names, ["1", "2"])
        self.assertEqual(bundle["model"].matrix.data[1, 0], 0.0)

    def test_unsupported_media_types(self):
        for media_type in ("text/plain", "application/xml"):
            with self.subTest(media_type=media_type), self.assertRaisesRegex(ValueError, "Unsupported input"):
                self.inference.input_fn("{}", media_type)
        with self.assertRaisesRegex(ValueError, "Unsupported output"):
            self.inference.output_fn({}, "text/csv")

    def test_invalid_json_and_customer_shapes(self):
        for payload in ("{", "null", '"customer"', "1"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.inference.input_fn(payload, "application/json")
        bundle = self.inference.model_fn(self.model_dir)
        for payload in ([], {}, [1], {"Contract": ["A"]}):
            with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                self.inference.predict_fn(payload, bundle)

    def test_missing_columns_unseen_categories_and_headerless_csv(self):
        for payload, media_type in (
            ('{"Contract": "A"}', "application/json"),
            ('{"Tenure Months": 0, "Contract": "unknown"}', "application/json"),
            ("0,A\n10,B\n", "text/csv"),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.serve(payload, media_type)

    def test_output_serialization_rejects_nonfinite_numbers(self):
        with self.assertRaises(ValueError):
            self.inference.output_fn({"predictions": [float("nan")]}, "application/json")

    def test_missing_and_tampered_artifacts_fail_before_model_load(self):
        path = self.model_dir / "preprocessing.py"
        original = path.read_bytes()
        for content in (None, original + b"\n# changed"):
            if content is None:
                path.unlink()
            else:
                path.write_bytes(content)
            with self.subTest(content=content is None), patch.object(
                self.inference.xgb, "Booster", side_effect=AssertionError("Must validate first")
            ), self.assertRaisesRegex(ValueError, "Missing model artifact|integrity mismatch"):
                self.inference.model_fn(self.model_dir)

    def test_invalid_metadata(self):
        for key, value in (("best_iteration", 8), ("selected_threshold", 1.5), ("feature_count", 3)):
            original = self.metadata[key]
            self.metadata[key] = value
            self.write_metadata()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.inference.model_fn(self.model_dir)
            self.metadata[key] = original


if __name__ == "__main__":
    unittest.main()
