"""Synthetic contract tests; no AWS calls or training/evaluation execution."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from steps.preprocess import preprocess as prep
from steps.train.preprocessing_artifacts import bundle_preprocessing, inspect_preprocessing


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "CustomerID": ["a", "b", "c", "d"],
            "Tenure Months": [0, 2, 4, 6],
            "Monthly Charges": [10., 10., 10., 10.],
            "Total Charges": [" ", "20", "40", "60"],
            "Payment Method": ["Bank", "Card", "Cash", "Bank"],
            "Churn Label": ["No", "Yes", "No", "Yes"],
            "Churn Value": [0, 1, 0, 1],
        })
        self.state = prep.fit_preprocessing(self.raw)

    def test_roundtrip_and_packaged_code(self):
        expected = prep.transform_features(self.raw, self.state)
        with tempfile.TemporaryDirectory() as directory:
            prep.save_preprocessing(self.state, directory)
            spec = importlib.util.spec_from_file_location("packaged_preprocessing", Path(directory) / "preprocessing.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded = module.load_preprocessing(directory)
            assert_frame_equal(expected, module.transform_features(self.raw, loaded))
            self.assertEqual(loaded, self.state)
            self.assertEqual((Path(directory) / "preprocessing.py").read_bytes(), Path(prep.__file__).read_bytes())

    def test_single_record_batch_and_reordered_columns(self):
        expected = prep.transform_features(self.raw, self.state)
        for index in self.raw.index:
            assert_frame_equal(expected.loc[[index]], prep.transform_features(self.raw.loc[[index]], self.state))
        assert_frame_equal(expected, prep.transform_features(self.raw[self.raw.columns[::-1]], self.state))

    def test_missing_reference_does_not_change_encoding(self):
        subset = self.raw.loc[[1, 2]]
        result = prep.transform_features(subset, self.state)
        self.assertEqual(result["PaymentMethod_Card"].tolist(), [1., 0.])
        self.assertEqual(result["PaymentMethod_Cash"].tolist(), [0., 1.])

    def test_heldout_values_do_not_mutate_state(self):
        original = copy.deepcopy(self.state)
        heldout = self.raw.loc[[1]].copy()
        heldout["Total Charges"] = "10000"
        prep.transform_features(heldout, self.state)
        heldout["Payment Method"] = "New"
        with self.assertRaisesRegex(ValueError, "unseen.*PaymentMethod"):
            prep.transform_features(heldout, self.state)
        self.assertEqual(self.state, original)
        self.assertEqual(self.state["imputation"]["TotalCharges"], 40.)
        self.assertEqual(self.state["scaler"]["tenure"]["mean"], 3.)

    def test_imputation_scaling_and_aliases(self):
        raw = self.raw.copy()
        raw["Total Charges"] = ["bad", "", "40", "60"]
        state = prep.fit_preprocessing(raw)
        canonical = prep.normalize_features(raw)
        result = prep.transform_features(canonical, state)
        np.testing.assert_allclose(result["MonthlyCharges"], 0.)
        self.assertEqual(state["scaler"]["MonthlyCharges"]["scale"], 1.)
        stats = state["scaler"]["TotalCharges"]
        restored = result["TotalCharges"] * stats["scale"] + stats["mean"]
        np.testing.assert_allclose(restored, [0., 50., 40., 60.])
        raw["Total Charges"] = ""
        self.assertEqual(prep.fit_preprocessing(raw)["imputation"]["TotalCharges"], 0.)

    def test_unlabeled_and_extra_columns(self):
        raw = self.raw.drop(columns=["Churn Value", "Churn Label"]).assign(extra="ignored")
        assert_frame_equal(prep.transform_features(self.raw, self.state), prep.transform_features(raw, self.state))
        self.assertNotIn("Churn", self.state["required_features"])

    def test_invalid_records(self):
        cases = [
            (self.raw.drop(columns="Monthly Charges"), "Missing required"),
            (self.raw.assign(tenure=1), "Duplicate canonical"),
            (self.raw.assign(**{"Payment Method": "New"}), "unseen"),
            (self.raw.assign(**{"Payment Method": None}), "Missing or unseen"),
            (self.raw.assign(**{"Monthly Charges": "bad"}), "Invalid.*MonthlyCharges"),
            (self.raw.assign(**{"Total Charges": np.inf}), "nonfinite"),
        ]
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prep.transform_features(raw, self.state)

    def test_numeric_and_mixed_targets(self):
        for labels in ([0, 1, 0, 1], [0., 1., 0., 1.], ["No", 1, 0, "Yes"]):
            with self.subTest(labels=labels):
                result = prep.clean_and_preprocess(
                    self.raw.assign(**{"Churn Value": labels})
                )
                self.assertEqual(result["Churn"].tolist(), [0, 1, 0, 1])
                self.assertTrue(pd.api.types.is_integer_dtype(result["Churn"]))

    def test_target_validation(self):
        raw = self.raw.copy()
        raw["Churn Value"] = ["No", "Yes", "No", "Yes"]
        self.assertEqual(prep.clean_and_preprocess(raw)["Churn"].tolist(), [0, 1, 0, 1])
        for labels in [["No", "unknown", "No", "Yes"], [0, 1, None, 1], [0, 2, 0, 1]]:
            with self.assertRaises(ValueError):
                prep.clean_and_preprocess(self.raw.assign(**{"Churn Value": labels}))

    def test_invalid_fit_inputs(self):
        for raw in [self.raw.iloc[:0], self.raw.assign(**{"Payment Method": None}), self.raw.assign(**{"Monthly Charges": np.inf})]:
            with self.assertRaises(ValueError):
                prep.fit_preprocessing(raw)

    def test_malformed_state(self):
        states = [None, {}, {"schema_version": 99}]
        for key, value in [("scaler", {}), ("output_features", ["wrong"]), ("categories", {}), ("feature_types", {}), ("imputation", {})]:
            bad = copy.deepcopy(self.state)
            bad[key] = value
            states.append(bad)
        for state in states:
            with self.subTest(state=state), self.assertRaises(ValueError):
                prep.transform_features(self.raw, state)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "preprocessing.json").write_text("not JSON")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                prep.load_preprocessing(directory)

    def test_artifact_bundle_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "input", root / "model"
            prep.save_preprocessing(self.state, source)
            count = len(self.state["output_features"])
            inspected = inspect_preprocessing(source, count)
            bundled = bundle_preprocessing(source, target, count, inspected)
            self.assertEqual(inspected, bundled)
            for name, metadata in bundled["files"].items():
                self.assertEqual((source / name).read_bytes(), (target / name).read_bytes())
                self.assertEqual(metadata["sha256"], hashlib.sha256((target / name).read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "feature count"):
                inspect_preprocessing(source, count + 1)
            (source / "preprocessing.py").write_text("# changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                bundle_preprocessing(source, target, count, inspected)
            (source / "preprocessing.json").write_text(json.dumps({"schema_version": 99}))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                inspect_preprocessing(source, count)

    def test_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                inspect_preprocessing(directory, 1)


if __name__ == "__main__":
    unittest.main()
