"""Train the DriftWatch XGBoost churn model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score


try:
	from .preprocessing_artifacts import inspect_preprocessing, bundle_preprocessing
except ImportError:
	from preprocessing_artifacts import inspect_preprocessing, bundle_preprocessing


RANDOM_STATE = 42
SCALE_POS_WEIGHT = 2.8


def load_sagemaker_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
	"""Load a headerless SageMaker XGBoost CSV."""
	dataframe = pd.read_csv(path, header=None)
	labels = dataframe.iloc[:, 0].astype(int)
	features = dataframe.iloc[:, 1:].astype(float)
	return features, labels


def select_threshold(
	model: xgb.Booster,
	validation_matrix: xgb.DMatrix,
	labels: pd.Series,
) -> tuple[float, pd.Series]:
	"""Select the threshold with the best validation F1 score."""
	validation_probabilities = model.predict(
		validation_matrix,
		iteration_range=(0, model.best_iteration + 1),
	)

	threshold_candidates = np.arange(0.00, 1.00, 0.01)
	threshold_scores = []
	for threshold in threshold_candidates:
		validation_predictions = (validation_probabilities >= threshold).astype(int)
		threshold_scores.append(
			{
				"threshold": threshold,
				"f1": f1_score(labels, validation_predictions, zero_division=0),
				"precision": precision_score(
					labels, validation_predictions, zero_division=0
				),
				"recall": recall_score(labels, validation_predictions, zero_division=0),
			}
		)

	threshold_table = pd.DataFrame(threshold_scores)
	best_threshold_row = threshold_table.loc[threshold_table["f1"].idxmax()]
	return float(best_threshold_row["threshold"]), best_threshold_row


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--train-path",
		type=Path,
		default=Path("driftwatch-bucket/data/processed/train/train.csv"),
	)
	parser.add_argument(
		"--validation-path",
		type=Path,
		default=Path("driftwatch-bucket/data/processed/validate/validation.csv"),
	)
	parser.add_argument(
		"--model-dir",
		type=Path,
		default=Path("artifacts/local-training"),
	)
	parser.add_argument(
		"--preprocessing-dir", type=Path,
		default=Path("driftwatch-bucket/data/processed/preprocessing"),
	)
	parser.add_argument("--num-boost-round", type=int, default=300)
	parser.add_argument("--early-stopping-rounds", type=int, default=15)
	parser.add_argument("--max_depth", type=int, default=5)
	parser.add_argument("--eta", type=float, default=0.1)
	parser.add_argument("--min_child_weight", type=float, default=1.0)
	parser.add_argument("--_tuning_objective_metric", default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model_path = args.model_dir / "xgboost-model"
	metadata_path = args.model_dir / "training_metadata.json"

	print(f"Training data: {args.train_path}")
	print(f"Validation data: {args.validation_path}")
	print(f"XGBoost version: {xgb.__version__}")

	X_train, y_train = load_sagemaker_csv(args.train_path)
	X_validation, y_validation = load_sagemaker_csv(args.validation_path)

	if X_train.shape[1] != X_validation.shape[1]:
		raise ValueError("Training and validation feature counts differ")
	assert set(y_train.unique()).issubset({0, 1})
	assert set(y_validation.unique()).issubset({0, 1})
	assert not X_train.isna().any().any()
	assert not X_validation.isna().any().any()

	print(f"Train:      {X_train.shape}, churn rate={y_train.mean():.2%}")
	print(f"Validation: {X_validation.shape}, churn rate={y_validation.mean():.2%}")
	print(f"Feature count: {X_train.shape[1]}")

	preprocessing_metadata = inspect_preprocessing(args.preprocessing_dir, X_train.shape[1])

	train_matrix = xgb.DMatrix(X_train, label=y_train)
	validation_matrix = xgb.DMatrix(X_validation, label=y_validation)
	evaluation_results = {}
	training_parameters = {
		"objective": "binary:logistic",
		"eval_metric": "aucpr",
		"max_depth": args.max_depth,
		"eta": args.eta,
		"gamma": 0.0,
		"min_child_weight": args.min_child_weight,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"scale_pos_weight": SCALE_POS_WEIGHT,
		"tree_method": "hist",
		"seed": RANDOM_STATE,
	}

	model = xgb.train(
		params=training_parameters,
		dtrain=train_matrix,
		num_boost_round=args.num_boost_round,
		evals=[(train_matrix, "train"), (validation_matrix, "validation")],
		evals_result=evaluation_results,
		early_stopping_rounds=args.early_stopping_rounds,
		verbose_eval=2,
	)

	selected_threshold, best_threshold_row = select_threshold(
		model, validation_matrix, y_validation
	)
	print(f"Best iteration: {model.best_iteration}")
	print(f"Best validation XGBoost PR-AUC (aucpr): {model.best_score:.4f}")
	print(f"Selected threshold: {selected_threshold:.2f}")
	print(f"Validation F1: {best_threshold_row['f1']:.4f}")
	print(f"Validation precision: {best_threshold_row['precision']:.4f}")
	print(f"Validation recall: {best_threshold_row['recall']:.4f}")

	args.model_dir.mkdir(parents=True, exist_ok=True)
	model.save_model(model_path)
	preprocessing_metadata = bundle_preprocessing(
		args.preprocessing_dir, args.model_dir, X_train.shape[1], preprocessing_metadata
	)
	metadata = {
		"preprocessing": preprocessing_metadata,
		"model_format": "xgboost-json",
		"feature_count": int(X_train.shape[1]),
		"best_iteration": int(model.best_iteration),
		"selected_threshold": selected_threshold,
		"training_parameters": training_parameters,
		"validation_metrics": {
			"aucpr": float(model.best_score),
			"f1": float(best_threshold_row["f1"]),
			"precision": float(best_threshold_row["precision"]),
			"recall": float(best_threshold_row["recall"]),
		},
	}
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	print(f"Saved model: {model_path}")
	print(f"Saved metadata: {metadata_path}")
	print(f"HPO_METRIC best_aucpr={float(model.best_score):.10f}", flush=True)


if __name__ == "__main__":
	main()
