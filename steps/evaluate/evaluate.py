"""Evaluate the finalized DriftWatch XGBoost churn model."""

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
	average_precision_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


def load_sagemaker_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
	"""Load a headerless SageMaker XGBoost CSV."""
	dataframe = pd.read_csv(path, header=None)
	labels = dataframe.iloc[:, 0].astype(int)
	features = dataframe.iloc[:, 1:].astype(float)
	return features, labels


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--test-path",
		type=Path,
		default=Path("driftwatch-bucket/data/processed/test/test.csv"),
	)
	parser.add_argument(
		"--model-dir",
		type=Path,
		default=Path("artifacts/local-training"),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model_path = args.model_dir / "xgboost-model"
	metadata_path = args.model_dir / "training_metadata.json"
	model_card_path = args.model_dir / "model_card.md"

	print(f"Test data: {args.test_path}")
	print(f"Model: {model_path}")

	X_test, y_test = load_sagemaker_csv(args.test_path)
	metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

	assert set(y_test.unique()).issubset({0, 1})
	assert not X_test.isna().any().any()
	assert X_test.shape[1] == metadata["feature_count"]

	model = xgb.Booster()
	model.load_model(model_path)
	test_matrix = xgb.DMatrix(X_test)
	test_probabilities = model.predict(
		test_matrix,
		iteration_range=(0, metadata["best_iteration"] + 1),
	)
	selected_threshold = float(metadata["selected_threshold"])
	test_predictions = (test_probabilities >= selected_threshold).astype(int)

	test_metrics = {
		"pr_auc": average_precision_score(y_test, test_probabilities),
		"f1": f1_score(y_test, test_predictions, zero_division=0),
		"precision": precision_score(y_test, test_predictions, zero_division=0),
		"recall": recall_score(y_test, test_predictions, zero_division=0),
	}

	print(f"Test: {X_test.shape}, churn rate={y_test.mean():.2%}")
	print(f"Feature count: {X_test.shape[1]}")
	print(f"Selected threshold: {selected_threshold:.2f}")
	print(f"Best iteration: {metadata['best_iteration']}")
	print("Final test metrics:")
	for metric_name, metric_value in test_metrics.items():
		print(f"  {metric_name}: {metric_value:.4f}")
	print("\nConfusion matrix [rows=true, columns=predicted]:")
	print(confusion_matrix(y_test, test_predictions))
	print("\nClassification report:")
	print(classification_report(y_test, test_predictions, digits=4, zero_division=0))

	confusion = confusion_matrix(y_test, test_predictions)
	model_card = f"""# DriftWatch Churn Model Card

## Model details

- **Model:** XGBoost binary classifier
- **Task:** Predict whether a telecom customer will churn
- **Model artifact:** `{model_path}`
- **Best boosting iteration:** {metadata["best_iteration"]}
- **Feature count:** {metadata["feature_count"]}
- **Classification threshold:** {selected_threshold:.2f}

## Intended use

This model is intended to prioritize customers who may benefit from proactive retention outreach. Predictions should support review and prioritization, not make fully automated eligibility, pricing, or service decisions.

## Data and evaluation

- **Evaluation split:** Held-out test set
- **Test rows:** {len(y_test)}
- **Test churn rate:** {y_test.mean():.2%}
- **Training data:** Preprocessed Telco Customer Churn data stored in the project data pipeline
- **Feature schema:** {metadata["feature_count"]} numeric features produced by preprocessing

## Final test metrics

| Metric | Value |
|---|---:|
| PR-AUC | {test_metrics["pr_auc"]:.4f} |
| F1 | {test_metrics["f1"]:.4f} |
| Precision | {test_metrics["precision"]:.4f} |
| Recall | {test_metrics["recall"]:.4f} |

### Confusion matrix

Rows are true labels and columns are predicted labels:

```text
{confusion}
```

## Limitations and risks

- Performance may change when customer behavior, products, pricing, or data collection practices change.
- The selected threshold was optimized on validation data and may not match every operational cost tradeoff.
- Test metrics describe this dataset and should not be treated as a guarantee for future customers.
- Predictions should be monitored for data drift and subgroup performance before production decisions.

## Reproducibility

The model uses the training artifact's recorded best iteration and classification threshold. The test set is used only for final evaluation and is not used for training, early stopping, hyperparameter selection, or threshold selection.
"""
	model_card_path.write_text(model_card, encoding="utf-8")
	print(f"Saved model card: {model_card_path}")


if __name__ == "__main__":
	main()
