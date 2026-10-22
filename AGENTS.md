# DriftWatch agent guide

## Project and entry points

DriftWatch builds an XGBoost classifier for IBM Telco Customer Churn using AWS SageMaker. Work from the repository root; script defaults and pipeline source paths are relative to it.

- `Pipeline.ipynb`: SageMaker orchestration: preprocessing → Bayesian tuning → evaluation → registration of the winning artifact as `PendingManualApproval`, without a metric threshold gate or endpoint deployment. Uses SDK v3 imports (`sagemaker.core`, `sagemaker.train`, `sagemaker.mlops`); preserve these interfaces when changing pipeline code. `ModelBuilder` packages the standalone inference entry script separately from model data; `ModelStep` registers the winner and its evaluation metrics after evaluation succeeds. Evaluation outputs use execution-specific S3 prefixes. The submission cell provisions the default `DriftWatchChurn` package group; runtime `ModelPackageGroupName` overrides must reference an existing group in the pipeline region.
- Tuning runs the XGBoost script in a scikit-learn `1.2-1` framework training container so SageMaker accepts the custom `validation:best_aucpr` objective. `steps/train/requirements.txt` installs XGBoost 1.7.6 and requires package-index access. Evaluation uses a separate XGBoost `1.7-1` container; processing jobs do not install the training requirements.
- `steps/{preprocess,train,evaluate}/*.py`: standalone CLI scripts used by SageMaker jobs. Each uses `argparse` and a guarded `main()`; keep job paths configurable through arguments.
- Matching `.ipynb` files contain exploration/local previews. Keep training/evaluation logic consistent with their scripts when changing shared behavior. The preprocessing notebook is exploratory and differs from the production script in encoding order and output layout; use `preprocess.py` for the job contract.
- `steps/inference/inference.py` provides verified artifact loading and shared single-customer/batch `predict_churn()` inference; its notebook imports those functions. SageMaker `model_fn`, `input_fn`, `predict_fn`, and `output_fn` hooks accept raw customer JSON objects/lists or headered CSV and return JSON predictions using only bundled model/preprocessing artifacts. The CLI accepts raw JSON/CSV via `--input`, optional `--output`, and either `--model-dir` for offline use or configurable S3 artifact URIs. Without `--model-dir`, it downloads the configured S3 artifacts to `--cache-dir` (default `artifacts/inference/`). Imports do not contact AWS. Endpoint deployment is not implemented.
- `getbucket.py` downloads the entire configured S3 bucket to `driftwatch-bucket/`; it is an AWS operation, not a setup check.
- `HPO.md` describes the source-only HPO implementation and static review requirements. Check current code before copying plan values: the notebook currently sets `max_jobs=10`.

## Data and model contracts

- Preprocessing drops identifiers/leakage columns, normalizes IBM column names, and makes stratified 70/15/15 splits with seed 42. Fit imputation, category vocabularies/reference levels, feature order, and numeric scaling only on training data; transform all splits using that state. Unseen/missing categorical values are rejected.
- Labeled CSVs have binary `Churn` in column zero, numeric features, no header, and no index. Outputs are `train/train.csv`, `validate/validation.csv`, and `test/test.csv`. The `validation` channel/output name intentionally maps to the `validate` directory.
- `baseline/baseline.csv` contains training features only, with headers. Preprocessing also writes `preprocessing/preprocessing.json` (schema v1, normalization, categories, imputation, feature order, scaling) and `preprocessing/preprocessing.py` (the same importable implementation). `load_preprocessing(directory)` and `transform_features(raw_dataframe, state)` support unlabeled records without AWS access.
- Local training/evaluation default to `driftwatch-bucket/data/processed/` and `artifacts/local-training/`. Preprocessing instead defaults to `data/processed/`; supply `--output_dir` when connecting local workflows.
- Training requires `--preprocessing-dir` (local default `driftwatch-bucket/data/processed/preprocessing`, SageMaker channel `preprocessing`), validates schema version/feature count, and saves `preprocessing.json` plus `preprocessing.py` beside `xgboost-model` and `training_metadata.json`. Metadata records the preprocessing schema version and SHA-256 hashes. Regenerate datasets and retrain to adopt this contract; never attach newly fitted state to existing models. Old models remain evaluable. Evaluation accepts that directory or a model tar archive, uses metadata's `feature_count`, `best_iteration`, and `selected_threshold`, and writes `evaluation.json` plus `model_card.md`.
- Keep the test set exclusively for final evaluation. Early stopping, hyperparameter selection, and F1 threshold selection use validation data. Predictions use trees through `best_iteration + 1`.
- Use XGBoost's `aucpr` consistently; scikit-learn average precision is a different metric. Preserve `validation_metrics.aucpr`, `classification_metrics.aucpr.value`, and the `HPO_METRIC best_aucpr=...` log format consumed by the tuner's regex.

## Environment and checks

There is no repository-wide dependency manifest, installation recipe, CI workflow, or configured lint/type-check command. Isolated synthetic contract tests live in `tests/`; run `python -m unittest discover -s tests -v` with pandas, NumPy, and scikit-learn installed. These tests exercise transformations, artifact copying, and shared inference and CLI contracts with a fake booster, without AWS calls or model training/evaluation; no dependency pins are inferred from the available interpreter. `steps/train/requirements.txt` adds XGBoost to the framework training container; it is not a complete local environment manifest. Imports require pandas, NumPy, scikit-learn, XGBoost, boto3, and SageMaker for orchestration; notebook previews also use IPython. Do not infer package pins from the local `.venv/`.

For source changes, parse Python and notebook JSON/code without importing or executing project code. This repository-wide syntax check uses only the standard library:

```sh
python - <<'PY'
import ast
import json
from pathlib import Path

for path in [Path("getbucket.py"), *Path("steps").glob("*/*.py")]:
    ast.parse(path.read_text(), filename=str(path))
for path in [Path("Pipeline.ipynb"), *Path("steps").glob("*/*.ipynb")]:
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path}:cell{index}")
print("Python and notebook syntax OK")
PY
```

Review CLI names, SageMaker channels/paths, artifact metadata, and duplicated notebook logic for changes affecting those contracts. Run `git diff --check` before completion. Static checks do not establish runtime/container compatibility or model quality.

## Execution and generated files

- For source-only HPO and registry changes, do not execute notebooks, training/evaluation scripts, or SageMaker jobs. Treat runtime execution as a separate requested task. Notebook initialization contacts AWS; pipeline construction/definition can package or upload code, the submission cell can create a model package group, and `upsert()`/`start()` change cloud resources. A pipeline run now also registers a model package.
- Preserve notebook outputs and execution counts when editing source cells. Historical `pr_auc` results represent average precision; do not relabel them as new `aucpr` measurements.
- Keep `.env`, virtual environments, Jupyter checkpoints, caches, `data/processed/`, `artifacts/`, `models/`, and the downloaded `driftwatch-bucket/` contents out of changes; `.gitignore` excludes them. Do not hand-edit generated model/evaluation artifacts to match source changes.


## Maintaining AGENTS.md

- Keep this file synchronized with the project's actual
  architecture, tooling, and development workflows.
- After making significant changes, check whether AGENTS.md
  needs updating.
- Update AGENTS.md when:
  - The architecture or directory structure changes.
  - New dependencies or frameworks are introduced.
  - Development, testing, or deployment commands change.
  - Coding conventions or project constraints change.
- Do not update AGENTS.md for minor bug fixes, refactoring,
  or changes that do not affect its instructions.
- Never add unverified assumptions or temporary information.
- Keep instructions concise and avoid duplicating the README.
- Include AGENTS.md changes in the same diff as the relevant
  code changes.

