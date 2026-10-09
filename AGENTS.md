# DriftWatch agent guide

## Project and entry points

DriftWatch builds an XGBoost classifier for IBM Telco Customer Churn using AWS SageMaker. Work from the repository root; script defaults and pipeline source paths are relative to it.

- `Pipeline.ipynb`: SageMaker orchestration: preprocessing → Bayesian tuning → evaluation of the winning artifact. Uses SDK v3 imports (`sagemaker.core`, `sagemaker.train`, `sagemaker.mlops`); preserve these interfaces when changing pipeline code.
- Tuning runs the XGBoost script in a scikit-learn `1.2-1` framework training container so SageMaker accepts the custom `validation:best_aucpr` objective. `steps/train/requirements.txt` installs XGBoost 1.7.6 and requires package-index access. Evaluation uses a separate XGBoost `1.7-1` container; processing jobs do not install the training requirements.
- `steps/{preprocess,train,evaluate}/*.py`: standalone CLI scripts used by SageMaker jobs. Each uses `argparse` and a guarded `main()`; keep job paths configurable through arguments.
- Matching `.ipynb` files contain exploration/local previews. Keep training/evaluation logic consistent with their scripts when changing shared behavior. The preprocessing notebook is exploratory and differs from the production script in encoding order and output layout; use `preprocess.py` for the job contract.
- `steps/inference/inference.py` is empty; no serving implementation exists yet.
- `getbucket.py` downloads the entire configured S3 bucket to `driftwatch-bucket/`; it is an AWS operation, not a setup check.
- `HPO.md` describes the source-only HPO implementation and static review requirements. Check current code before copying plan values: the notebook currently sets `max_jobs=2`, while the plan proposes 10.

## Data and model contracts

- Preprocessing drops identifiers/leakage columns, normalizes IBM column names, and makes stratified 70/15/15 splits with seed 42. Align validation/test dummy columns to training columns; fit numeric scaling only on training data.
- Labeled CSVs have binary `Churn` in column zero, numeric features, no header, and no index. Outputs are `train/train.csv`, `validate/validation.csv`, and `test/test.csv`. The `validation` channel/output name intentionally maps to the `validate` directory.
- `baseline/baseline.csv` contains training features only, with headers. Preprocessing currently does not persist a scaler or encoder for serving.
- Local training/evaluation default to `driftwatch-bucket/data/processed/` and `artifacts/local-training/`. Preprocessing instead defaults to `data/processed/`; supply `--output_dir` when connecting local workflows.
- Training saves `xgboost-model` and `training_metadata.json` together. Evaluation accepts that directory or a model tar archive, uses metadata's `feature_count`, `best_iteration`, and `selected_threshold`, and writes `evaluation.json` plus `model_card.md`.
- Keep the test set exclusively for final evaluation. Early stopping, hyperparameter selection, and F1 threshold selection use validation data. Predictions use trees through `best_iteration + 1`.
- Use XGBoost's `aucpr` consistently; scikit-learn average precision is a different metric. Preserve `validation_metrics.aucpr`, `classification_metrics.aucpr.value`, and the `HPO_METRIC best_aucpr=...` log format consumed by the tuner's regex.

## Environment and checks

There is no repository-wide dependency manifest, installation recipe, CI workflow, test suite, or configured lint/type-check command. `steps/train/requirements.txt` adds XGBoost to the framework training container; it is not a complete local environment manifest. Imports require pandas, NumPy, scikit-learn, XGBoost, boto3, and SageMaker for orchestration; notebook previews also use IPython. Do not infer package pins from the local `.venv/`.

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

- For the source-only HPO work described in `HPO.md`, do not execute notebooks, training/evaluation scripts, or SageMaker jobs. Treat runtime execution as a separate requested task. Notebook initialization contacts AWS; pipeline construction/definition can package or upload code, and `upsert()`/`start()` change cloud resources.
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

