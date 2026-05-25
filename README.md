# Deep Learning Football Match Prediction

Pre-match football match outcome prediction from team form sequences using classical machine learning and deep learning.

## Project Structure

```text
data/
  raw/              # Original datasets
  raw/splits/       # Chronological raw train/val/test CSVs
  interim/          # Legacy/intermediate experiments
  processed/        # Generated model-ready datasets
  processed/splits/ # Generated leakage-free train/val/test feature CSVs
configs/            # YAML experiment configs
notebooks/          # Jupyter exploration and experiments
outputs/
  models/           # Trained model artifacts
  plots/            # Figures and diagnostics
  results/          # Metrics and prediction outputs
  reports/          # Human-readable reports committed for milestones
  logs/             # Training logs
src/
  models/           # Baseline ML, LSTM, and Transformer model definitions
  training/         # Training loops and orchestration helpers
  preprocessing/    # Raw data cleaning
  features/         # Leakage-free feature engineering and sequence creation
  evaluation/       # Metrics and reports
  utils/            # Shared utilities
tests/              # Smoke tests and unit tests
```

Current canonical pipeline modules:

- `src/features/build_team_features.py` builds the team-centric historical features.
- `src/utils/data_utils.py` loads processed splits and validates feature columns.
- `src/models/ml_baselines.py` trains final classical ML baselines.
- `scripts/build_features.py` is the CLI wrapper for feature generation.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests:

```powershell
python -m pytest
python scripts\smoke_test.py
```

## Preprocessing Pipeline

The valid pre-match ML pipeline is:

```text
raw CSVs -> team-centric feature engineering -> processed splits -> ML baselines
```

1. Place source season CSVs in `data/raw/`, or use existing split CSVs in `data/raw/splits/`.
2. Build leakage-free team-centric historical features:

```powershell
python scripts\build_features.py
```

This writes:

```text
data/processed/team_centric_features.csv
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
outputs/reports/preprocessing_report.txt
```

The feature builder converts each fixture into two team-perspective rows, one
for each team. It then computes shifted rolling form features over the previous
3, 5, and 10 matches, adds opponent-side rolling features for the same fixture,
and writes chronological train/validation/test splits.

## Leakage Prevention

Current-match statistics such as goals, half-time goals/results, shots, shots
on target, fouls, corners, yellow cards, and red cards are used only to build
shifted historical rolling features from previous matches. They are never direct
model inputs for the target match.

The final ML input matrix is loaded from `data/processed/splits/` and validated
against direct leakage columns before training. The excluded direct columns are:

```text
FTHG, FTAG, HTHG, HTAG, FTR, HTR, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR
```

The older `data/interim/interim_encode.py` path is kept as legacy/intermediate
work and should not be treated as the canonical pre-match baseline pipeline.

## Baseline Training

Train the valid ML baselines:

```powershell
python -m src.models.ml_baselines
```

This writes:

```text
outputs/reports/final_baseline_report.txt
outputs/results/final_baseline_predictions.csv
```

Current test-split results:

```text
dummy_most_frequent  accuracy=0.3808  macro_f1=0.1839
logistic_regression  accuracy=0.5056  macro_f1=0.4391
random_forest        accuracy=0.5107  macro_f1=0.4078
gradient_boosting    accuracy=0.5080  macro_f1=0.4015
xgboost              accuracy=0.5157  macro_f1=0.3977
```

The strongest accuracy in this milestone is XGBoost at `0.5157`. The strongest
macro F1 is logistic regression at `0.4391`.

## Deep Learning Experiments

For deep-learning experiments, train a model from a YAML config:

```powershell
python -m src.train --config configs/default.yaml
```

Evaluate a trained model:

```powershell
python -m src.evaluate --config configs/default.yaml
```

## Modeling Plan

The starter code supports:

- Classical ML baselines through scikit-learn-compatible wrappers.
- PyTorch sequence models for team form, including LSTM and Transformer placeholders.
- Config-driven experiments with YAML files.
- Centralized output folders for models, plots, metrics, and logs.

## Next Steps

- Generate sequence datasets from the validated team-centric feature splits.
- Train LSTM and GRU models over team match-history windows.
- Add a Transformer encoder baseline for historical match sequences.
- Explore attention over match history to learn which previous fixtures matter most.
