# Deep Learning Football Match Prediction

Pre-match football match outcome prediction from team form sequences using classical machine learning and deep learning.

## Project Structure

```text
data/
  raw/              # Original datasets
  interim/          # Intermediate cleaned files
  processed/        # Model-ready datasets
configs/            # YAML experiment configs
notebooks/          # Jupyter exploration and experiments
outputs/
  models/           # Trained model artifacts
  plots/            # Figures and diagnostics
  results/          # Metrics and prediction outputs
  logs/             # Training logs
src/
  models/           # LSTM, Transformer, and ML model definitions
  training/         # Training loops and orchestration helpers
  preprocessing/    # Raw data cleaning
  features/         # Feature engineering and sequence creation
  evaluation/       # Metrics and reports
  utils/            # Shared utilities
tests/              # Smoke tests and unit tests
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the smoke test:

```powershell
python scripts\smoke_test.py
```

## Typical Workflow

1. Place source data in `data/raw/`.
2. Clean and standardize data with modules under `src/preprocessing/`.
3. Build team-form sequence features with `src/features/`.
4. Train a model from a YAML config:

```powershell
python -m src.train --config configs/default.yaml
```

5. Evaluate a trained model:

```powershell
python -m src.evaluate --config configs/default.yaml
```

## Modeling Plan

The starter code supports:

- Classical ML baselines through scikit-learn-compatible wrappers.
- PyTorch sequence models for team form, including LSTM and Transformer placeholders.
- Config-driven experiments with YAML files.
- Centralized output folders for models, plots, metrics, and logs.
