# Deep Learning Football Match Prediction

Pre-match football match outcome prediction from team form sequences using classical machine learning and deep learning.

## Project Structure

```text
data/
  raw/premier-league/ # Downloaded source season CSVs
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
- `src/features/build_fixture_features.py` builds one-row-per-fixture features.
- `src/features/sequences.py` builds generated sequence datasets for RNN/GRU/CNN experiments.
- `src/utils/data_utils.py` loads processed splits and validates feature columns.
- `src/models/ml_baselines.py` trains final classical ML baselines.
- `scripts/build_features.py` is the CLI wrapper for feature generation.
- `scripts/build_sequences.py` is the CLI wrapper for canonical sequence generation.

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

The valid pre-match pipeline is:

```text
raw season CSVs
-> raw chronological splits
-> team-centric features
-> fixture-level features
-> sequence datasets
-> ML / RNN / GRU experiments
```

1. Download the source season CSVs:

```powershell
python data\download_data.py
```

This downloads and validates Premier League `season-*.csv` files into:

```text
data/raw/premier-league/
```

You can also place source season CSVs there manually.

Expected filenames use the existing `season-*.csv` convention, for example:

```text
data/raw/premier-league/season-0001.csv
data/raw/premier-league/season-0102.csv
```

2. Create the chronological raw train/validation/test splits:

```powershell
python data\split_data.py
```

This writes:

```text
data/raw/splits/train.csv
data/raw/splits/val.csv
data/raw/splits/test.csv
```

The feature builders require these split files. If they are missing, run
`python data\split_data.py` before building features.

3. Build leakage-free team-centric historical features:

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

4. Build fixture-level features:

```powershell
python scripts\build_fixture_features.py
```

This writes:

```text
data/processed/fixture_level/features.csv
data/processed/fixture_level/train.csv
data/processed/fixture_level/val.csv
data/processed/fixture_level/test.csv
outputs/reports/fixture_level_feature_report.txt
```

5. Build generated sequence datasets for scratch RNN/GRU/CNN experiments:

```powershell
python scripts\build_sequences.py
```

Optional examples:

```powershell
python scripts\build_sequences.py --variants home_only
python scripts\build_sequences.py --sequence-length 10 --variants home_away
python scripts\build_sequences.py --feature-mode raw_plus_rolling --sequence-length 50 --variants home_away
```

This writes generated arrays and metadata under `data/processed/sequences/`.
For `--feature-mode raw_plus_rolling`, outputs are written under
`data/processed/sequences/raw_plus_rolling/`. These files are build artifacts
and are intentionally ignored by git, so a fresh clone must rebuild them before
running scratch RNN/GRU scripts.

The default sequence representation is raw `home_away` history. The
`raw_plus_rolling` mode is optional for controlled comparisons and is not the
default pipeline.

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

## Fixture-Level Baseline

The repo also includes a fixture-level baseline where each original match is one
sample with a home-team perspective target:

```text
0 = HomeWin
1 = Draw
2 = AwayWin
```

Build the fixture-level features:

```powershell
python scripts\build_fixture_features.py
```

Train the fixture-level baselines:

```powershell
python -m src.models.ml_baselines --dataset fixture
```

This writes:

```text
data/processed/fixture_level/features.csv
data/processed/fixture_level/train.csv
data/processed/fixture_level/val.csv
data/processed/fixture_level/test.csv
outputs/reports/fixture_level_feature_report.txt
outputs/reports/fixture_level_baseline_report.txt
outputs/results/fixture_level_baseline_predictions.csv
```

Current fixture-level test results:

```text
dummy_most_frequent  accuracy=0.4416  macro_f1=0.2042
logistic_regression  accuracy=0.4731  macro_f1=0.4237
random_forest        accuracy=0.5211  macro_f1=0.4169
gradient_boosting    accuracy=0.5077  macro_f1=0.3932
xgboost              accuracy=0.5152  macro_f1=0.3908
```

The strongest fixture-level accuracy is Random Forest at `0.5211`. The
strongest fixture-level macro F1 is logistic regression at `0.4237`.

## Deep Learning Experiments

Build sequence datasets before running scratch recurrent experiments:

```powershell
python scripts\build_sequences.py --variants home_only
python scripts\build_sequences.py --sequence-length 10 --variants home_away
```

Train scratch RNN/GRU experiments:

```powershell
python scripts\train_rnn_from_scratch.py
python scripts\train_dual_rnn_from_scratch.py --sequence-length 10 --experiment-name dual_rnn_seq10
python scripts\train_gru_from_scratch.py --sequence-length 10 --experiment-name gru_seq10
```

CNN and CNN+LSTM scripts also load the canonical sequence artifacts through
`src/dataset_cnn.py` or `src/raw_dataset_cnn.py`. Build the needed
`home_away` sequence length first with `scripts\build_sequences.py`.

## Running Scripts Individually

Most generated datasets are ignored by git, so a fresh clone must rebuild the
inputs before running training scripts. Run commands from the repository root.

### One-Time Data Setup

```powershell
python data\download_data.py
python data\split_data.py
```

If you already have the source CSVs, place them under
`data/raw/premier-league/` as `season-*.csv`, then run only
`python data\split_data.py`.

### Feature And Baseline Scripts

| Script | Required input | Build first | Main output |
| --- | --- | --- | --- |
| `scripts\build_features.py` | `data/raw/premier-league/season-*.csv`, `data/raw/splits/*.csv` | `python data\split_data.py` | `data/processed/splits/*.csv`, `outputs/reports/preprocessing_report.txt` |
| `scripts\build_fixture_features.py` | raw season CSVs and raw splits | `python data\split_data.py` | `data/processed/fixture_level/*.csv`, `outputs/reports/fixture_level_feature_report.txt` |
| `python -m src.models.ml_baselines` | `data/processed/splits/*.csv` | `python scripts\build_features.py` | `outputs/reports/final_baseline_report.txt`, `outputs/results/final_baseline_predictions.csv` |
| `python -m src.models.ml_baselines --dataset fixture` | `data/processed/fixture_level/*.csv` | `python scripts\build_fixture_features.py` | `outputs/reports/fixture_level_baseline_report.txt`, `outputs/results/fixture_level_baseline_predictions.csv` |
| `scripts\run_elo_ml_comparison.py` | fixture-level processed splits | `python scripts\build_fixture_features.py` | `outputs/reports/elo_ml_comparison.txt` |
| `scripts\build_elo_validation_report.py` | raw season CSVs | `python data\download_data.py` or manual CSV placement | `outputs/reports/elo_validation_report.txt` |
| `scripts\train_liverpool_baselines.py` | `data/processed/team_centric_features.csv` | `python scripts\build_features.py` | `outputs/reports/liverpool_baseline_report.txt`, `outputs/results/liverpool_baseline_predictions.csv` |
| `scripts\inspect_splits.py` | processed splits | `python scripts\build_features.py` | printed split diagnostics |

### Sequence Builders

| Need | Command | Output |
| --- | --- | --- |
| Default home-only seq5 for `train_rnn_from_scratch.py` | `python scripts\build_sequences.py --variants home_only` | `data/processed/sequences/home_only_*.npz` |
| Home-away seq5 | `python scripts\build_sequences.py --variants home_away` | `data/processed/sequences/home_away_*.npz` |
| Home-away seq10 for dual RNN/GRU | `python scripts\build_sequences.py --sequence-length 10 --variants home_away` | `data/processed/sequences/home_away_seq10_*.npz` |
| CNN raw logic seq3/5/10 sweep | Run the three commands below | canonical home-away seq3, seq5, seq10 |
| Optional raw_plus_rolling comparison | `python scripts\build_sequences.py --feature-mode raw_plus_rolling --sequence-length 10 --variants home_away` | `data/processed/sequences/raw_plus_rolling/` |

For `scripts\cnn_raw_logic.py`, build all configured lengths first:

```powershell
python scripts\build_sequences.py --sequence-length 3 --variants home_away
python scripts\build_sequences.py --sequence-length 5 --variants home_away
python scripts\build_sequences.py --sequence-length 10 --variants home_away
```

### RNN And GRU Scripts

| Script | Required sequence files | Build command | Output |
| --- | --- | --- | --- |
| `scripts\train_rnn_from_scratch.py` | `home_only_train.npz`, `home_only_val.npz`, `home_only_test.npz` | `python scripts\build_sequences.py --variants home_only` | `outputs/reports/rnn_from_scratch_home_only_report.txt` |
| `scripts\train_dual_rnn_from_scratch.py --sequence-length 10 --experiment-name dual_rnn_seq10` | `home_away_seq10_*.npz` | `python scripts\build_sequences.py --sequence-length 10 --variants home_away` | `outputs/reports/dual_rnn_from_scratch_dual_rnn_seq10.txt` |
| `scripts\train_gru_from_scratch.py --sequence-length 10 --experiment-name gru_seq10` | `home_away_seq10_*.npz` | `python scripts\build_sequences.py --sequence-length 10 --variants home_away` | `outputs/reports/gru/gru_seq10.txt` |

### CNN And CNN+LSTM Scripts

CNN scripts use canonical sequence artifacts from `data/processed/sequences/`.
Build the matching `home_away` sequence length before running the script.

| Script group | Default sequence length(s) | Build first | Output |
| --- | ---: | --- | --- |
| `scripts\cnn_raw_logic.py` | 3, 5, 10 | build seq3, seq5, seq10 home-away | `artifacts/cnn_results/cnn_raw_results/` |
| `scripts\cnn_raw_sweep.py` | 10 | `python scripts\build_sequences.py --sequence-length 10 --variants home_away` | `artifacts/cnn_results_raw_sweep/` |
| `scripts\raw_cnn_sweep_lr_dropout_batchsize.py` | 3 | `python scripts\build_sequences.py --sequence-length 3 --variants home_away` | `artifacts/cnn_results_raw_sweep/` |
| `scripts\train_cnn_lstm.py` | 3, 5, 10 | build seq3, seq5, seq10 home-away | `artifacts/cnn_lstm/` |
| `scripts\train_cnn_lstm_2.py` | 3, 5, 10 | build seq3, seq5, seq10 home-away | `artifacts/cnn_lstm_2/` |
| `scripts\train_cnn_1d.py`, `scripts\train_cnn_2d.py`, `scripts\train_home_away_hybrid.py` | see `SEQUENCE_LENGTH` constant in the script | build that home-away sequence length | `artifacts/cnn_results/`, `outputs/models/` |
| tuning scripts such as `cnn_batch_size.py`, `cnn_hybrid.py`, `cnn_k_10.py`, `cnn_k_3.py`, `dropout_cnn.py`, `focal_loss_cnn.py`, `lr_cnn.py` | see `SEQUENCE_LENGTH` constant in each script | build that home-away sequence length | `artifacts/cnn_results/`, `outputs/models/` |

Example:

```powershell
python scripts\build_sequences.py --sequence-length 10 --variants home_away
python scripts\cnn_raw_sweep.py
```

### Validation Commands

```powershell
python -m pytest
python -m compileall src scripts
python scripts\smoke_test.py
```

## Modeling Plan

The starter code supports:

- Classical ML baselines through scikit-learn-compatible wrappers.
- PyTorch sequence models for team form, including RNN, LSTM, and Transformer definitions.
- Scratch RNN and GRU experiments over generated home/away history sequences.
- Config-driven experiments with YAML files.
- Centralized output folders for models, plots, metrics, and logs.

## Next Steps

- Keep generated sequence arrays out of git and rebuild them from raw splits when needed.
- Compare scratch RNN and GRU experiments against the leakage-free ML baselines.
- Add a Transformer encoder baseline for historical match sequences.
- Explore attention over match history to learn which previous fixtures matter most.
