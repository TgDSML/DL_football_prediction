# Leakage-Free Baseline ML Pipeline Milestone

## Dataset Sizes

- Raw matches: 9,870
- Team-centric rows: 19,740
- Rows after history filter: 19,384
- Train split: 12,614 rows
- Validation split: 3,020 rows
- Test split: 3,750 rows

## Preprocessing Summary

The canonical preprocessing path is:

```text
raw CSVs -> team-centric feature engineering -> processed splits -> ML baselines
```

`src/features/build_team_features.py` converts matches into team-perspective
rows, computes shifted rolling features over previous matches, joins opponent
history for the target fixture, and writes model-ready splits under
`data/processed/splits/`.

## Baseline Metrics

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| dummy_most_frequent | 0.3808 | 0.1839 |
| logistic_regression | 0.5056 | 0.4391 |
| random_forest | 0.5107 | 0.4078 |
| gradient_boosting | 0.5080 | 0.4015 |
| xgboost | 0.5157 | 0.3977 |

## Best Models

- Best accuracy: `xgboost` with `0.5157`
- Best macro F1: `logistic_regression` with `0.4391`

## Confirmed Leakage Prevention

Current-match statistics are used only as source values for shifted historical
rolling features. They are not direct inputs for the target match. The final ML
input matrix is loaded from `data/processed/splits/` and validated against
direct leakage columns before model training.

## Next Research Step

Build sequence datasets from the validated processed splits, then compare
LSTM/GRU and Transformer encoder models with attention over match history.
