from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, multi_class="auto")),
        ]
    )


def build_random_forest_baseline(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
