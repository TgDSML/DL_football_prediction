from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

TARGET = "FTR"
CLASS_NAMES = ["Away win", "Draw", "Home win"]

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "interim" / "train_processed.csv"
TEST_PATH = BASE_DIR / "data" / "interim" / "test_processed.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "baseline"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

y_train = df_train[TARGET]
y_test = df_test[TARGET]

X_train_dummy = np.zeros((len(y_train), 1))
X_test_dummy = np.zeros((len(y_test), 1))

clf = DummyClassifier(strategy="stratified", random_state=42)
clf.fit(X_train_dummy, y_train)

y_pred = clf.predict(X_test_dummy)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

class_priors = {
    str(cls): float(p)
    for cls, p in zip(clf.classes_, clf.class_prior_)
}

lines = [
    "STRATIFIED NAIVE BASELINE - TEST SET",
    "=" * 60,
    f"Samples: {len(y_test)}",
    f"Accuracy: {acc:.4f}",
    f"Macro F1: {macro_f1:.4f}",
    f"Weighted F1: {weighted_f1:.4f}",
    "",
    "Train class probabilities:",
    str(class_priors),
    "",
    "Interpretation:",
    "This model does not use match features.",
    "It guesses outcomes randomly according to the class distribution in the training set.",
    "Its accuracy estimates how often a naive probability-based guess would be correct.",
    "",
    "Confusion Matrix (rows=true, cols=pred):",
    str(cm),
    "",
    "Classification Report:",
    report,
]

output_file = ARTIFACTS_DIR / "stratified_test_metrics.txt"
output_file.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved: {output_file}")