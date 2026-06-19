import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
ML_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, "ml_artifacts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ML_ARTIFACTS_DIR, exist_ok=True)

X_TRAIN = os.path.join(PROCESSED_DIR, "X_train.csv")
X_VAL = os.path.join(PROCESSED_DIR, "X_val.csv")
X_TEST = os.path.join(PROCESSED_DIR, "X_test.csv")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.csv")
Y_VAL = os.path.join(PROCESSED_DIR, "y_val.csv")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.csv")


def load_data():
    required = {
        "X_train": X_TRAIN,
        "X_val": X_VAL,
        "X_test": X_TEST,
        "y_train": Y_TRAIN,
        "y_val": Y_VAL,
        "y_test": Y_TEST,
    }
    missing = [f"{k}: {v}" for k, v in required.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    X_train = pd.read_csv(X_TRAIN)
    X_val = pd.read_csv(X_VAL)
    X_test = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN).iloc[:, 0]
    y_val = pd.read_csv(Y_VAL).iloc[:, 0]
    y_test = pd.read_csv(Y_TEST).iloc[:, 0]
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def make_sample_weights(y, class_weight_dict):
    return np.array([class_weight_dict[int(label)] for label in y])


def fit_preprocess(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, imputer, scaler


def format_confusion_matrix(cm):
    return pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2"],
        columns=["pred_0", "pred_1", "pred_2"]
    ).to_string()


def evaluate_and_package(name, model, X_train, y_train, X_val, y_val, X_test, y_test, sample_weight_train=None):
    if sample_weight_train is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
    else:
        model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    val_cm = confusion_matrix(y_val, val_pred)
    test_cm = confusion_matrix(y_test, test_pred)

    val_report = classification_report(y_val, val_pred, digits=4)
    test_report = classification_report(y_test, test_pred, digits=4)

    metrics = {
        "model": name,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "val_f1_macro": float(f1_score(y_val, val_pred, average="macro")),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro")),
    }

    return {
        "metrics": metrics,
        "val_report": val_report,
        "test_report": test_report,
        "val_cm": val_cm,
        "test_cm": test_cm,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_proba": val_proba,
        "test_proba": test_proba,
        "model": model,
    }


def save_predictions(path, y_true, y_pred, y_proba=None):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if y_proba is not None:
        for i in range(y_proba.shape[1]):
            df[f"proba_class_{i}"] = y_proba[:, i]
    df.to_csv(path, index=False)


def save_model_configs(configs, class_weights, output_path):
    payload = {"class_weights": class_weights, "models": configs}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_run_report(results_by_model, output_path):
    lines = []
    lines.append("ML RUN REPORT")
    lines.append("=" * 80)

    for name, res in results_by_model.items():
        m = res["metrics"]
        lines.append(f"\nMODEL: {name}")
        lines.append("-" * 80)
        lines.append(f"Validation accuracy: {m['val_accuracy']:.4f}")
        lines.append(f"Test accuracy:       {m['test_accuracy']:.4f}")
        lines.append(f"Validation macro F1: {m['val_f1_macro']:.4f}")
        lines.append(f"Test macro F1:       {m['test_f1_macro']:.4f}")
        lines.append("\nValidation classification report:")
        lines.append(res["val_report"])
        lines.append("\nValidation confusion matrix:")
        lines.append(format_confusion_matrix(res["val_cm"]))
        lines.append("\nTest classification report:")
        lines.append(res["test_report"])
        lines.append("\nTest confusion matrix:")
        lines.append(format_confusion_matrix(res["test_cm"]))
        lines.append("\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def ensemble_predict(models, X):
    probas = [m.predict_proba(X) for m in models]
    avg_proba = np.mean(probas, axis=0)
    return np.argmax(avg_proba, axis=1), avg_proba


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    class_weight_dict = make_class_weights(y_train)
    sample_weight_train = make_sample_weights(y_train, class_weight_dict)

    X_train_s, X_val_s, X_test_s, imputer, scaler = fit_preprocess(X_train, X_val, X_test)

    model_configs = {
        "logistic_regression_weighted": {
            "type": "LogisticRegression",
            "params": {
                "C": 0.1,
                "max_iter": 5000,
                "class_weight": class_weight_dict,
                "random_state": 42
            }
        },
        "svm_rbf_weighted": {
            "type": "SVC",
            "params": {
                "kernel": "rbf",
                "probability": True,
                "C": 0.1,
                "gamma": "scale",
                "class_weight": class_weight_dict,
                "random_state": 42
            }
        },
        "xgboost_weighted": {
            "type": "XGBClassifier",
            "params": {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.01,
                "min_child_weight": 7,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "random_state": 42
            }
        },
        "ensemble_avg_proba": {
            "type": "ensemble",
            "params": {
                "members": ["logistic_regression_weighted", "svm_rbf_weighted", "xgboost_weighted"],
                "strategy": "average probabilities"
            }
        }
    }

    save_model_configs(
        model_configs,
        class_weight_dict,
        os.path.join(ML_ARTIFACTS_DIR, "model_configs.json")
    )

    lr = LogisticRegression(**model_configs["logistic_regression_weighted"]["params"])
    svm = SVC(**model_configs["svm_rbf_weighted"]["params"])
    xgb = XGBClassifier(**model_configs["xgboost_weighted"]["params"])

    results = {}

    results["logistic_regression_weighted"] = evaluate_and_package(
        "logistic_regression_weighted", lr,
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    results["svm_rbf_weighted"] = evaluate_and_package(
        "svm_rbf_weighted", svm,
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    results["xgboost_weighted"] = evaluate_and_package(
        "xgboost_weighted", xgb,
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
        sample_weight_train=sample_weight_train
    )

    ensemble_models = [lr, svm, xgb]
    val_ens_pred, val_ens_proba = ensemble_predict(ensemble_models, X_val_s)
    test_ens_pred, test_ens_proba = ensemble_predict(ensemble_models, X_test_s)

    results["ensemble_avg_proba"] = {
        "metrics": {
            "model": "ensemble_avg_proba",
            "val_accuracy": float(accuracy_score(y_val, val_ens_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_ens_pred)),
            "val_f1_macro": float(f1_score(y_val, val_ens_pred, average="macro")),
            "test_f1_macro": float(f1_score(y_test, test_ens_pred, average="macro")),
        },
        "val_report": classification_report(y_val, val_ens_pred, digits=4),
        "test_report": classification_report(y_test, test_ens_pred, digits=4),
        "val_cm": confusion_matrix(y_val, val_ens_pred),
        "test_cm": confusion_matrix(y_test, test_ens_pred),
        "val_pred": val_ens_pred,
        "test_pred": test_ens_pred,
        "val_proba": val_ens_proba,
        "test_proba": test_ens_proba,
        "model": None,
    }

    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "logistic_regression_weighted_val_predictions.csv"),
        y_val, results["logistic_regression_weighted"]["val_pred"], results["logistic_regression_weighted"]["val_proba"]
    )
    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "logistic_regression_weighted_test_predictions.csv"),
        y_test, results["logistic_regression_weighted"]["test_pred"], results["logistic_regression_weighted"]["test_proba"]
    )

    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "svm_rbf_weighted_val_predictions.csv"),
        y_val, results["svm_rbf_weighted"]["val_pred"], results["svm_rbf_weighted"]["val_proba"]
    )
    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "svm_rbf_weighted_test_predictions.csv"),
        y_test, results["svm_rbf_weighted"]["test_pred"], results["svm_rbf_weighted"]["test_proba"]
    )

    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "xgboost_weighted_val_predictions.csv"),
        y_val, results["xgboost_weighted"]["val_pred"], results["xgboost_weighted"]["val_proba"]
    )
    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "xgboost_weighted_test_predictions.csv"),
        y_test, results["xgboost_weighted"]["test_pred"], results["xgboost_weighted"]["test_proba"]
    )

    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "ensemble_avg_proba_val_predictions.csv"),
        y_val, results["ensemble_avg_proba"]["val_pred"], results["ensemble_avg_proba"]["val_proba"]
    )
    save_predictions(
        os.path.join(ML_ARTIFACTS_DIR, "ensemble_avg_proba_test_predictions.csv"),
        y_test, results["ensemble_avg_proba"]["test_pred"], results["ensemble_avg_proba"]["test_proba"]
    )

    build_run_report(results, os.path.join(ML_ARTIFACTS_DIR, "ml_run_report.txt"))

    summary_df = pd.DataFrame([results[m]["metrics"] for m in results])
    summary_df.to_csv(os.path.join(ML_ARTIFACTS_DIR, "model_comparison_weighted.csv"), index=False)
    print(summary_df.sort_values("val_f1_macro", ascending=False))

    joblib.dump(imputer, os.path.join(ML_ARTIFACTS_DIR, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(ML_ARTIFACTS_DIR, "scaler.pkl"))


if __name__ == "__main__":
    main()