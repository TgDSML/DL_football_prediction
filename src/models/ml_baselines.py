import sys
import os
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.data_utils import FootballDataPipeline

if __name__ == "__main__":
    # --- THE MISSING LINE: Initializing the pipeline ---
    print("Initializing data pipeline...")
    pipeline = FootballDataPipeline(sequence_length=5)
    
    # 1. Load the data using the pipeline
    print("Loading processed datasets...")
    X_train_raw, Y_train = pipeline.process_file('data/interim/train_processed.csv')
    X_val_raw, Y_val = pipeline.process_file('data/interim/val_processed.csv')
    X_test_raw, Y_test = pipeline.process_file('data/interim/test_processed.csv')
    
    # 2. Scale the features (Crucial for Logistic Regression)
    # 2. Scale the features
    print("Scaling features...")
    X_train = pipeline.scaler.fit_transform(X_train_raw)
    X_val = pipeline.scaler.transform(X_val_raw)
    X_test = pipeline.scaler.transform(X_test_raw)
    
    # --- ADD THIS TO INSPECT YOUR DATA ---
    print("\n=== DATA INSPECTOR (FIRST ROW) ===")
    print(f"Shape of X_train: {X_train.shape} (Matches, Features)")
    print("Raw Engineered Array (Before Scaling):")
    print(np.round(X_train_raw[0], 2)) 
    print("Scaled Array (What the model actually sees):")
    print(np.round(X_train[0], 2))
    print("==================================\n")
    
    # 3. Define the models for the bake-off
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=3, 
            learning_rate=0.05, 
            max_depth=5,
            random_state=42
        )
    }

    # 4. Train and Evaluate each model
    print("\n--- ML BAKE-OFF RESULTS ---")
    for name, model in models.items():
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(Y_test, preds)
        print(f"{name} Accuracy: {acc:.3f}")