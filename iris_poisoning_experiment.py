"""
Iris Dataset Poisoning Experiment
---------------------------------
This script performs controlled data poisoning at multiple ratios (0%, 5%, 10%, 50%)
and trains a Logistic Regression classifier on the resulting datasets.
All experiments are tracked using MLflow.


Author: Munawwar
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
import random
import os
from datetime import datetime

# =========================================================
# 1️⃣ Configuration & Environment Setup
# =========================================================

SEED = 42
POISON_LEVELS = [0.0, 0.05, 0.10, 0.50]
NOISE_LEVEL = 1.5

np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Set MLflow experiment name
mlflow.set_experiment("Iris_Poisoning_Experiment_Pro")

# =========================================================
# 2️⃣ Data Loading
# =========================================================

def load_data():
    """Loads the Iris dataset as a pandas DataFrame."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y

# =========================================================
# 3️⃣ Poisoning Logic
# =========================================================

def poison_data(X, y, poison_ratio=0.1, noise_level=1.0):
    """
    Applies feature noise and random label flipping.

    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        poison_ratio (float): % of rows to poison
        noise_level (float): Noise scale

    Returns:
        (X_poisoned, y_poisoned)
    """
    X_poisoned = X.copy()
    y_poisoned = y.copy()

    n_samples = int(len(X) * poison_ratio)
    indices = np.random.choice(X.index, n_samples, replace=False)

    # Feature noise
    X_poisoned.loc[indices] += np.random.normal(0, noise_level, X.shape[1])

    # Label flipping
    for i in indices:
        original = y_poisoned[i]
        choices = [c for c in [0, 1, 2] if c != original]
        y_poisoned[i] = np.random.choice(choices)

    return X_poisoned, y_poisoned

# =========================================================
# 4️⃣ Training + MLflow Logging
# =========================================================

def train_and_log_model(X, y, poison_ratio):
    """
    Trains a Logistic Regression model and logs metrics + artifacts to MLflow.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Start MLflow Run
        with mlflow.start_run(run_name=f"Poison_{int(poison_ratio*100)}%"):

            # Params
            mlflow.log_param("poison_ratio", poison_ratio)
            mlflow.log_param("noise_level", NOISE_LEVEL)
            mlflow.log_param("model", "LogisticRegression")

            # Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Tags
            mlflow.set_tag("developer", "<your name>")
            mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
            mlflow.set_tag("description", "Iris poisoning robustness experiment")

            # Store model
            mlflow.sklearn.log_model(model, "model")

        print(f"✓ Completed | Poison {int(poison_ratio*100)}% | "
              f"Acc: {acc:.4f} | F1: {f1:.4f}")

    except Exception as e:
        print("❌ Error during training:", str(e))

# =========================================================
# 5️⃣ Main Experiment
# =========================================================

def run_experiment():
    X, y = load_data()
    print("🚀 Starting Iris Data Poisoning Experiment")

    for ratio in POISON_LEVELS:
        print(f"\n---- Running experiment for {int(ratio*100)}% poisoning ----")
        X_poisoned, y_poisoned = poison_data(X, y, poison_ratio=ratio, noise_level=NOISE_LEVEL)
        train_and_log_model(X_poisoned, y_poisoned, ratio)

    print("\n🎉 All experiments completed. View results in MLflow UI.")

# =========================================================
# 6️⃣ Entry Point
# =========================================================

if __name__ == "__main__":
    run_experiment()

