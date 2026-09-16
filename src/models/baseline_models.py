import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.clean_data import get_preprocessor, load_and_clean_data


def train_centralized_baseline():
    csv_path = PROJECT_ROOT / "data" / "raw" / "Phenotypic_V1_0b.csv"
    df, num_features, cat_features = load_and_clean_data(csv_path)

    X = df[num_features + cat_features]
    y = df["target"].values

    preprocessor = get_preprocessor(num_features, cat_features)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "accuracy": [],
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_proc, y_train)

        preds = model.predict(X_val_proc)
        probs = model.predict_proba(X_val_proc)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_val, preds))
        metrics["roc_auc"].append(roc_auc_score(y_val, probs))
        metrics["precision"].append(precision_score(y_val, preds, zero_division=0))
        metrics["recall"].append(recall_score(y_val, preds, zero_division=0))
        metrics["f1"].append(f1_score(y_val, preds, zero_division=0))

    print("=== Centralized Logistic Regression Baseline (5-Fold CV) ===")
    for metric, values in metrics.items():
        print(f"{metric.upper():10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    train_centralized_baseline()
