import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_data import get_preprocessor, load_and_clean_data
from src.federated.dataset_partition import create_site_non_iid_partitions

RESULT_PATH = PROJECT_ROOT / "results" / "site_level_performance.csv"
PLOT_PATH = PROJECT_ROOT / "figures" / "site_level_comparison.png"


class FederatedLogisticRegression:
    def __init__(self, n_features):
        self.coef_ = np.zeros((1, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(1, dtype=np.float64)

    def set_parameters(self, coef, intercept):
        self.coef_ = coef.copy()
        self.intercept_ = intercept.copy()


def _prepare_site_data():
    df, num_features, cat_features = load_and_clean_data()
    X = df[num_features + cat_features].copy()
    y = df["target"].to_numpy()
    site_labels = df["SITE_ID"].to_numpy()

    X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(
        X,
        y,
        site_labels,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = get_preprocessor(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, y_train, X_test_proc, y_test, site_train, site_test


def _build_site_partitions(X_train, y_train, site_train):
    site_to_indices = {}
    for idx, site in enumerate(site_train):
        site_to_indices.setdefault(site, []).append(idx)

    valid_sites = [site for site, idxs in site_to_indices.items() if len(idxs) >= 10]
    partitions = []
    for site in valid_sites:
        idxs = np.array(site_to_indices[site], dtype=int)
        partitions.append((site, X_train[idxs], y_train[idxs]))
    return partitions


def _fit_local_model(X_site, y_site, seed=42):
    if np.unique(y_site).size < 2:
        return None

    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=seed)
    model.fit(X_site, y_site)
    return model


def _evaluate_model(model, X_eval, y_eval):
    if model is None:
        return np.nan, np.nan

    probs = model.predict_proba(X_eval)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return float(roc_auc_score(y_eval, probs)), float(accuracy_score(y_eval, preds))


def _fit_global_non_iid_model(X_train, y_train, site_train):
    partitions = _build_site_partitions(X_train, y_train, site_train)
    if not partitions:
        raise ValueError("No valid non-IID partitions were created for the global model.")

    n_features = X_train.shape[1]
    global_coef = np.zeros((1, n_features), dtype=np.float64)
    global_intercept = np.zeros(1, dtype=np.float64)

    for _, X_site, y_site in partitions:
        if np.unique(y_site).size < 2:
            continue
        model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
        model.fit(X_site, y_site)
        global_coef += model.coef_.reshape(1, -1) * len(y_site)
        global_intercept += model.intercept_.reshape(1) * len(y_site)

    total_samples = sum(len(y_site) for _, _, y_site in partitions if np.unique(y_site).size >= 2)
    if total_samples == 0:
        raise ValueError("No valid local site samples were available to aggregate the global model.")

    global_coef /= total_samples
    global_intercept /= total_samples

    model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    model.classes_ = np.array([0, 1])
    model.coef_ = global_coef
    model.intercept_ = global_intercept
    return model


def _site_test_indices(site_test, site_name):
    return np.where(site_test == site_name)[0]


def main():
    X_train, y_train, X_test, y_test, site_train, site_test = _prepare_site_data()

    site_names = sorted(np.unique(site_test))
    rows = []

    global_model = _fit_global_non_iid_model(X_train, y_train, site_train)

    for site in site_names:
        test_idx = _site_test_indices(site_test, site)
        if test_idx.size == 0:
            continue

        X_site_test = X_test[test_idx]
        y_site_test = y_test[test_idx]

        local_model = _fit_local_model(X_train[site_train == site], y_train[site_train == site], seed=42)
        local_roc, local_acc = _evaluate_model(local_model, X_site_test, y_site_test)

        global_roc, global_acc = _evaluate_model(global_model, X_site_test, y_site_test)

        rows.append(
            {
                "site": site,
                "sample_count": int(test_idx.size),
                "local_roc_auc": float(local_roc) if not np.isnan(local_roc) else np.nan,
                "local_accuracy": float(local_acc) if not np.isnan(local_acc) else np.nan,
                "global_roc_auc": float(global_roc) if not np.isnan(global_roc) else np.nan,
                "global_accuracy": float(global_acc) if not np.isnan(global_acc) else np.nan,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("site").reset_index(drop=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULT_PATH, index=False)

    plt.figure(figsize=(14, 8))
    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(x - width / 2, results_df["local_roc_auc"], width=width, label="Local-Only", color="#4C72B0")
    plt.bar(x + width / 2, results_df["global_roc_auc"], width=width, label="Global FedAvg Non-IID", color="#55A868")

    plt.xticks(x, results_df["site"], rotation=45, ha="right")
    plt.ylabel("ROC-AUC")
    plt.xlabel("Site ID")
    plt.title("Local vs. Global ROC-AUC by Site")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(results_df.to_string(index=False))
    print(f"\nSaved comparative table to: {RESULT_PATH}")
    print(f"Saved grouped bar chart to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
