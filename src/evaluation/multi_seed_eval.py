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
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_data import get_preprocessor, load_and_clean_data

SEEDS = [42, 100, 2024, 7, 99]
RESULT_PATH = PROJECT_ROOT / "results" / "multi_seed_benchmark.csv"


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def centralized_baseline(seed):
    df, num_features, cat_features = load_and_clean_data()

    X = df[num_features + cat_features]
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    preprocessor = get_preprocessor(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear")
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob)


def _fedavg_train(partitions, rounds=10, seed=42):
    if not partitions:
        raise ValueError("No client partitions were supplied for federated training.")

    n_features = partitions[0][0].shape[1]
    global_coef = np.zeros((1, n_features), dtype=np.float64)
    global_bias = np.zeros(1, dtype=np.float64)

    for _ in range(rounds):
        local_coefs = []
        local_biases = []
        sample_counts = []

        for X_client, y_client in partitions:
            if np.unique(y_client).size < 2:
                continue

            client_model = LogisticRegression(
                max_iter=500,
                solver="liblinear",
                random_state=seed,
            )
            client_model.fit(X_client, y_client)
            local_coefs.append(client_model.coef_.reshape(1, -1))
            local_biases.append(client_model.intercept_.reshape(1))
            sample_counts.append(len(y_client))

        if not local_coefs:
            raise ValueError("No client produced a valid local model.")

        total_samples = sum(sample_counts)
        new_coef = np.zeros_like(global_coef)
        new_bias = np.zeros_like(global_bias)

        for coef, bias, count in zip(local_coefs, local_biases, sample_counts):
            weight = count / total_samples
            new_coef += coef * weight
            new_bias += bias * weight

        global_coef = new_coef
        global_bias = new_bias

    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    model.classes_ = np.array([0, 1])
    model.coef_ = global_coef
    model.intercept_ = global_bias
    return model


def iid_federated_baseline(seed):
    df, num_features, cat_features = load_and_clean_data()
    X = df[num_features + cat_features].copy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    preprocessor = get_preprocessor(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    rng = np.random.RandomState(seed)
    train_indices = rng.permutation(len(X_train_proc))
    client_splits = np.array_split(train_indices, 5)

    partitions = []
    for split in client_splits:
        partitions.append((X_train_proc[split], y_train[split]))

    model = _fedavg_train(partitions, rounds=10, seed=seed)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    return compute_metrics(y_test, y_pred, y_prob)


def site_based_non_iid_baseline(seed):
    df, num_features, cat_features = load_and_clean_data()

    X = df[num_features + cat_features].copy()
    y = df["target"].to_numpy()
    site_labels = df["SITE_ID"].to_numpy()

    X_train, X_test, y_train, y_test, site_train, _ = train_test_split(
        X,
        y,
        site_labels,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    preprocessor = get_preprocessor(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    site_counts = pd.Series(site_train).value_counts()
    valid_sites = site_counts[site_counts >= 10].index.tolist()

    partitions = []
    for site in valid_sites:
        site_mask = pd.Series(site_train) == site
        idx = np.where(site_mask.to_numpy())[0]
        partitions.append((X_train_proc[idx], y_train[idx]))

    model = _fedavg_train(partitions, rounds=10, seed=seed)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    return compute_metrics(y_test, y_pred, y_prob)


def summarize_results(results_by_setting):
    records = []
    for setting, values in results_by_setting.items():
        summary = {
            "Setting": setting,
        }
        for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            values_for_metric = [result[metric] for result in values]
            summary[f"{metric}_mean"] = float(np.mean(values_for_metric))
            summary[f"{metric}_std"] = float(np.std(values_for_metric, ddof=0))
        records.append(summary)

    return pd.DataFrame(records)


def main():
    settings = {
        "Centralized_Baseline": [],
        "Federated_IID": [],
        "Federated_Site_NonIID": [],
    }

    for seed in SEEDS:
        settings["Centralized_Baseline"].append(centralized_baseline(seed))
        settings["Federated_IID"].append(iid_federated_baseline(seed))
        settings["Federated_Site_NonIID"].append(site_based_non_iid_baseline(seed))

    summary_df = summarize_results(settings)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULT_PATH, index=False)

    print("=== Multi-seed benchmark summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved benchmark CSV to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
