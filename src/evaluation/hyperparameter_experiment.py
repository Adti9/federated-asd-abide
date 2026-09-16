import json
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

LOCAL_EPOCHS = [1, 5, 10]
COMMUNICATION_ROUNDS = [5, 10, 20]
RESULT_PATH = PROJECT_ROOT / "results" / "fl_hyperparameter_tradeoff.csv"
PLOT_PATH = PROJECT_ROOT / "figures" / "fl_convergence_epochs.png"


class FederatedLogisticRegression:
    def __init__(self, n_features):
        self.coef_ = np.zeros((1, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(1, dtype=np.float64)

    def set_parameters(self, coef, intercept):
        self.coef_ = coef.copy()
        self.intercept_ = intercept.copy()


def _prepare_data():
    df, num_features, cat_features = load_and_clean_data()
    feature_cols = num_features + cat_features

    X = df[feature_cols].copy()
    y = df["target"].to_numpy()
    site_ids = df["SITE_ID"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    preprocessor = get_preprocessor(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    train_sites = df.iloc[X_train.index]["SITE_ID"].to_numpy()
    return X_train_proc, y_train, X_test_proc, y_test, train_sites


def _build_iid_partitions(X_train, y_train, num_clients=5):
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X_train))
    splits = np.array_split(indices, num_clients)
    return [(X_train[split], y_train[split]) for split in splits]


def _build_site_non_iid_partitions(X_train, y_train, site_labels, min_samples=10):
    site_to_indices = {}
    for idx, site in enumerate(site_labels):
        site_to_indices.setdefault(site, []).append(idx)

    valid_sites = [site for site, idxs in site_to_indices.items() if len(idxs) >= min_samples]
    partitions = []
    for site in valid_sites:
        idxs = np.array(site_to_indices[site], dtype=int)
        partitions.append((X_train[idxs], y_train[idxs]))
    return partitions


def _local_train(X_client, y_client, local_epochs, seed):
    if np.unique(y_client).size < 2:
        return np.zeros((1, X_client.shape[1]), dtype=np.float64), np.zeros(1, dtype=np.float64), 0

    model = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        random_state=seed,
    )
    model.fit(X_client, y_client)
    coef = model.coef_.copy()
    intercept = model.intercept_.copy()

    for _ in range(1, local_epochs):
        model = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            random_state=seed,
        )
        model.fit(X_client, y_client)
        coef = model.coef_.copy()
        intercept = model.intercept_.copy()

    return coef.reshape(1, -1), intercept.reshape(1), len(y_client)


def _evaluate_global_model(global_model, X_eval, y_eval):
    logits = X_eval @ global_model.coef_.T + global_model.intercept_
    probs = 1.0 / (1.0 + np.exp(-logits.ravel()))
    preds = (probs >= 0.5).astype(int)
    return accuracy_score(y_eval, preds), roc_auc_score(y_eval, probs)


def _run_experiment(partitions, rounds, local_epochs, X_test, y_test):
    if not partitions:
        raise ValueError("No client partitions were provided.")

    n_features = partitions[0][0].shape[1]
    global_model = FederatedLogisticRegression(n_features)

    params_per_client = (n_features + 1) * 8
    total_upload_bytes = 0
    total_download_bytes = 0
    round_history = []

    for round_idx in range(1, rounds + 1):
        local_coefs = []
        local_intercepts = []
        sample_counts = []

        for X_client, y_client in partitions:
            if np.unique(y_client).size < 2:
                continue
            total_download_bytes += params_per_client
            coef, intercept, sample_count = _local_train(X_client, y_client, local_epochs, seed=42 + round_idx)
            local_coefs.append(coef)
            local_intercepts.append(intercept)
            sample_counts.append(sample_count)
            total_upload_bytes += params_per_client

        if not local_coefs:
            raise ValueError("No valid local updates were produced.")

        total_samples = sum(sample_counts)
        new_coef = np.zeros_like(global_model.coef_)
        new_intercept = np.zeros_like(global_model.intercept_)

        for coef, intercept, count in zip(local_coefs, local_intercepts, sample_counts):
            weight = count / total_samples
            new_coef += coef * weight
            new_intercept += intercept * weight

        global_model.set_parameters(new_coef, new_intercept)
        accuracy, roc_auc = _evaluate_global_model(global_model, X_test, y_test)

        round_history.append(
            {
                "round": round_idx,
                "accuracy": round(float(accuracy), 6),
                "roc_auc": round(float(roc_auc), 6),
                "upload_bytes": total_upload_bytes,
                "download_bytes": total_download_bytes,
                "total_communication_bytes": total_upload_bytes + total_download_bytes,
            }
        )

    final = round_history[-1]
    total_kb = (total_upload_bytes + total_download_bytes) / 1024.0
    return {
        "rounds": rounds,
        "local_epochs": local_epochs,
        "final_accuracy": final["accuracy"],
        "final_roc_auc": final["roc_auc"],
        "total_communication_kb": round(float(total_kb), 6),
        "round_history": round_history,
    }


def _save_plot(results):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for setting_idx, setting_name in enumerate(["Federated_IID", "Federated_Site_NonIID"]):
        for round_idx, rounds in enumerate(COMMUNICATION_ROUNDS):
            ax = axes[setting_idx * 3 + round_idx]
            for local_epochs in LOCAL_EPOCHS:
                match = next(
                    (
                        item
                        for item in results
                        if item["setting"] == setting_name and item["rounds"] == rounds and item["local_epochs"] == local_epochs
                    ),
                    None,
                )
                if match is None:
                    continue
                x = [entry["round"] for entry in match["round_history"]]
                y = [entry["roc_auc"] for entry in match["round_history"]]
                ax.plot(x, y, marker="o", linewidth=2, label=f"E={local_epochs}")

            ax.set_title(f"{setting_name} | R={rounds}")
            ax.set_xlabel("Round")
            ax.set_ylabel("ROC-AUC")
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right")

    fig.suptitle("Federated learning convergence under varying local epochs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    X_train, y_train, X_test, y_test, train_sites = _prepare_data()
    results = []

    for setting_name in ["Federated_IID", "Federated_Site_NonIID"]:
        if setting_name == "Federated_IID":
            partitions = _build_iid_partitions(X_train, y_train)
        else:
            partitions = _build_site_non_iid_partitions(X_train, y_train, train_sites)

        for rounds in COMMUNICATION_ROUNDS:
            for local_epochs in LOCAL_EPOCHS:
                experiment = _run_experiment(partitions, rounds, local_epochs, X_test, y_test)
                experiment["setting"] = setting_name
                results.append(experiment)

    rows = []
    for item in results:
        rows.append(
            {
                "setting": item["setting"],
                "rounds": item["rounds"],
                "local_epochs": item["local_epochs"],
                "final_roc_auc": item["final_roc_auc"],
                "final_accuracy": item["final_accuracy"],
                "total_network_communication_kb": item["total_communication_kb"],
                "round_history": json.dumps(item["round_history"]),
            }
        )

    results_df = pd.DataFrame(rows)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULT_PATH, index=False)
    _save_plot(results)

    print(results_df.to_string(index=False))
    print(f"\nSaved results grid to: {RESULT_PATH}")
    print(f"Saved convergence plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
