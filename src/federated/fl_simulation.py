import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.federated.dataset_partition import create_iid_partitions, create_site_non_iid_partitions


class FederatedLogisticRegression:
    def __init__(self, n_features):
        self.coef_ = np.zeros((1, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(1, dtype=np.float64)

    def get_parameters(self):
        return self.coef_.copy(), self.intercept_.copy()

    def set_parameters(self, coef, intercept):
        self.coef_ = coef.copy()
        self.intercept_ = intercept.copy()


def train_local_client(X_client, y_client):
    """Fit a logistic regression on one client and return the local model params."""
    if len(np.unique(y_client)) < 2:
        return np.zeros((1, X_client.shape[1])), np.zeros(1), len(y_client)

    model = LogisticRegression(max_iter=200, solver="liblinear", random_state=42)
    model.fit(X_client, y_client)
    return model.coef_.reshape(1, -1), model.intercept_.reshape(1), len(y_client)


def evaluate_global_model(global_model, X, y):
    model = LogisticRegression(max_iter=200, solver="liblinear", random_state=42)
    model.classes_ = np.array([0, 1])
    model.coef_ = global_model.coef_.copy()
    model.intercept_ = global_model.intercept_.copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return accuracy_score(y, preds), roc_auc_score(y, probs)


def run_fedavg_simulation(client_partitions, num_rounds=10):
    """Run a FedAvg-like simulation and track model quality and communication overhead."""
    if isinstance(client_partitions, dict):
        clients_list = list(client_partitions.values())
    else:
        clients_list = client_partitions

    sample_X = clients_list[0][0]
    n_features = sample_X.shape[1]
    global_model = FederatedLogisticRegression(n_features)

    log = []
    total_upload_bytes = 0
    total_download_bytes = 0
    params_per_client = (n_features + 1) * 8

    all_X = np.vstack([client[0] for client in clients_list])
    all_y = np.concatenate([client[1] for client in clients_list])

    for round_idx in range(1, num_rounds + 1):
        local_coefs = []
        local_intercepts = []
        sample_counts = []

        for X_c, y_c in clients_list:
            total_download_bytes += params_per_client
            local_coef, local_intercept, count = train_local_client(X_c, y_c)
            local_coefs.append(local_coef)
            local_intercepts.append(local_intercept)
            sample_counts.append(count)
            total_upload_bytes += params_per_client

        total_samples = sum(sample_counts)
        if total_samples == 0:
            raise ValueError("No valid samples were found in the selected clients.")

        new_coef = np.zeros_like(global_model.coef_)
        new_intercept = np.zeros_like(global_model.intercept_)
        for coef, intercept, count in zip(local_coefs, local_intercepts, sample_counts):
            weight = count / total_samples
            new_coef += coef * weight
            new_intercept += intercept * weight

        global_model.set_parameters(new_coef, new_intercept)
        accuracy, roc_auc = evaluate_global_model(global_model, all_X, all_y)

        round_bytes = total_upload_bytes + total_download_bytes
        log.append(
            {
                "round": round_idx,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "upload_bytes": total_upload_bytes,
                "download_bytes": total_download_bytes,
                "total_bytes": round_bytes,
            }
        )

        print(
            f"Round {round_idx:02d} | Accuracy={accuracy:.4f} | ROC-AUC={roc_auc:.4f} | "
            f"Upload={total_upload_bytes} bytes | Download={total_download_bytes} bytes | Total={round_bytes} bytes"
        )

    final_metrics = log[-1]
    return {
        "final_accuracy": final_metrics["accuracy"],
        "final_roc_auc": final_metrics["roc_auc"],
        "total_upload_bytes": total_upload_bytes,
        "total_download_bytes": total_download_bytes,
        "total_communication_bytes": total_upload_bytes + total_download_bytes,
        "round_metrics": log,
    }


def main():
    print("=== Experiment A: Federated IID ===")
    iid_partitions = create_iid_partitions(num_clients=5, random_seed=42)
    iid_results = run_fedavg_simulation(iid_partitions, num_rounds=10)

    print("\n=== Experiment B: Federated Site-based Non-IID ===")
    non_iid_partitions = create_site_non_iid_partitions(min_samples=20)
    non_iid_results = run_fedavg_simulation(non_iid_partitions, num_rounds=10)

    print("\n=== Comparative Results ===")
    for label, result in [("A (IID)", iid_results), ("B (Site Non-IID)", non_iid_results)]:
        print(f"Experiment {label}:")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Final ROC-AUC: {result['final_roc_auc']:.4f}")
        print(f"  Total Upload Bytes: {result['total_upload_bytes']}")
        print(f"  Total Download Bytes: {result['total_download_bytes']}")
        print(f"  Total Communication Overhead: {result['total_communication_bytes']} bytes")


if __name__ == "__main__":
    main()
