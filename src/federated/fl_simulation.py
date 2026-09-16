import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.preprocessing.clean_data import build_preprocessor


class IterativeFedAvgSimulation:
    def __init__(self, num_rounds=10, local_epochs=1, lr=0.01, random_seed=42):
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.random_seed = random_seed
        self.global_weights = None
        self.global_intercept = None

    def fit_evaluate(self, client_train_data, test_df, num_features, cat_features):
        """
        client_train_data: dict or list of raw dataframes per client (unpreprocessed)
        test_df: raw dataframe for global evaluation
        """
        np.random.seed(self.random_seed)

        if isinstance(client_train_data, dict):
            client_train_data = list(client_train_data.values())

        normalized_clients = []
        for client_raw in client_train_data:
            if isinstance(client_raw, np.ndarray):
                columns = list(test_df.columns)
                client_raw = pd.DataFrame(client_raw, columns=columns)
            normalized_clients.append(client_raw)

        client_train_data = normalized_clients

        # Fit a shared pooled feature schema for dimension alignment without using pooled statistics in training.
        pooled_train_df = pd.concat(client_train_data, axis=0)
        global_eval_preproc = build_preprocessor(num_features, cat_features)
        X_train_global = global_eval_preproc.fit_transform(pooled_train_df[num_features + cat_features])
        global_feature_names = global_eval_preproc.get_feature_names_out()
        X_test_global = global_eval_preproc.transform(test_df[num_features + cat_features])
        y_test_global = test_df["target"].values

        # 1. Fit local preprocessors independently per client to prevent cross-client leakage
        client_processed = []
        for client_raw in client_train_data:
            preproc = build_preprocessor(num_features, cat_features)
            X_c = preproc.fit_transform(client_raw[num_features + cat_features])
            local_feature_names = preproc.get_feature_names_out()

            aligned_X = np.zeros((X_c.shape[0], X_train_global.shape[1]), dtype=np.float64)
            for j, feat_name in enumerate(local_feature_names):
                if feat_name in global_feature_names:
                    aligned_X[:, np.where(global_feature_names == feat_name)[0][0]] = X_c[:, j]

            y_c = client_raw["target"].values
            client_processed.append((aligned_X, y_c, preproc))

        n_features = X_train_global.shape[1]

        # Initialize Global Model Parameters
        self.global_weights = np.zeros(n_features)
        self.global_intercept = np.zeros(1)

        param_count = n_features + 1
        param_bytes = param_count * 8

        num_clients = len(client_processed)
        total_upload_bytes = 0
        total_download_bytes = 0

        # Iterative FedAvg Loop
        for _ in range(self.num_rounds):
            local_weights_list = []
            local_intercept_list = []
            sample_counts = []

            # Server sends current global model to all clients (Download)
            total_download_bytes += num_clients * param_bytes

            for X_c, y_c, _ in client_processed:
                # Initialize SGD model with current global parameters
                client_model = SGDClassifier(
                    loss="log_loss",
                    learning_rate="constant",
                    eta0=self.lr,
                    max_iter=1,
                    warm_start=True,
                    random_state=self.random_seed,
                )
                client_model.coef_ = self.global_weights.reshape(1, -1).copy()
                client_model.intercept_ = self.global_intercept.copy()

                # Perform true local epochs as repeated partial-fit passes over the client's data.
                if len(np.unique(y_c)) > 1:
                    for _ in range(self.local_epochs):
                        client_model.partial_fit(X_c, y_c, classes=np.array([0, 1]))

                local_weights_list.append(client_model.coef_[0].copy())
                local_intercept_list.append(client_model.intercept_[0])
                sample_counts.append(len(y_c))

                # Client uploads updated parameters back to server
                total_upload_bytes += param_bytes

            # Weighted Aggregation (FedAvg)
            total_samples = sum(sample_counts)
            new_weights = np.zeros(n_features)
            new_intercept = 0.0

            for idx in range(num_clients):
                w_i = sample_counts[idx] / total_samples
                new_weights += local_weights_list[idx] * w_i
                new_intercept += local_intercept_list[idx] * w_i

            self.global_weights = new_weights
            self.global_intercept = np.array([new_intercept])

        # 3. Final Global Model Evaluation on UNSEEN Global Test Set
        eval_model = SGDClassifier(loss="log_loss", random_state=self.random_seed)
        eval_model.classes_ = np.array([0, 1])
        eval_model.coef_ = self.global_weights.reshape(1, -1)
        eval_model.intercept_ = self.global_intercept

        probs = eval_model.predict_proba(X_test_global)[:, 1]
        preds = eval_model.predict(X_test_global)

        return {
            "roc_auc": roc_auc_score(y_test_global, probs),
            "accuracy": accuracy_score(y_test_global, preds),
            "precision": precision_score(y_test_global, preds, zero_division=0),
            "recall": recall_score(y_test_global, preds, zero_division=0),
            "f1": f1_score(y_test_global, preds, zero_division=0),
            "total_comm_kb": (total_upload_bytes + total_download_bytes) / 1024.0,
            "bytes_per_round_kb": ((num_clients * param_bytes * 2) / 1024.0),
        }
