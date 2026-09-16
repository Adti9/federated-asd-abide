import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.clean_data import transform_features


class IterativeFedAvgSimulation:
    def __init__(self, num_rounds=10, local_epochs=5, lr=0.01, random_seed=42):
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.random_seed = random_seed
        self.global_weights = None
        self.global_intercept = None

    def fit_evaluate(self, client_train_data, test_df, cat_features):
        """
        client_train_data: list of dataframes per client
        test_df: held-out test dataframe
        """
        np.random.seed(self.random_seed)

        normalized_clients = []
        for client_df in client_train_data:
            if isinstance(client_df, np.ndarray):
                client_df = pd.DataFrame(client_df, columns=test_df.columns)
            normalized_clients.append(client_df)
        client_train_data = normalized_clients

        # Transform held-out global test set
        X_test, y_test = transform_features(test_df, cat_features)

        # Prepare client feature matrices (deterministic scaling guaranteed)
        client_processed = [transform_features(client_df, cat_features) for client_df in client_train_data]

        n_features = X_test.shape[1]
        self.global_weights = np.zeros(n_features)
        self.global_intercept = np.zeros(1)

        param_bytes = (n_features + 1) * 8
        num_clients = len(client_processed)
        total_upload_bytes = 0
        total_download_bytes = 0

        for _ in range(self.num_rounds):
            local_weights_list = []
            local_intercept_list = []
            sample_counts = []

            total_download_bytes += num_clients * param_bytes

            for X_c, y_c in client_processed:
                client_model = LogisticRegression(
                    max_iter=max(200, self.local_epochs * 200),
                    solver='lbfgs',
                    random_state=self.random_seed,
                )
                client_model.coef_ = self.global_weights.reshape(1, -1).copy()
                client_model.intercept_ = self.global_intercept.copy()

                if len(np.unique(y_c)) > 1:
                    client_model.fit(X_c, y_c)

                local_weights_list.append(client_model.coef_[0].copy())
                local_intercept_list.append(client_model.intercept_[0])
                sample_counts.append(len(y_c))

                total_upload_bytes += param_bytes

            total_samples = sum(sample_counts)
            new_weights = np.zeros(n_features)
            new_intercept = 0.0

            for idx in range(num_clients):
                w_i = sample_counts[idx] / total_samples
                new_weights += local_weights_list[idx] * w_i
                new_intercept += local_intercept_list[idx] * w_i

            self.global_weights = new_weights
            self.global_intercept = np.array([new_intercept])

        eval_model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        eval_model.coef_ = self.global_weights.reshape(1, -1)
        eval_model.intercept_ = self.global_intercept
        eval_model.classes_ = np.array([0, 1])

        probs = eval_model.predict_proba(X_test)[:, 1]
        preds = eval_model.predict(X_test)

        return {
            'roc_auc': roc_auc_score(y_test, probs),
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'total_comm_kb': (total_upload_bytes + total_download_bytes) / 1024.0,
        }
