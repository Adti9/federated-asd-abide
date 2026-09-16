import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.federated.fl_simulation import IterativeFedAvgSimulation
from src.preprocessing.clean_data import build_preprocessor, load_and_clean_data


def run_multi_seed_benchmark():
    df, num_features, cat_features = load_and_clean_data("data/raw/Phenotypic_V1_0b.csv")

    # Filter sites with sample size >= 20 for non-IID partitioning
    site_counts = df["SITE_ID"].value_counts()
    valid_sites = site_counts[site_counts >= 20].index
    filtered_df = df[df["SITE_ID"].isin(valid_sites)].copy()

    seeds = [42, 100, 2024, 7, 99]

    results = {
        "Centralized": {"roc_auc": [], "accuracy": [], "f1": []},
        "Fed_IID": {"roc_auc": [], "accuracy": [], "f1": []},
        "Fed_NonIID": {"roc_auc": [], "accuracy": [], "f1": []},
    }

    for seed in seeds:
        # 1. Stratified Train/Test Split (80% Train, 20% Unseen Test)
        train_df, test_df = train_test_split(
            filtered_df,
            test_size=0.20,
            random_state=seed,
            stratify=filtered_df["target"],
        )

        # A. Centralized Baseline (Preprocess Train -> Transform Test)
        cent_preproc = build_preprocessor(num_features, cat_features)
        X_tr = cent_preproc.fit_transform(train_df[num_features + cat_features])
        X_te = cent_preproc.transform(test_df[num_features + cat_features])
        y_tr, y_te = train_df["target"].values, test_df["target"].values

        cent_model = LogisticRegression(max_iter=1000, random_state=seed)
        cent_model.fit(X_tr, y_tr)
        probs_cent = cent_model.predict_proba(X_te)[:, 1]
        preds_cent = cent_model.predict(X_te)

        results["Centralized"]["roc_auc"].append(roc_auc_score(y_te, probs_cent))
        results["Centralized"]["accuracy"].append(accuracy_score(y_te, preds_cent))
        results["Centralized"]["f1"].append(f1_score(y_te, preds_cent, zero_division=0))

        # B. Federated IID (5 clients)
        shuffled_train = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        iid_splits = np.array_split(shuffled_train, 5)

        sim_iid = IterativeFedAvgSimulation(num_rounds=10, local_epochs=5, random_seed=seed)
        res_iid = sim_iid.fit_evaluate(iid_splits, test_df, num_features, cat_features)
        results["Fed_IID"]["roc_auc"].append(res_iid["roc_auc"])
        results["Fed_IID"]["accuracy"].append(res_iid["accuracy"])
        results["Fed_IID"]["f1"].append(res_iid["f1"])

        # C. Federated Site Non-IID
        site_splits = [group.reset_index(drop=True) for _, group in train_df.groupby("SITE_ID")]
        sim_non_iid = IterativeFedAvgSimulation(num_rounds=10, local_epochs=5, random_seed=seed)
        res_non_iid = sim_non_iid.fit_evaluate(site_splits, test_df, num_features, cat_features)
        results["Fed_NonIID"]["roc_auc"].append(res_non_iid["roc_auc"])
        results["Fed_NonIID"]["accuracy"].append(res_non_iid["accuracy"])
        results["Fed_NonIID"]["f1"].append(res_non_iid["f1"])

    print("\n=== REPAIRED MULTI-SEED BENCHMARK (UNSEEN HELD-OUT TEST SET) ===")
    summary_data = []
    for setting, metrics in results.items():
        auc_mean, auc_std = np.mean(metrics["roc_auc"]), np.std(metrics["roc_auc"])
        acc_mean, acc_std = np.mean(metrics["accuracy"]), np.std(metrics["accuracy"])
        f1_mean, f1_std = np.mean(metrics["f1"]), np.std(metrics["f1"])
        print(
            f"{setting:12s} | ROC-AUC: {auc_mean:.4f} ± {auc_std:.4f} | "
            f"ACC: {acc_mean:.4f} ± {acc_std:.4f} | F1: {f1_mean:.4f} ± {f1_std:.4f}"
        )
        summary_data.append(
            {
                "Setting": setting,
                "ROC_AUC_Mean": auc_mean,
                "ROC_AUC_Std": auc_std,
                "ACC_Mean": acc_mean,
                "ACC_Std": acc_std,
                "F1_Mean": f1_mean,
                "F1_Std": f1_std,
            }
        )

    pd.DataFrame(summary_data).to_csv("results/multi_seed_benchmark.csv", index=False)


if __name__ == "__main__":
    run_multi_seed_benchmark()
