import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated.dataset_partition import create_iid_partitions, create_site_non_iid_partitions
from src.preprocessing.clean_data import get_preprocessor, load_and_clean_data

RESULT_PATH = PROJECT_ROOT / "results" / "shap_stability_comparison.csv"
PLOT_PATH = PROJECT_ROOT / "figures" / "shap_stability_comparison.png"


def _prepare_data():
    df, num_features, cat_features = load_and_clean_data()
    feature_cols = num_features + cat_features

    X = df[feature_cols].copy()
    y = df["target"].to_numpy()

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

    return X_train_proc, y_train, X_test_proc, y_test, preprocessor.get_feature_names_out().tolist()


def _train_centralized_model(X_train, y_train):
    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
    model.fit(X_train, y_train)
    return model


def _train_federated_model(X_train, y_train, setting):
    if setting == "Federated_IID":
        client_partitions = create_iid_partitions(num_clients=5, random_seed=42)
    elif setting == "Federated_Site_NonIID":
        client_partitions = list(create_site_non_iid_partitions(min_samples=20).values())
    else:
        raise ValueError(f"Unknown setting: {setting}")

    if not client_partitions:
        raise ValueError(f"No client partitions available for {setting}.")

    n_features = X_train.shape[1]
    global_coef = np.zeros((1, n_features), dtype=np.float64)
    global_intercept = np.zeros(1, dtype=np.float64)
    total_samples = 0

    for X_client, y_client in client_partitions:
        if np.unique(y_client).size < 2:
            continue
        client_model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
        client_model.fit(X_client, y_client)
        global_coef += client_model.coef_.reshape(1, -1) * len(y_client)
        global_intercept += client_model.intercept_.reshape(1) * len(y_client)
        total_samples += len(y_client)

    if total_samples == 0:
        raise ValueError(f"No valid local models were trained for {setting}.")

    global_coef /= total_samples
    global_intercept /= total_samples

    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
    model.classes_ = np.array([0, 1])
    model.coef_ = global_coef
    model.intercept_ = global_intercept
    return model


def _compute_mean_abs_shap(model, X_eval, feature_names):
    explainer = shap.LinearExplainer(model, X_eval, feature_names=feature_names)
    shap_values = explainer(X_eval)
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[:, :, 0]

    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_index()


def _feature_rank_vector(importances):
    ranked = importances.sort_values(ascending=False).index.tolist()
    rank_map = {feature: rank for rank, feature in enumerate(ranked)}
    return pd.Series([rank_map.get(feature, len(rank_map)) for feature in importances.index], index=importances.index)


def _corr_similarity(v1, v2):
    v1_vals = v1.to_numpy(dtype=float)
    v2_vals = v2.to_numpy(dtype=float)
    return float(np.corrcoef(v1_vals, v2_vals)[0, 1])


def main():
    X_train, y_train, X_test, y_test, feature_names = _prepare_data()

    settings = {
        "Centralized_Baseline": _train_centralized_model(X_train, y_train),
        "Federated_IID": _train_federated_model(X_train, y_train, "Federated_IID"),
        "Federated_Site_NonIID": _train_federated_model(X_train, y_train, "Federated_Site_NonIID"),
    }

    shap_importance = {}
    for setting_name, model in settings.items():
        shap_importance[setting_name] = _compute_mean_abs_shap(model, X_train, feature_names)

    comparison_df = pd.DataFrame(shap_importance)
    comparison_df["feature"] = comparison_df.index
    comparison_df = comparison_df[["feature"] + [col for col in comparison_df.columns if col != "feature"]]

    ranked_vectors = {
        name: _feature_rank_vector(importances)
        for name, importances in shap_importance.items()
    }

    pairwise_rows = []
    setting_names = list(settings.keys())
    for i in range(len(setting_names)):
        for j in range(i + 1, len(setting_names)):
            a, b = setting_names[i], setting_names[j]
            pairwise_rows.append(
                {
                    "setting_a": a,
                    "setting_b": b,
                    "cosine_similarity": _corr_similarity(ranked_vectors[a], ranked_vectors[b]),
                    "pearson_correlation": _corr_similarity(ranked_vectors[a], ranked_vectors[b]),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)

    comparison_path = RESULT_PATH
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    pairwise_path = comparison_path.with_name("shap_stability_similarity.csv")
    pairwise_df.to_csv(pairwise_path, index=False)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(comparison_df))
    width = 0.25
    settings_order = ["Centralized_Baseline", "Federated_IID", "Federated_Site_NonIID"]

    for idx, setting_name in enumerate(settings_order):
        values = comparison_df[setting_name].to_numpy()
        ax.bar(x + (idx - 1) * width, values, width=width, label=setting_name)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP| Value")
    ax.set_title("SHAP feature importance stability across training settings")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=== SHAP Stability Summary ===")
    print(comparison_df.to_string(index=False))
    print("\n=== Rank Correlation Summary ===")
    print(pairwise_df.to_string(index=False))
    print(f"\nSaved comparison table to: {comparison_path}")
    print(f"Saved similarity table to: {pairwise_path}")
    print(f"Saved grouped bar chart to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
