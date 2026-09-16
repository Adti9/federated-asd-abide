import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_data import get_preprocessor, load_and_clean_data

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def compute_shap_feature_importance():
    df, num_features, cat_features = load_and_clean_data("data/raw/Phenotypic_V1_0b.csv")
    feature_cols = num_features + cat_features
    X = df[feature_cols]
    y = df["target"].to_numpy(dtype=int)

    preprocessor = get_preprocessor(num_features, cat_features)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_processed, y)

    explainer = shap.LinearExplainer(model, X_processed, feature_names=feature_names)
    shap_values = explainer(X_processed)

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)
    feature_ranking = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)

    shap.summary_plot(
        shap_values,
        plot_type="dot",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_dot.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.bar(shap_values, max_display=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("=== SHAP Feature Importance (Mean |SHAP value|) ===")
    for name, importance in feature_ranking:
        print(f"{name}: {importance:.6f}")

    return feature_ranking


if __name__ == "__main__":
    compute_shap_feature_importance()
    print(f"\nSaved SHAP plots to: {FIGURES_DIR}")
