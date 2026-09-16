import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.audit import log_data_leakage_audit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Phenotypic_V1_0b.csv"
AUDIT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "audit_feature_summary.json"

FINAL_NUM_FEATURES = ["AGE_AT_SCAN", "FIQ", "VIQ", "PIQ"]
FINAL_CAT_FEATURES = ["SEX", "HANDEDNESS_CATEGORY"]
FINAL_FEATURES = FINAL_NUM_FEATURES + FINAL_CAT_FEATURES


def load_and_clean_data(csv_path=None):
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    df = pd.read_csv(csv_path)

    df["target"] = df["DX_GROUP"].apply(lambda x: 1 if x == 1 else 0)

    # Explicitly identify diagnosis-linked assessment instruments before dropping them.
    diagnostic_columns = [
        col for col in df.columns
        if col == "DSM_IV_TR"
        or col.startswith("ADI_R_")
        or col.startswith("ADOS_")
        or col.startswith("SRS_")
        or col.startswith("SCQ_")
        or col.startswith("AQ_")
    ]

    # Replace sentinel values for missingness before any model use.
    for col in FINAL_NUM_FEATURES:
        df[col] = df[col].replace(-9999, np.nan)

    df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].replace("-9999", np.nan)

    # Explicitly drop all diagnosis-linked diagnostic instruments due to leakage risk.
    df = df.drop(columns=diagnostic_columns, errors="ignore")

    clean_df = df[FINAL_FEATURES + ["SITE_ID", "target"]].copy()

    # Save the finalized feature audit summary and explicitly log the dropped leakage variables.
    summary = log_data_leakage_audit(
        df=df,
        final_feature_list=FINAL_FEATURES,
        output_path=AUDIT_SUMMARY_PATH,
        dropped_columns=diagnostic_columns,
    )
    with open(AUDIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return clean_df, FINAL_NUM_FEATURES, FINAL_CAT_FEATURES


def get_preprocessor(num_features, cat_features):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    df, num_cols, cat_cols = load_and_clean_data(DEFAULT_CSV_PATH)
    print("Cleaned Data Preview:")
    print(df.head())
    print(f"\nTotal Records: {len(df)}")
    print(f"Final numeric features: {num_cols}")
    print(f"Final categorical features: {cat_cols}")
    print(f"Audit summary saved to: {AUDIT_SUMMARY_PATH}")
