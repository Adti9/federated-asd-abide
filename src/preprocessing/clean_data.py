import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ["AGE_AT_SCAN", "FIQ", "VIQ", "PIQ"]
CATEGORICAL_FEATURES = ["SEX", "HANDEDNESS_CATEGORY"]
TARGET_COL = "target"
SITE_COL = "SITE_ID"


def load_and_clean_data(csv_path="data/raw/Phenotypic_V1_0b.csv"):
    df = pd.read_csv(csv_path)

    # 1. Map target label (DX_GROUP: 1 = ASD, 2 = Control/TD) -> (1 = ASD, 0 = Control)
    df["target"] = df["DX_GROUP"].apply(lambda x: 1 if x == 1 else 0)

    # 2. Clean missing value sentinels (-9999 -> NaN)
    for col in NUMERICAL_FEATURES:
        df[col] = df[col].replace(-9999, np.nan)

    df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].replace("-9999", np.nan)

    # Fill site-wide missing features with standard baseline domain defaults to prevent empty-column crashes
    for col in ["FIQ", "VIQ", "PIQ"]:
        df[col] = df[col].fillna(df[col].median() if not np.isnan(df[col].median()) else 100.0)

    df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].fillna("R")

    keep_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [SITE_COL, TARGET_COL]
    clean_df = df[keep_cols].copy()

    return clean_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(num_features, cat_features):
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    return preprocessor


def get_preprocessor(num_features, cat_features):
    """Backward-compatible alias for project callers expecting the older API name."""
    return build_preprocessor(num_features, cat_features)
