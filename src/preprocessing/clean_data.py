from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Phenotypic_V1_0b.csv"


def load_and_clean_data(csv_path=None):
    csv_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    df = pd.read_csv(csv_path)

    df["target"] = df["DX_GROUP"].apply(lambda x: 1 if x == 1 else 0)

    num_features = ["AGE_AT_SCAN", "FIQ", "VIQ", "PIQ"]
    cat_features = ["SEX", "EYE_STATUS_AT_SCAN", "HANDEDNESS_CATEGORY"]

    for col in num_features:
        df[col] = df[col].replace(-9999, np.nan)

    df["HANDEDNESS_CATEGORY"] = df["HANDEDNESS_CATEGORY"].replace("-9999", np.nan)

    features = num_features + cat_features + ["SITE_ID", "target"]
    clean_df = df[features].copy()

    return clean_df, num_features, cat_features


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
