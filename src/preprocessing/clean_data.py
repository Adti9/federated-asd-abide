import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

NUMERICAL_FEATURES = ['AGE_AT_SCAN', 'FIQ', 'VIQ', 'PIQ']
CATEGORICAL_FEATURES = ['SEX', 'HANDEDNESS_CATEGORY']
TARGET_COL = 'target'
SITE_COL = 'SITE_ID'


def load_and_clean_data(csv_path="data/raw/Phenotypic_V1_0b.csv"):
    df = pd.read_csv(csv_path)

    # Target mapping: 1 = ASD, 2 = Control -> (1, 0)
    df['target'] = df['DX_GROUP'].apply(lambda x: 1 if x == 1 else 0)

    # Replace missing sentinels (-9999 -> NaN)
    for col in NUMERICAL_FEATURES:
        df[col] = df[col].replace(-9999, np.nan)
    df['HANDEDNESS_CATEGORY'] = df['HANDEDNESS_CATEGORY'].replace('-9999', np.nan)

    # Deterministic domain fallbacks (no data-driven fitting across clients)
    for col in ['FIQ', 'VIQ', 'PIQ']:
        df[col] = df[col].fillna(100.0)
    df['AGE_AT_SCAN'] = df['AGE_AT_SCAN'].fillna(df['AGE_AT_SCAN'].median())
    df['HANDEDNESS_CATEGORY'] = df['HANDEDNESS_CATEGORY'].fillna('R')
    df['SEX'] = df['SEX'].fillna(1)

    # Deterministic domain feature scaling (ensures consistent weight interpretation across FedAvg clients)
    df['AGE_AT_SCAN'] = df['AGE_AT_SCAN'] / 100.0
    df['FIQ'] = df['FIQ'] / 200.0
    df['VIQ'] = df['VIQ'] / 200.0
    df['PIQ'] = df['PIQ'] / 200.0

    keep_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [SITE_COL, TARGET_COL]
    return df[keep_cols].copy(), NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def transform_features(df, cat_features):
    """Deterministic One-Hot Encoding for categorical variables."""
    encoder = OneHotEncoder(
        categories=[[1, 2], ['R', 'L', 'Ambi', 'Mixed']],
        handle_unknown='ignore',
        sparse_output=False,
    )
    encoded_cat = encoder.fit_transform(df[cat_features])
    cat_cols = encoder.get_feature_names_out(cat_features)

    num_data = df[NUMERICAL_FEATURES].values
    X = np.hstack([num_data, encoded_cat])
    y = df['target'].astype(np.int64).to_numpy()
    return X, y


def build_preprocessor(num_features, cat_features):
    return None


def get_preprocessor(num_features, cat_features):
    return build_preprocessor(num_features, cat_features)
