import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.clean_data import get_preprocessor, load_and_clean_data


def create_iid_partitions(num_clients=5, random_seed=42):
    """Split the cleaned dataset randomly across a fixed number of IID clients."""
    df, num_features, cat_features = load_and_clean_data("data/raw/Phenotypic_V1_0b.csv")
    feature_cols = num_features + cat_features

    preprocessor = get_preprocessor(num_features, cat_features)
    X = preprocessor.fit_transform(df[feature_cols])
    y = df["target"].values

    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(len(df))
    splits = np.array_split(indices, num_clients)

    client_data = []
    for split in splits:
        client_data.append((X[split], y[split]))

    return client_data


def create_site_non_iid_partitions(min_samples=20):
    """Partition the dataset by real acquisition site, keeping site-based non-IID splits."""
    df, num_features, cat_features = load_and_clean_data("data/raw/Phenotypic_V1_0b.csv")
    feature_cols = num_features + cat_features

    site_counts = df["SITE_ID"].value_counts()
    valid_sites = site_counts[site_counts >= min_samples].index.tolist()
    filtered_df = df[df["SITE_ID"].isin(valid_sites)].copy()

    preprocessor = get_preprocessor(num_features, cat_features)
    X_all = preprocessor.fit_transform(filtered_df[feature_cols])

    client_data = {}
    for site in valid_sites:
        site_mask = filtered_df["SITE_ID"] == site
        site_indices = filtered_df.index[site_mask]
        site_positions = [filtered_df.index.get_loc(idx) for idx in site_indices]
        client_data[site] = (X_all[site_positions], filtered_df.loc[site_indices, "target"].to_numpy())

    return client_data


if __name__ == "__main__":
    iid = create_iid_partitions(num_clients=5)
    print(f"IID Partition created: {len(iid)} clients.")

    non_iid = create_site_non_iid_partitions(min_samples=20)
    print(f"Site-based Non-IID Partition created: {len(non_iid)} client sites.")
    print("Example site keys:", list(non_iid.keys())[:10])
