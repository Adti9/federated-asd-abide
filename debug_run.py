import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.clean_data import load_and_clean_data
from src.federated.fl_simulation import IterativeFedAvgSimulation


df, _, cat_features = load_and_clean_data('data/raw/Phenotypic_V1_0b.csv')
site_counts = df['SITE_ID'].value_counts()
valid_sites = site_counts[site_counts >= 20].index
filtered_df = df[df['SITE_ID'].isin(valid_sites)].copy()
train_df, test_df = train_test_split(filtered_df, test_size=0.20, random_state=42, stratify=filtered_df['target'])
shuffled_train = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
iid_splits = [pd.DataFrame(split, columns=train_df.columns) for split in np.array_split(shuffled_train, 5)]
print('starting fit evaluate')
res = IterativeFedAvgSimulation(num_rounds=2, local_epochs=2, random_seed=42).fit_evaluate(iid_splits, test_df, cat_features)
print(res)
