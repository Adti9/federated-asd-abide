import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from src.preprocessing.clean_data import load_and_clean_data, transform_features


df, _, cat_features = load_and_clean_data('data/raw/Phenotypic_V1_0b.csv')
site_counts = df['SITE_ID'].value_counts()
valid_sites = site_counts[site_counts >= 20].index
filtered_df = df[df['SITE_ID'].isin(valid_sites)].copy()
train_df, test_df = train_test_split(filtered_df, test_size=0.20, random_state=42, stratify=filtered_df['target'])
shuffled_train = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
iid_splits = [pd.DataFrame(split, columns=train_df.columns) for split in np.array_split(shuffled_train, 5)]

for i, client_df in enumerate(iid_splits):
    X_c, y_c = transform_features(client_df, cat_features)
    print('client', i, 'y dtype', y_c.dtype, 'unique', np.unique(y_c), 'shape', X_c.shape)
    model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, max_iter=5, random_state=42)
    print('before classes', getattr(model, 'classes_', 'missing'))
    try:
        model.fit(X_c, y_c)
        print('fit ok', i, model.classes_)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('err', type(e), repr(e))
        raise
    break
