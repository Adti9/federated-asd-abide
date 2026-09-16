import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'Phenotypic_V1_0b.csv'
df = pd.read_csv(p)

print('shape:', df.shape)
print('columns:', len(df.columns))
print('first_20_cols:', list(df.columns[:20]))
print('duplicate_rows:', int(df.duplicated().sum()))
print('null_pct_top15:')
print((df.isna().mean() * 100).sort_values(ascending=False).head(15).to_string())
print('\nhead3:')
print(df.head(3).to_string(index=False))

# Quick diagnostics for likely target/site columns
for col in ['DX_GROUP', 'SITE_ID', 'SUB_ID', 'SEX', 'AGE_AT_SCAN', 'FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'SRS_TOTAL_RAW', 'DSM_IV_TR', 'CURRENT_DIAGNOSIS']:
    if col in df.columns:
        print(f'\n--- {col} ---')
        print(df[col].value_counts(dropna=False).head(10).to_string())
