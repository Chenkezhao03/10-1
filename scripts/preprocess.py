import json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

with open('data/merged.geojson', encoding='utf-8') as f:
    gj = json.load(f)
df = pd.DataFrame([feat['properties'] for feat in gj['features']])

feature_cols = ['latitude','longitude','elevation','noaa_depth',
                'osm_piers','osm_lighthouses','osm_breakwaters',
                'osm_harbors','osm_rivers','osm_lakes',
                'osm_beaches','osm_cliffs','osm_industrial']
feature_cols = [c for c in feature_cols if c in df.columns]
print("Features:", feature_cols)

for c in feature_cols:
    if df[c].isna().any():
        fill = df[c].median() if ('elevation' in c or 'depth' in c) else 0
        df[c] = df[c].fillna(fill)

classes = sorted(df['class_code'].unique())
print("Classes:", classes)
df['label'] = df['class_code'].map({c:i for i,c in enumerate(classes)})

os.makedirs('data/processed', exist_ok=True)
with open('data/processed/classes.txt', 'w') as f:
    f.write('\n'.join(classes))

X = StandardScaler().fit_transform(df[feature_cols].values)
y = df['label'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)

def save(p, X, y):
    np.savetxt(p, np.column_stack([X, y]), delimiter=',', fmt='%.6f')

save('data/processed/train.csv', X_tr, y_tr)
save('data/processed/val.csv',   X_va, y_va)
save('data/processed/test.csv',  X_te, y_te)
print(f"Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)}")
