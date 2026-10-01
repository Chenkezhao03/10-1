import json, os, glob

sessions_dir = 'data/sessions'
out_path = 'data/merged.geojson'

all_features = []
for fp in sorted(glob.glob(os.path.join(sessions_dir, '*.geojson'))):
    with open(fp, encoding='utf-8') as f:
        gj = json.load(f)
    if gj.get('type') != 'FeatureCollection':
        continue
    for feat in gj.get('features', []):
        all_features.append(feat)
    print(f"{os.path.basename(fp)}: {len(gj.get('features', []))} points")

merged = {
    "type": "FeatureCollection",
    "name": "merged",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
    },
    "features": all_features
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f)

print(f"\nTotal: {len(all_features)} points -> {out_path}")
