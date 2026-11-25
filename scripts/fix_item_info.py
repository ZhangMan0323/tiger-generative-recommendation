import ast
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data")
OUT_DIR = DATA_DIR / "Beauty"
META_GZ = DATA_DIR / "meta_Beauty.json.gz"

item_mapping = np.load(OUT_DIR / "item_mapping.npy", allow_pickle=True).item()

rows = []
bad = 0

with gzip.open(META_GZ, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            obj = ast.literal_eval(line)
        except Exception:
            bad += 1
            continue

        asin = obj.get("asin")
        if asin not in item_mapping:
            continue

        item_id = item_mapping[asin]
        rows.append({
            "ItemID": item_id,
            "title": obj.get("title"),
            "price": obj.get("price"),
            "salesRank": json.dumps(obj.get("salesRank"), ensure_ascii=False),
            "brand": obj.get("brand"),
            "categories": json.dumps(obj.get("categories"), ensure_ascii=False),
        })

df = pd.DataFrame(rows)
df.to_parquet(OUT_DIR / "item_info.parquet", index=False)

print("item_info rows:", len(df))
print("bad lines:", bad)
print("items without metadata:", len(item_mapping) - len(df))
print("saved:", OUT_DIR / "item_info.parquet")