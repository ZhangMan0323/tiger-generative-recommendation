import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data")
DATASET = "Beauty"
OUT_DIR = DATA_DIR / DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)

reviews_gz = DATA_DIR / "reviews_Beauty_5.json.gz"
meta_gz = DATA_DIR / "meta_Beauty.json.gz"
reviews_json = OUT_DIR / "Beauty.json"
meta_json = OUT_DIR / "Beauty_metadata.json"


def parse_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


print("Step 1: reviews -> Beauty.json")
n = 0
with open(reviews_json, "w", encoding="utf-8") as out:
    for obj in parse_gzip(reviews_gz):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n += 1
        if n % 100000 == 0:
            print("  reviews:", n)
print("reviews lines:", n)

print("Step 2: user/item mapping + sequences")
user_mapping = {}
item_mapping = {}
user_items = {}

for obj in parse_gzip(reviews_gz):
    uid = obj["reviewerID"]
    iid = obj["asin"]
    ts = obj["unixReviewTime"]

    if uid not in user_mapping:
        user_mapping[uid] = len(user_mapping) + 1
    if iid not in item_mapping:
        item_mapping[iid] = len(item_mapping) + 1

    u = user_mapping[uid]
    i = item_mapping[iid]
    user_items.setdefault(u, []).append((i, ts))

print("user_num:", len(user_mapping))
print("item_num:", len(item_mapping))

for u in user_items:
    user_items[u].sort(key=lambda x: x[1])
    user_items[u] = [i for i, _ in user_items[u]]

np.save(OUT_DIR / "user_mapping.npy", user_mapping)
np.save(OUT_DIR / "item_mapping.npy", item_mapping)

train_rows, valid_rows, test_rows = [], [], []
dropped = 0
for u, seq in user_items.items():
    if len(seq) < 3:
        dropped += 1
        continue

    train_seq = seq[:-2]
    valid_seq = seq[:-1]
    test_seq = seq

    train_rows.append({"user": u, "history": train_seq[:-1], "target": train_seq[-1]})
    valid_rows.append({"user": u, "history": valid_seq[:-1], "target": valid_seq[-1]})
    test_rows.append({"user": u, "history": test_seq[:-1], "target": test_seq[-1]})

train_df = pd.DataFrame(train_rows)
valid_df = pd.DataFrame(valid_rows)
test_df = pd.DataFrame(test_rows)

train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
valid_df.to_parquet(OUT_DIR / "valid.parquet", index=False)
test_df.to_parquet(OUT_DIR / "test.parquet", index=False)

print("dropped users (<3):", dropped)
print("train:", train_df.shape)
print("valid:", valid_df.shape)
print("test:", test_df.shape)

print("Step 3: metadata -> item_info")
reverse_item = {v: k for k, v in item_mapping.items()}
rows = []
missing = 0

for obj in parse_gzip(meta_gz):
    asin = obj.get("asin")
    if asin not in reverse_item:
        continue
    item_id = item_mapping[asin]
    rows.append({
        "ItemID": item_id,
        "title": obj.get("title"),
        "price": obj.get("price"),
        "salesRank": str(obj.get("salesRank")),
        "brand": obj.get("brand"),
        "categories": json.dumps(obj.get("categories"), ensure_ascii=False),
    })

item_info = pd.DataFrame(rows)
item_info.to_parquet(OUT_DIR / "item_info.parquet", index=False)

print("item_info:", item_info.shape)
print("items without metadata:", len(item_mapping) - len(item_info))
print("saved:", OUT_DIR)