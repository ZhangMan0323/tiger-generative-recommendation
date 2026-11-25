import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT_DIR = Path(r"D:\P3-AmazonBeauty\outputs\service")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# item title
item_info = pd.read_parquet(DATA / "item_info.parquet")
title_map = dict(zip(item_info["ItemID"].astype(int), item_info["title"]))

# popularity fallback
train = pd.read_parquet(DATA / "train.parquet")
cnt = Counter()
for h, t in zip(train["history"], train["target"]):
    cnt.update(int(x) for x in h)
    cnt[int(t)] += 1
pop_items = [int(i) for i, _ in cnt.most_common() if 1 <= int(i) <= 12101][:10]

df = pd.read_parquet(G4 / "g4_test_features_v2.parquet")
test = pd.read_parquet(DATA / "test.parquet")

selected = {}
for user, g in df.groupby("user", sort=False):
    g = g.copy()
    selected_items = []
    seen = set()

    def add(rows):
        for item in rows["item"].to_numpy():
            if len(selected_items) >= 10:
                return
            item = int(item)
            if item not in seen:
                seen.add(item)
                selected_items.append(item)

    exact = g[g["exact"] > 0].sort_values("beam_rank")
    add(exact.head(10))

    if len(selected_items) < 10:
        prefix = g[(g["prefix2"] > 0) & (~g["item"].isin(seen))].copy()
        prefix = prefix.sort_values(
            ["prefix2_votes", "best_beam_rank", "hamming", "content_sim"],
            ascending=[False, True, True, False],
        )
        add(prefix)

    if len(selected_items) < 10:
        add(g[~g["item"].isin(seen)].sort_values("content_sim", ascending=False))

    if len(selected_items) < 10:
        for item in pop_items:
            if item not in seen:
                seen.add(item)
                selected_items.append(item)
            if len(selected_items) >= 10:
                break

    selected[int(user)] = selected_items[:10]

rows = []
for idx, row in test.iterrows():
    user_id = int(row["user"])
    items = selected.get(idx, pop_items[:10])
    rows.append({
        "user_id": user_id,
        "items": json.dumps(items),
        "titles": json.dumps([title_map.get(i, "") for i in items], ensure_ascii=False),
    })

out = pd.DataFrame(rows)
out.to_parquet(OUT_DIR / "service_top10.parquet", index=False)
print(out.shape)
print(out.head(3).to_string())
print("saved:", OUT_DIR / "service_top10.parquet")