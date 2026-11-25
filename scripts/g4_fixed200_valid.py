### 只做 valid 的固定 200 候选 + Top-10 保护:
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT = G4 / "g4_fixed200_valid.json"

# train popularity for tail split
train = pd.read_parquet(DATA / "train.parquet")
cnt = Counter()
for h, t in zip(train["history"], train["target"]):
    cnt.update(int(x) for x in h)
    cnt[int(t)] += 1

freq = np.zeros(13000, dtype=np.int32)
for i, c in cnt.items():
    if 1 <= i < len(freq):
        freq[i] = c

# tail threshold: bottom 20% of positive item frequencies
vals = np.array([v for v in freq[1:] if v > 0])
tail_thr = np.quantile(vals, 0.2)
print("tail threshold:", tail_thr)

df = pd.read_parquet(G4 / "g4_valid_features_v2.parquet")


def build_user_pool(g):
    g = g.copy()
    selected = []
    seen = set()

    def add_rows(rows, limit=None):
        for item in rows["item"].to_numpy():
            if len(selected) >= 200:
                return
            if limit is not None and len(selected) >= limit:
                return
            item = int(item)
            if item not in seen:
                seen.add(item)
                selected.append(item)

    # 1) exact top10, order preserved
    exact = g[g["exact"] > 0].sort_values("beam_rank")
    add_rows(exact.head(10), limit=200)

    # 2) rest of exact top20
    add_rows(exact.iloc[10:20], limit=200)

    # 3) prefix2 candidates
    prefix = g[(g["prefix2"] > 0) & (~g["item"].isin(seen))].copy()
    prefix = prefix.sort_values(
        ["prefix2_votes", "best_beam_rank", "hamming", "content_sim"],
        ascending=[False, True, True, False],
    )
    add_rows(prefix, limit=200)

    # 4) content candidates
    content = g[(g["content"] > 0) & (~g["item"].isin(seen))].copy()
    content = content.sort_values("content_sim", ascending=False)
    add_rows(content, limit=200)

    # 5) fill leftovers
    rest = g[~g["item"].isin(seen)].copy()
    rest = rest.sort_values("content_sim", ascending=False)
    add_rows(rest, limit=200)

    return selected[:200]


def evaluate(selected_map, split_name):
    n = len(split_name)
    rec200 = 0
    rec10 = 0
    ndcg10 = 0.0
    tail_n = 0
    tail_rec200 = 0

    for user, target in zip(split_name["user"], split_name["target"]):
        target = int(target)
        items = selected_map.get(int(user), [])
        if target in items:
            rec200 += 1
        if target in items[:10]:
            rec10 += 1
            rank = items[:10].index(target) + 1
            ndcg10 += 1.0 / np.log2(rank + 1.0)

        if 1 <= target < len(freq) and freq[target] <= tail_thr:
            tail_n += 1
            if target in items:
                tail_rec200 += 1

    return {
        "n": n,
        "recall200": rec200 / n,
        "recall10": rec10 / n,
        "ndcg10": ndcg10 / n,
        "tail_n": tail_n,
        "tail_recall200": tail_rec200 / tail_n if tail_n else 0.0,
    }


# build per-user pools
selected_map = {}
for user, g in df.groupby("user", sort=False):
    selected_map[int(user)] = build_user_pool(g)

# valid uses row index as user in features
split = pd.read_parquet(DATA / "valid.parquet")
split["user"] = np.arange(len(split))
res = evaluate(selected_map, split)
print(json.dumps(res, ensure_ascii=False, indent=2))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("saved:", OUT)