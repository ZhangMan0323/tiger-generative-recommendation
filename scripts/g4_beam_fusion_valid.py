import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
FINAL = Path(r"D:\P3-AmazonBeauty\outputs\final_200")
G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT_JSON = G4 / "g4_beam_fusion_valid.json"

# SID raw -> offsets
raw = np.load(DATA / "Beauty_t5_rqvae.npy")
sid = raw.copy()
for level in range(sid.shape[1]):
    sid[:, level] = raw[:, level] + level * 256 + 1

# popularity / tail
train = pd.read_parquet(DATA / "train.parquet")
cnt = Counter()
for h, t in zip(train["history"], train["target"]):
    cnt.update(int(x) for x in h)
    cnt[int(t)] += 1

freq = np.zeros(13000, dtype=np.int32)
for i, c in cnt.items():
    if 1 <= i < len(freq):
        freq[i] = c

vals = np.array([v for v in freq[1:] if v > 0])
tail_thr = np.quantile(vals, 0.2)
print("tail threshold:", tail_thr)


def add_score_features(name):
    feat = pd.read_parquet(G4 / f"g4_{name}_features_v2.parquet")
    feat = feat.sort_values("user").reset_index(drop=True)
    cand = pd.read_parquet(FINAL / f"tiger_{name}_candidates_scores.parquet")

    n = len(feat)
    beam_item_margin = np.zeros(n, dtype=np.float32)
    prefix2_max_margin = np.zeros(n, dtype=np.float32)
    prefix2_mean_margin = np.zeros(n, dtype=np.float32)

    pos = 0
    for user, g in feat.groupby("user", sort=False):
        c = cand.iloc[int(user)]
        items = json.loads(c["beam_items"])
        sids = json.loads(c["beam_sids"])
        scores = json.loads(c["beam_scores"])
        top1 = float(scores[0])

        item_score = {}
        prefix_scores = {}
        for item, code, sc in zip(items, sids, scores):
            item = int(item)
            sc = float(sc)
            item_score[item] = max(item_score.get(item, -1e9), sc)
            pre = tuple(int(x) for x in code[:2])
            prefix_scores.setdefault(pre, []).append(sc)

        prefix_max = {p: max(v) for p, v in prefix_scores.items()}
        prefix_mean = {p: float(np.mean(v)) for p, v in prefix_scores.items()}

        for k, (_, row) in enumerate(g.iterrows()):
            item = int(row["item"])
            code = sid[item - 1]
            pre = tuple(int(x) for x in code[:2])
            idx = pos + k
            beam_item_margin[idx] = item_score.get(item, -1e9) - top1
            prefix2_max_margin[idx] = prefix_max.get(pre, -1e9) - top1
            prefix2_mean_margin[idx] = prefix_mean.get(pre, -1e9) - top1

        pos += len(g)

    feat["beam_item_margin"] = beam_item_margin
    feat["prefix2_max_margin"] = prefix2_max_margin
    feat["prefix2_mean_margin"] = prefix2_mean_margin

    path = G4 / f"g4_{name}_features_v3.parquet"
    feat.to_parquet(path, index=False)
    print(name, feat.shape, "saved:", path)
    return feat


def build_pool(g):
    selected = []
    seen = set()

    def add(rows):
        for item in rows["item"].to_numpy():
            if len(selected) >= 200:
                return
            item = int(item)
            if item not in seen:
                seen.add(item)
                selected.append(item)

    exact = g[g["exact"] > 0].sort_values("beam_rank")
    add(exact.head(10))
    add(exact.iloc[10:20])

    prefix = g[(g["prefix2"] > 0) & (~g["item"].isin(seen))].copy()
    prefix = prefix.sort_values(
        [
            "prefix2_max_margin",
            "prefix2_votes",
            "best_beam_rank",
            "hamming",
            "content_sim",
        ],
        ascending=[False, False, True, True, False],
    )
    add(prefix)

    content = g[(g["content"] > 0) & (~g["item"].isin(seen))].copy()
    content = content.sort_values("content_sim", ascending=False)
    add(content)

    rest = g[~g["item"].isin(seen)].copy()
    rest = rest.sort_values("content_sim", ascending=False)
    add(rest)

    return selected[:200]


def evaluate(selected_map, split):
    n = len(split)
    r200 = 0
    r10 = 0
    ndcg = 0.0
    tail_n = 0
    tail_r200 = 0

    for user, target in zip(split["user"], split["target"]):
        target = int(target)
        items = selected_map.get(int(user), [])

        if target in items:
            r200 += 1

        if target in items[:10]:
            r10 += 1
            rank = items[:10].index(target) + 1
            ndcg += 1.0 / np.log2(rank + 1.0)

        if 1 <= target < len(freq) and freq[target] <= tail_thr:
            tail_n += 1
            if target in items:
                tail_r200 += 1

    return {
        "n": n,
        "recall200": r200 / n,
        "recall10": r10 / n,
        "ndcg10": ndcg / n,
        "tail_n": tail_n,
        "tail_recall200": tail_r200 / tail_n if tail_n else 0.0,
    }


feat = add_score_features("valid")

selected = {}
for user, g in feat.groupby("user", sort=False):
    selected[int(user)] = build_pool(g)

split = pd.read_parquet(DATA / "valid.parquet")
split["user"] = np.arange(len(split))

res = evaluate(selected, split)
print(json.dumps(res, ensure_ascii=False, indent=2))

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("saved:", OUT_JSON)