import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")

train = pd.read_parquet(DATA / "train.parquet")
cnt = Counter()
for h, t in zip(train["history"], train["target"]):
    cnt.update(int(x) for x in h)
    cnt[int(t)] += 1

pop_items = [int(i) for i, _ in cnt.most_common() if 1 <= int(i) <= 12101]


def run(name):
    df = pd.read_parquet(G4 / f"g4_{name}_features_v2.parquet")
    split = pd.read_parquet(DATA / f"{name}.parquet")
    split["user"] = np.arange(len(split))

    selected = {}
    for user, g in df.groupby("user", sort=False):
        g = g[g["exact"] > 0].sort_values("beam_rank")
        items = []
        seen = set()
        for x in g["item"].astype(int):
            if x not in seen:
                seen.add(x)
                items.append(x)
        for x in pop_items:
            if len(items) >= 200:
                break
            if x not in seen:
                seen.add(x)
                items.append(x)
        selected[int(user)] = items[:200]

    n = len(split)
    r200 = 0
    r10 = 0
    ndcg10 = 0.0
    for user, target in zip(split["user"], split["target"]):
        target = int(target)
        items = selected.get(int(user), [])
        if target in items:
            r200 += 1
        if target in items[:10]:
            r10 += 1
            rank = items[:10].index(target) + 1
            ndcg10 += 1.0 / np.log2(rank + 1.0)

    res = {
        "split": name,
        "n": n,
        "recall200": r200 / n,
        "recall10": r10 / n,
        "ndcg10": ndcg10 / n,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


out = {}
out["valid"] = run("valid")
out["test"] = run("test")

with open(G4 / "g4_exact_pop200.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("saved:", G4 / "g4_exact_pop200.json")