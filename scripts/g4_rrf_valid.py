import json
from pathlib import Path

import numpy as np
import pandas as pd

G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT = G4 / "g4_rrf_valid.json"

FEATS = [
    "exact", "prefix2", "prefix1",
    "content_sim", "pop", "hamming", "beam_rank",
]

df = pd.read_parquet(G4 / "g4_valid_features.parquet")

rng = np.random.default_rng(42)
users = df["user"].unique()
rng.shuffle(users)
n_train = int(len(users) * 0.8)
train_users = set(users[:n_train].tolist())
eval_users = set(users[n_train:].tolist())

tr = df[df["user"].isin(train_users)].copy()
va = df[df["user"].isin(eval_users)].copy()

# 为搜索提速，从 train 用户里抽 3000 个
search_users = rng.choice(
    np.array(list(train_users)), size=min(3000, len(train_users)), replace=False
)
tr_s = tr[tr["user"].isin(set(search_users.tolist()))].copy()


def metrics(frame, scores, n_users, k=10):
    tmp = pd.DataFrame({
        "user": frame["user"].to_numpy(),
        "label": frame["label"].to_numpy(),
        "score": scores,
    })
    tmp = tmp.sort_values(["user", "score"], ascending=[True, False])
    tmp["rank"] = tmp.groupby("user").cumcount() + 1
    hits = tmp[(tmp["label"] == 1) & (tmp["rank"] <= k)]
    recall = hits["user"].nunique() / n_users
    ndcg = sum(1.0 / np.log2(r + 1.0) for r in hits["rank"].to_numpy()) / n_users
    return float(recall), float(ndcg)


def score(frame, w):
    return (
        w[0] * frame["exact"].to_numpy()
        + w[1] * frame["prefix2"].to_numpy()
        + w[2] * frame["prefix1"].to_numpy()
        + w[3] * frame["content_sim"].to_numpy()
        + w[4] * frame["pop"].to_numpy()
        - w[5] * frame["hamming"].to_numpy()
        - w[6] * frame["beam_rank"].to_numpy()
    )


best = None
for _ in range(150):
    w = np.array([
        rng.uniform(0, 20),   # exact
        rng.uniform(0, 10),   # prefix2
        rng.uniform(0, 5),    # prefix1
        rng.uniform(0, 3),    # content_sim
        rng.uniform(0, 1),    # pop
        rng.uniform(0, 1),    # hamming
        rng.uniform(0, 0.2),  # beam_rank
    ])
    r, nd = metrics(tr_s, score(tr_s, w), tr_s["user"].nunique(), k=10)
    if best is None or r > best[0]:
        best = (r, nd, w)

print("best search R@10:", best[0])
print("weights:", best[2].tolist())

r_eval, nd_eval = metrics(
    va, score(va, best[2]), va["user"].nunique(), k=10
)
print("valid eval R@10:", r_eval)
print("valid eval NDCG@10:", nd_eval)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({
        "weights": best[2].tolist(),
        "valid_eval_recall10": r_eval,
        "valid_eval_ndcg10": nd_eval,
    }, f, ensure_ascii=False, indent=2)

print("saved:", OUT)