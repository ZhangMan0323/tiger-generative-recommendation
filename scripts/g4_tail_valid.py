import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT = G4 / "g4_tail_valid.json"

# popularity from train
train = pd.read_parquet(DATA / "train.parquet")
cnt = Counter()
for h, t in zip(train["history"], train["target"]):
    cnt.update(int(x) for x in h)
    cnt[int(t)] += 1
pop = np.zeros(13000, dtype=np.float32)
for i, c in cnt.items():
    if 1 <= i < len(pop):
        pop[i] = np.log1p(c)

valid_split = pd.read_parquet(DATA / "valid.parquet")
df = pd.read_parquet(G4 / "g4_valid_features_v2.parquet")

# request-level flags
hist_len = valid_split["history"].map(len).to_numpy()
hist_pop = np.array([
    pop[np.asarray(h, dtype=np.int64)].mean() for h in valid_split["history"]
], dtype=np.float32)
cold = hist_len <= 5
med = float(np.median(hist_pop))
head = (hist_pop >= med) & (~cold)
tail = ~head

flags = pd.DataFrame({
    "user": np.arange(len(valid_split), dtype=np.int32),
    "hist_len": hist_len,
    "hist_pop": hist_pop,
    "cold": cold,
    "head": head,
    "tail": tail,
})
df = df.merge(flags, on="user", how="left")
print("head users:", int(flags["head"].sum()),
      "tail users:", int(flags["tail"].sum()),
      "cold:", int(flags["cold"].sum()))

# user split
rng = np.random.default_rng(42)
users = np.arange(len(valid_split))
rng.shuffle(users)
n_train = int(len(users) * 0.8)
train_users = set(users[:n_train].tolist())
eval_users = set(users[n_train:].tolist())

tr = df[df["user"].isin(train_users)].copy()
va = df[df["user"].isin(eval_users)].copy()


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


def exact_score(frame):
    return np.where(
        frame["exact"].to_numpy() > 0,
        -frame["beam_rank"].to_numpy(),
        -1000.0,
    )


def tail_score(frame, w):
    return (
        w[0] * frame["exact"].to_numpy()
        + w[1] * frame["prefix2"].to_numpy()
        + w[2] * frame["content"].to_numpy()
        + w[3] * frame["prefix2_votes"].to_numpy()
        + w[4] * frame["prefix1_votes"].to_numpy()
        + w[5] * frame["content_sim"].to_numpy()
        + w[6] * frame["pop"].to_numpy()
        - w[7] * frame["hamming"].to_numpy()
        - w[8] * frame["beam_rank"].to_numpy()
        - w[9] * frame["content_rank"].to_numpy()
    )


# search only on tail training users
tr_tail = tr[tr["tail"]]
search_users = rng.choice(
    tr_tail["user"].unique(),
    size=min(3000, tr_tail["user"].nunique()),
    replace=False,
)
tr_s = tr_tail[tr_tail["user"].isin(set(search_users.tolist()))]

best = None
for _ in range(150):
    w = np.array([
        rng.uniform(0, 20),   # exact
        rng.uniform(0, 10),   # prefix2
        rng.uniform(0, 5),    # content
        rng.uniform(0, 3),    # prefix2_votes
        rng.uniform(0, 2),    # prefix1_votes
        rng.uniform(0, 3),    # content_sim
        rng.uniform(0, 1),    # pop
        rng.uniform(0, 1),    # hamming
        rng.uniform(0, 0.2),  # beam_rank
        rng.uniform(0, 0.02), # content_rank
    ])
    r, _ = metrics(tr_s, tail_score(tr_s, w), tr_s["user"].nunique(), k=10)
    if best is None or r > best[0]:
        best = (r, w)

print("best tail search R@10:", best[0])
print("weights:", best[1].tolist())

# eval
va_head = va[va["head"]]
va_tail = va[va["tail"]]

# exact baseline
head_exact = metrics(va_head, exact_score(va_head), va_head["user"].nunique(), k=10)
tail_exact = metrics(va_tail, exact_score(va_tail), va_tail["user"].nunique(), k=10)
print("head exact R@10/NDCG:", head_exact)
print("tail exact R@10/NDCG:", tail_exact)

# tail-aware
head_scores = exact_score(va_head)
tail_scores = tail_score(va_tail, best[1])
head_ta = metrics(va_head, head_scores, va_head["user"].nunique(), k=10)
tail_ta = metrics(va_tail, tail_scores, va_tail["user"].nunique(), k=10)
print("head tail-aware R@10/NDCG:", head_ta)
print("tail tail-aware R@10/NDCG:", tail_ta)

# overall weighted
n_all = len(eval_users)
overall_recall = (
    head_ta[0] * va_head["user"].nunique()
    + tail_ta[0] * va_tail["user"].nunique()
) / n_all
overall_ndcg = (
    head_ta[1] * va_head["user"].nunique()
    + tail_ta[1] * va_tail["user"].nunique()
) / n_all
print("overall R@10/NDCG:", overall_recall, overall_ndcg)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({
        "weights": best[1].tolist(),
        "head_exact": head_exact,
        "tail_exact": tail_exact,
        "head_tail_aware": head_ta,
        "tail_tail_aware": tail_ta,
        "overall_recall10": overall_recall,
        "overall_ndcg10": overall_ndcg,
    }, f, ensure_ascii=False, indent=2)

print("saved:", OUT)