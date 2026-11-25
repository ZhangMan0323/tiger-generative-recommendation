import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT = G4 / "g4_lambdarank_content20.json"

FEATS = [
    "exact", "prefix2", "prefix1",
    "prefix2_votes", "prefix1_votes",
    "best_beam_rank", "best_beam_rank1",
    "beam_rank", "hamming", "content_sim", "content_rank", "pop",
]

df = pd.read_parquet(G4 / "g4_valid_features_v2.parquet")

# 每个用户内，content 候选按 content_sim 排名
df["content_rank_content"] = 9999
mask = df["content"] > 0
df.loc[mask, "content_rank_content"] = (
    df.loc[mask]
    .groupby("user")["content_sim"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# 候选池：exact + prefix2 + content Top-20
pool = df[
    (df["exact"] > 0)
    | (df["prefix2"] > 0)
    | ((df["content"] > 0) & (df["content_rank_content"] <= 20))
].copy()

print("pool:", pool.shape)
print("avg pool:", pool.groupby("user").size().mean())

# 用户切分
rng = np.random.default_rng(42)
users = pool["user"].unique()
rng.shuffle(users)
n_train = int(len(users) * 0.8)
train_users = set(users[:n_train].tolist())
eval_users = set(users[n_train:].tolist())

tr = pool[pool["user"].isin(train_users)].copy()
va = pool[pool["user"].isin(eval_users)].copy()

tr = tr.sort_values("user").reset_index(drop=True)
va = va.sort_values("user").reset_index(drop=True)

# LambdaRank 训练组必须有正样本
tr = tr.groupby("user").filter(lambda g: g["label"].sum() > 0).reset_index(drop=True)

tr_groups = tr.groupby("user").size().to_numpy()
va_groups = va.groupby("user").size().to_numpy()

print("train:", tr.shape, "groups:", len(tr_groups))
print("eval:", va.shape, "groups:", len(va_groups))

X = tr[FEATS].to_numpy(dtype=np.float32)
y = tr["label"].to_numpy(dtype=np.int32)
Xv = va[FEATS].to_numpy(dtype=np.float32)
yv = va["label"].to_numpy(dtype=np.int32)

dtrain = lgb.Dataset(X, label=y, group=tr_groups)
dvalid = lgb.Dataset(Xv, label=yv, group=va_groups, reference=dtrain)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20],
    "label_gain": [0, 1],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 20,
    "verbosity": -1,
    "num_threads": 8,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=400,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)],
)

print("best iter:", model.best_iteration)


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


va_scores = model.predict(va[FEATS].to_numpy(dtype=np.float32))
res = metrics(va, va_scores, va["user"].nunique(), k=10)

print("valid eval R@10:", res[0])
print("valid eval NDCG@10:", res[1])

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({
        "pool_avg": float(pool.groupby("user").size().mean()),
        "train_groups": int(len(tr_groups)),
        "eval_groups": int(len(va_groups)),
        "best_iter": int(model.best_iteration),
        "valid_eval_recall10": res[0],
        "valid_eval_ndcg10": res[1],
    }, f, ensure_ascii=False, indent=2)

print("saved:", OUT)