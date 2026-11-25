import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
CAND = Path(r"D:\P3-AmazonBeauty\outputs\final_200")
OUT = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT.mkdir(parents=True, exist_ok=True)

# item embeddings
emb_df = pd.read_parquet(DATA / "item_emb.parquet")
item_mat = np.stack(emb_df["embedding"].values).astype(np.float32)
item_mat /= np.linalg.norm(item_mat, axis=1, keepdims=True) + 1e-9
item_ids = emb_df["ItemID"].to_numpy(dtype=np.int64)

# SID offsets
raw = np.load(DATA / "Beauty_t5_rqvae.npy")
sid = raw.copy()
for level in range(sid.shape[1]):
    sid[:, level] = raw[:, level] + level * 256 + 1

prefix2 = {}
for idx, code in enumerate(sid):
    pre = tuple(int(x) for x in code[:2])
    prefix2.setdefault(pre, []).append(int(idx + 1))
prefix2 = {k: np.array(v, dtype=np.int64) for k, v in prefix2.items()}

# popularity
train = pd.read_parquet(DATA / "train.parquet")
cnt = Counter()
for h, t in zip(train["history"], train["target"]):
    cnt.update(int(x) for x in h)
    cnt[int(t)] += 1
pop = np.zeros(len(sid) + 1, dtype=np.float32)
for i, c in cnt.items():
    if 1 <= i <= len(sid):
        pop[i] = np.log1p(c)


def build(name, cand_name):
    split_df = pd.read_parquet(DATA / f"{name}.parquet")
    cand_df = pd.read_parquet(CAND / cand_name)
    assert len(split_df) == len(cand_df)

    cols = {k: [] for k in [
        "user", "item", "label", "exact", "prefix2", "prefix1",
        "content", "beam_rank", "hamming", "content_sim", "pop"
    ]}

    for i, (hist, target, beams_json, items_json) in enumerate(zip(
        split_df["history"], split_df["target"],
        cand_df["beam_sids"], cand_df["beam_items"]
    )):
        target = int(target)
        beams = np.array(json.loads(beams_json)[:20], dtype=np.int64)
        exact_items = [int(x) for x in json.loads(items_json)[:20] if int(x) > 0]
        exact_rank = {x: r for r, x in enumerate(exact_items)}

        p_items = []
        for b in beams:
            pre = tuple(int(x) for x in b[:2])
            arr = prefix2.get(pre)
            if arr is not None:
                p_items.append(arr)
        p_items = np.unique(np.concatenate(p_items)) if p_items else np.array([], dtype=np.int64)

        idx = np.asarray(hist, dtype=np.int64) - 1
        u = item_mat[idx].mean(axis=0)
        u /= np.linalg.norm(u) + 1e-9
        scores = item_mat @ u
        top_idx = np.argpartition(-scores, 200)[:200]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        c_items = item_ids[top_idx]

        union = np.array(sorted(set(exact_items) | set(p_items.tolist()) | set(c_items.tolist())), dtype=np.int64)
        cand_sid = sid[union - 1]

        hamming = (cand_sid[:, None, :] != beams[None, :, :]).sum(axis=2).min(axis=1)
        bp1 = {tuple(int(y) for y in b[:1]) for b in beams}
        bp2 = {tuple(int(y) for y in b[:2]) for b in beams}
        p1 = np.array([tuple(int(y) for y in x[:1]) in bp1 for x in cand_sid], dtype=np.float32)
        p2 = np.array([tuple(int(y) for y in x[:2]) in bp2 for x in cand_sid], dtype=np.float32)

        csim = item_mat[union - 1] @ u
        is_exact = np.array([1.0 if x in exact_rank else 0.0 for x in union], dtype=np.float32)
        is_content = np.array([1.0 if x in set(c_items.tolist()) else 0.0 for x in union], dtype=np.float32)
        rank = np.array([exact_rank.get(x, 20) for x in union], dtype=np.int32)
        label = (union == target).astype(np.float32)

        cols["user"].append(np.full(len(union), i, dtype=np.int32))
        cols["item"].append(union)
        cols["label"].append(label)
        cols["exact"].append(is_exact)
        cols["prefix2"].append(p2)
        cols["prefix1"].append(p1)
        cols["content"].append(is_content)
        cols["beam_rank"].append(rank)
        cols["hamming"].append(hamming.astype(np.float32))
        cols["content_sim"].append(csim.astype(np.float32))
        cols["pop"].append(pop[union].astype(np.float32))

        if (i + 1) % 5000 == 0:
            print(name, i + 1, flush=True)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df.to_parquet(OUT / f"g4_{name}_features.parquet", index=False)
    print(name, df.shape, "saved")


build("valid", "tiger_valid_candidates_beam50.parquet")
build("test", "tiger_test_candidates_beam50.parquet")