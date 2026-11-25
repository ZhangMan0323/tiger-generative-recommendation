import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
G4 = Path(r"D:\P3-AmazonBeauty\outputs\g4")

# SID raw -> TIGER offsets
raw = np.load(DATA / "Beauty_t5_rqvae.npy")
sid = raw.copy()
for level in range(sid.shape[1]):
    sid[:, level] = raw[:, level] + level * 256 + 1

# item_id -> sid
sid_by_item = {i + 1: sid[i] for i in range(len(sid))}


def build(name, cand_name, feat_name):
    cand = pd.read_parquet(
        Path(r"D:\P3-AmazonBeauty\outputs\final_200") / cand_name
    )
    feats = pd.read_parquet(G4 / feat_name)

    rows = []

    for user, beams_json in zip(cand["user"], cand["beam_sids"]):
        beams = json.loads(beams_json)[:20]

        p2 = Counter()
        p1 = Counter()
        best2 = {}
        best1 = {}

        for rank, b in enumerate(beams):
            k2 = tuple(int(x) for x in b[:2])
            k1 = tuple(int(x) for x in b[:1])
            p2[k2] += 1
            p1[k1] += 1
            best2[k2] = min(best2.get(k2, 99), rank)
            best1[k1] = min(best1.get(k1, 99), rank)

        user_feats = feats[feats["user"] == user]
        for item in user_feats["item"].to_numpy():
            code = sid_by_item[int(item)]
            k2 = tuple(int(x) for x in code[:2])
            k1 = tuple(int(x) for x in code[:1])
            rows.append({
                "user": int(user),
                "item": int(item),
                "prefix2_votes": int(p2.get(k2, 0)),
                "prefix1_votes": int(p1.get(k1, 0)),
                "best_beam_rank": int(best2.get(k2, 20)),
                "best_beam_rank1": int(best1.get(k1, 20)),
            })

        if (user + 1) % 5000 == 0:
            print(name, "done", user + 1, flush=True)

    cons = pd.DataFrame(rows)
    out = feats.merge(cons, on=["user", "item"], how="left")
    out[["prefix2_votes", "prefix1_votes"]] = out[
        ["prefix2_votes", "prefix1_votes"]
    ].fillna(0).astype(int)
    out[["best_beam_rank", "best_beam_rank1"]] = out[
        ["best_beam_rank", "best_beam_rank1"]
    ].fillna(20).astype(int)

    out["content_rank"] = (
        out.groupby("user")["content_sim"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    save = G4 / f"g4_{name}_features_v2.parquet"
    out.to_parquet(save, index=False)
    print(name, out.shape, "saved", save, flush=True)


build("valid", "tiger_valid_candidates_beam50.parquet", "g4_valid_features.parquet")
build("test", "tiger_test_candidates_beam50.parquet", "g4_test_features.parquet")