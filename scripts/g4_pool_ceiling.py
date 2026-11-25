import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
CAND = Path(r"D:\P3-AmazonBeauty\outputs\final_200")
OUT = Path(r"D:\P3-AmazonBeauty\outputs\g4")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- item embedding ----------
emb_df = pd.read_parquet(DATA / "item_emb.parquet")
item_mat = np.stack(emb_df["embedding"].values).astype(np.float32)
item_mat /= np.linalg.norm(item_mat, axis=1, keepdims=True) + 1e-9
item_ids = emb_df["ItemID"].to_numpy(dtype=np.int64)
print("item_emb:", item_mat.shape)

# ---------- SID: raw -> TIGER offsets ----------
raw = np.load(DATA / "Beauty_t5_rqvae.npy")
sid = raw.copy()
for level in range(sid.shape[1]):
    sid[:, level] = raw[:, level] + level * 256 + 1

prefix2 = {}
for idx, code in enumerate(sid):
    pre = tuple(int(x) for x in code[:2])
    prefix2.setdefault(pre, []).append(int(idx + 1))
prefix2 = {k: np.array(v, dtype=np.int64) for k, v in prefix2.items()}
print("prefix2 groups:", len(prefix2),
      "avg size:", float(np.mean([len(v) for v in prefix2.values()])))


def run_split(name, cand_name):
    split = pd.read_parquet(DATA / f"{name}.parquet")
    cand = pd.read_parquet(CAND / cand_name)
    assert len(split) == len(cand), (name, len(split), len(cand))

    n = len(split)
    hit = {"exact": 0, "prefix2": 0, "content": 0, "union": 0}
    pool_sizes = []

    for i, (hist, target, beams_json, items_json) in enumerate(zip(
        split["history"],
        split["target"],
        cand["beam_sids"],
        cand["beam_items"],
    )):
        target = int(target)

        # 1) TIGER exact top20
        beams = json.loads(beams_json)[:20]
        exact_items = [
            int(x) for x in json.loads(items_json)[:20] if int(x) > 0
        ]

        # 2) prefix2 expansion
        p_items = []
        for b in beams:
            pre = tuple(int(x) for x in b[:2])
            arr = prefix2.get(pre)
            if arr is not None:
                p_items.append(arr)
        p_items = (
            np.unique(np.concatenate(p_items))
            if p_items else np.array([], dtype=np.int64)
        )

        # 3) content ANN top200
        idx = np.asarray(hist, dtype=np.int64) - 1
        u = item_mat[idx].mean(axis=0)
        u /= np.linalg.norm(u) + 1e-9
        scores = item_mat @ u
        top_idx = np.argpartition(-scores, 200)[:200]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        content_items = item_ids[top_idx]

        # 4) union
        union = (
            set(exact_items)
            | set(int(x) for x in p_items)
            | set(int(x) for x in content_items)
        )
        pool_sizes.append(len(union))

        if target in exact_items:
            hit["exact"] += 1
        if target in p_items:
            hit["prefix2"] += 1
        if target in content_items:
            hit["content"] += 1
        if target in union:
            hit["union"] += 1

        if (i + 1) % 5000 == 0:
            print(name, "done", i + 1, flush=True)

    result = {
        "split": name,
        "n": n,
        "exact_top20": hit["exact"] / n,
        "prefix2": hit["prefix2"] / n,
        "content200": hit["content"] / n,
        "union": hit["union"] / n,
        "avg_pool": float(np.mean(pool_sizes)),
        "max_pool": int(np.max(pool_sizes)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


results = {}
results["valid"] = run_split(
    "valid", "tiger_valid_candidates_beam50.parquet"
)
results["test"] = run_split(
    "test", "tiger_test_candidates_beam50.parquet"
)

out_path = OUT / "g4_union_ceiling.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("saved:", out_path)