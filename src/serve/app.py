import json
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

BASE = Path(r"D:\P3-AmazonBeauty")
SERVICE_PATH = BASE / "outputs" / "service" / "service_top10.parquet"

app = FastAPI(title="P3 Generative Retrieval Service")

# user_id -> {"items": [...], "titles": [...]}
lookup = {}
fallback_items = []
fallback_titles = []


def load_service():
    global lookup, fallback_items, fallback_titles

    import pandas as pd

    df = pd.read_parquet(SERVICE_PATH)

    item_counter = Counter()
    for _, row in df.iterrows():
        items = json.loads(row["items"])
        titles = json.loads(row["titles"])
        titles = [t if isinstance(t, str) else "" for t in titles]
        lookup[int(row["user_id"])] = {
            "items": [int(x) for x in items],
            "titles": titles,
        }
        item_counter.update(int(x) for x in items)

    # 全局兜底：服务表里出现次数最多的 item
    top = [item for item, _ in item_counter.most_common(10)]
    fallback_items = top
    fallback_titles = [
        lookup[next(iter(lookup))]["titles"][0] if False else ""
        for _ in top
    ]

    # 用 item_info 补标题
    item_info = pd.read_parquet(
        BASE / "third_party" / "TIGER" / "data" / "Beauty" / "item_info.parquet"
    )
    title_map = {
    int(k): (v if isinstance(v, str) else "")
    for k, v in zip(item_info["ItemID"], item_info["title"])
}
    for k, v in lookup.items():
        v["titles"] = [title_map.get(i, "") for i in v["items"]]
    fallback_titles = [title_map.get(i, "") for i in fallback_items]


@app.on_event("startup")
def startup():
    load_service()


@app.get("/health")
def health():
    return {"status": "ok", "users": len(lookup)}


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=10),
):
    data = lookup.get(user_id)

    if data is None:
        items = fallback_items[:k]
        titles = fallback_titles[:k]
    else:
        items = data["items"][:k]
        titles = data["titles"][:k]

    if not items:
        raise HTTPException(status_code=404, detail="no recommendation")

    return {
        "user_id": user_id,
        "items": [
            {"item_id": int(i), "title": t}
            for i, t in zip(items, titles)
        ],
    }