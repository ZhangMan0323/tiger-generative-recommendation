import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
MODEL_NAME = "sentence-transformers/sentence-t5-base"

df = pd.read_parquet(DATA_DIR / "item_info.parquet")
print("items:", len(df))

def build_text(row):
    return (
        f"'title':{row.get('title', '')}\n"
        f"'price':{row.get('price', '')}\n"
        f"'salesRank':{row.get('salesRank', '')}\n"
        f"'brand':{row.get('brand', '')}\n"
        f"'categories':{row.get('categories', '')}"
    )

texts = [build_text(row) for _, row in df.iterrows()]

model = SentenceTransformer(MODEL_NAME, device="cpu")
emb = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("embedding shape:", emb.shape)

out = pd.DataFrame({
    "ItemID": df["ItemID"].values,
    "embedding": [x.tolist() for x in emb],
})
out.to_parquet(DATA_DIR / "item_emb.parquet", index=False)
print("saved:", DATA_DIR / "item_emb.parquet")