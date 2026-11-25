from pathlib import Path
import pandas as pd

SRC = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
DST = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty_tiny")
DST.mkdir(parents=True, exist_ok=True)

for split, n in [("train", 200), ("valid", 100), ("test", 100)]:
    df = pd.read_parquet(SRC / f"{split}.parquet")
    df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    df.to_parquet(DST / f"{split}.parquet", index=False)
    print(split, df.shape)