import json
import pandas as pd

P = r"D:\P3-AmazonBeauty\outputs\final_200\tiger_test_candidates_beam50.parquet"
df = pd.read_parquet(P)
n = len(df)

for k in [1, 2, 3, 4]:
    hit = 0
    for t, b in zip(df["target_sid"], df["beam_sids"]):
        target = json.loads(t)
        beams = json.loads(b)
        if any(x[:k] == target[:k] for x in beams[:10]):
            hit += 1
    print(f"prefix@{k} hit@10: {hit / n:.4f} ({hit}/{n})")