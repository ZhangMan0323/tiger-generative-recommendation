import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

RQV = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\rqvae")
DATA = Path(r"D:\P3-AmazonBeauty\third_party\TIGER\data\Beauty")
CKPT = RQV / "ckpt" / "Beauty" / "Jun-17-2025_15-21-52" / "best_collision_model.pth"

sys.path.insert(0, str(RQV))
from datasets import EmbDataset
from models.rqvae import RQVAE

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
args = ckpt["args"]
args.data_path = str(DATA / "item_emb.parquet")
args.num_workers = 0

data = EmbDataset(args.data_path)
print("items:", len(data), "dim:", data.dim)

model = RQVAE(
    in_dim=data.dim,
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=args.dropout_prob,
    bn=args.bn,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    beta=args.beta,
    kmeans_init=args.kmeans_init,
    kmeans_iters=args.kmeans_iters,
    sk_epsilons=args.sk_epsilons,
    sk_iters=args.sk_iters,
)
model.load_state_dict(ckpt["state_dict"])
model = model.to("cpu")
model.eval()

loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=0)

all_codes = []
with torch.no_grad():
    for batch in loader:
        indices = model.get_indices(batch, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        all_codes.append(indices)

codes = np.vstack(all_codes)
print("base codes:", codes.shape)

# 第 4 位 collision code
codes4 = np.hstack([codes, np.zeros((len(codes), 1), dtype=int)])

# 对重复 SID 组内编号
uniq, inv, counts = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
dup_groups = np.where(counts > 1)[0]

for g in dup_groups:
    idxs = np.where(inv == g)[0]
    codes4[idxs, -1] = np.arange(len(idxs))

print("unique base:", len(uniq))
print("unique final:", len(np.unique(codes4, axis=0)))
print("collision groups:", len(dup_groups))
print("max group:", int(counts.max()))

np.save(DATA / "Beauty_sid.npy", codes4)
print("saved:", DATA / "Beauty_sid.npy")
print("first 5:")
print(codes4[:5])