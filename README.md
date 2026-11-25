# P3: TIGER-style 生成式召回与候选扩展
基于 Amazon Beauty 的 next-item 推荐项目。  
使用 RQ-VAE Semantic ID + TIGER-style Transformer 做生成式召回，并进一步分析候选池可达性与 Top-10 保护。
---
## 1. 项目简介
- 数据：Amazon Beauty 5-core
- 用户数：22,363
- item 数：12,101
- 任务：根据用户历史，预测下一个 item
- 主模型：TIGER-style Transformer（4.59M 参数）
- 训练：200 epoch，best checkpoint 在 epoch 130
核心结论：
- TIGER exact 明显超过 B0/B1；
- 候选扩展 + Top-10 保护在 Recall@200 上有正结果；
- Top-10 未被重排提升，重排尝试作为负结果保留。
---
## 2. 主要结果
### 2.1 Test：TIGER exact
| 指标 | 值 |
|---|---:|
| Recall@5 | 3.66% |
| Recall@10 | 5.41% |
| Recall@20 | 8.04% |
| NDCG@10 | 0.02925 |
Baseline：
| 模型 | Recall@20 |
|---|---:|
| B0 热度 | 1.95% |
| B1 内容 | 2.96% |
| TIGER | **8.04%** |
### 2.2 候选层：固定 200 + Top-10 保护
同 200 候选预算下：
| split | exact + 热门 200 | 固定 200 | 提升 |
|---|---:|---:|---:|
| valid | 16.33% | **21.73%** | +5.40pt |
| test | 12.65% | **18.75%** | +6.10pt |
Top-10：
| split | Recall@10 | NDCG@10 |
|---|---:|---:|
| valid | 7.36% | 0.04087 |
| test | 5.41% | 0.02925 |
长尾 Recall@200：
```text
valid 15.57%
test  15.55%
```
注意：
> 正结果发生在 Recall@200，不是 Top-10 提升；Top-10 被原样保护。
---
## 3. 方法
### 3.1 Semantic ID
- item_emb：12,101 × 768，sentence-t5-base
- RQ-VAE：3 层语义码
- 第 4 位：collision code
- base SID unique：11,425 / 12,101 = 94.4%
- final unique：12,101
### 3.2 生成式召回
- TIGER-style Transformer
  - 4 层 encoder
  - 4 层 decoder
  - d_model 128
  - 4.59M 参数
- 输入：用户历史 SID 序列
- 输出：目标 item SID
- 训练：200 epoch，batch 256，beam 20
- best checkpoint：epoch 130
### 3.3 候选扩展
候选来源：
```text
TIGER exact Top-20
+ prefix2 组
+ content ANN Top-200
```
池上限：
| split | exact top20 | prefix2 | content200 | union |
|---|---:|---:|---:|---:|
| valid | 10.96% | 15.48% | 16.02% | 24.98% |
| test | 8.04% | 12.46% | 13.97% | 21.41% |
### 3.4 固定 200 + Top-10 保护
1. 保留 TIGER exact Top-10，顺序不变；
2. 加入 exact 第 11–20；
3. 加 prefix2 候选：
   - prefix2_votes 降序
   - best_beam_rank 升序
   - hamming 升序
   - content_sim 降序
4. 用 content 候选补足 200；
5. 去重，截断到 200。
---
## 4. 负结果：Top-10 重排
尝试过：
- LightGBM pointwise
- RRF 线性融合
- prefix consensus
- content Top-20 LambdaRank
- tail-aware selective expansion
- beam score fusion
结果：
```text
均未超过 TIGER exact 的 Top-10
```
原因：
- 内容候选噪声大；
- 特征缺少强排序信号；
- 候选可达性 ≠ 可排序；
- 重排容易把原本正确的 Top-10 挤掉。
因此最终保留：
```text
exact Top-10
+ 候选扩展
```
---
## 5. 服务
FastAPI + 预计算查表：
```text
GET /health
GET /recommend/{user_id}?k=10
```
性能：
```text
用户数：22,363
1000 次请求
mean 13.68ms
P50  15.35ms
P99  30.50ms
```
服务只查预计算 Top-10，不实时跑 TIGER。
---
## 6. 目录结构
```text
P3-AmazonBeauty/
  data/
    raw/
    processed/
  src/
    data/
    recall/
    rank/
    serve/
  scripts/
  outputs/
    final_200/
    g4/
    service/
  reports/
    final_results.md
    p3_experiments.md
    tiger_resume.log
  third_party/
    TIGER/
```
---
## 7. 运行
### 数据
```bash
python scripts/prepare_tiger_data.py
python scripts/fix_item_info.py
python scripts/make_item_emb.py
```
### SID
```bash
python scripts/generate_sid.py
```
### 训练
```bash
cd third_party/TIGER/model
python main.py --device cuda \
  --dataset_path ../data/Beauty \
  --code_path ../data/Beauty/Beauty_t5_rqvae.npy \
  --num_epochs 200 \
  --batch_size 256 \
  --beam_size 20
```
### 服务
```bash
uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
```
---
## 8. 局限
- 只使用 Amazon Beauty 一个数据集；
- TIGER 只训练到 200 epoch，未做多种子；
- Top-10 没有提升，正结果只在候选层；
- 重排尝试失败，未找到有效融合信号；
- 没有在线 A/B，只有离线评估；
- 服务是预计算查表，不是在线生成。
---
## 9. 总结
```text
生成式召回：成功
候选扩展 + Top-10 保护：成功（Recall@200）
Top-10 重排：失败（负结果）
```
最终结论：
> TIGER exact 在 Beauty 上明显超过 B0/B1；  
> 固定 200 + Top-10 保护可把 Recall@200 从 12.65% 提升到 18.75%（+6.10pt），  
> 但 Top-10 排序没有被重排提升。
---
## 10. 参考
- TIGER: Recommender Systems with Generative Retrieval
- TIGER 非官方实现：https://github.com/XiaoLongtaoo/TIGER
- SEATER: Generative Retrieval with Semantic Tree-Structured Item Identifiers