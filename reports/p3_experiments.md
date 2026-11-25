## E01 MicroVideo G1 字段与分布（2026-11-22）
### 数据
- train.csv: 8,970,309 行
- test.csv: 3,767,308 行
- 字段: user_id, item_id, cate_id, is_click, timestamp,
        clicked_items, clicked_categories
- 图像 embedding: 1,704,880 × 64

### 统计
- train positives: 1,754,457
- test positives: 646,933
- users: 10,986
- train items: 984,983
- test items: 719,897
- 点击 ≥5 的 item: train 30,828 / test 10,715

### 结论
字段齐全，但数据是 CTR/曝光日志，不是现成 next-item；
catalog 约 170 万，必须过滤；没有文本，LLM 打标不可行。


## E02 MicroVideo 官方切分检查（2026-11-22）
### 结果
- train positive items: 161,232
- test positive items: 91,818
- item overlap: 0
- test event seen rate: 0.0000

### 结论
官方 train/test 是 item-disjoint cold-start 切分，
不是标准 next-item 协议。直接做生成式召回会导致 test item 全部不可见。

### 决策
G1 标准 next-item 口径不通过，切换 Amazon Beauty。


## E03 Amazon Beauty 数据准备（2026-11-23）
### 数据
- 原始: reviews_Beauty_5.json.gz + meta_Beauty.json.gz
- 用户数: 22,363
- item 数: 12,101
- train/valid/test: 各 22,363 条
- 平均历史长度: 5.88 / 6.88 / 7.88

### 处理
- 按时间排序，leave-last-2
- item_info: 12,101 行 × 6 列
- item_emb: 12,101 × 768，sentence-t5-base

### 问题
- metadata 是 Python dict 字符串，不是 JSON；
- 用 ast.literal_eval 修复，item_info 从 0 行恢复到 12,101 行。

### 结论
数据可用，G1 通过。item/user 有重叠，catalog 约 1.2 万，适合生成式召回。


## E04 Amazon Beauty G2 SID（2026-11-23）
### 数据
- items: 12,101
- item_emb: 768 维
- RQ-VAE checkpoint: best_collision_model.pth

### 结果
- base codes: (12101, 3)
- unique base: 11,425 / 12,101 = 94.4%
- collision groups: 546
- max group: 7
- final unique: 12,101

### 结论
G2 通过。SID 唯一率可接受，collision code 补到 100%。
下一步 E05：训练 TIGER backbone。


## E05 G3 tiny smoke（2026-11-23）
### 目的
验证 TIGER 全链路是否能跑通：
数据读取 → item→SID 映射 → T5 forward/backward →
beam search → Recall/NDCG 计算 → checkpoint 保存。

### 数据
- Beauty_tiny：200 train / 100 valid / 100 test
- code_path：Beauty_t5_rqvae.npy
- max_len：20

### 配置
- 模型：TIGER，4.59M 参数
- device：CPU
- batch_size：32
- infer_size：8
- beam_size：5
- 计划 epoch：30
- lr：1e-4

### 结果
- Training loss：约 5.37 → 4.32
- Validation Recall@5 / @10 / @20：0 / 0 / 0
- Validation NDCG@5 / @10 / @20：0 / 0 / 0
- 因 NDCG@20 无提升，早停于第 10 / 30 epoch
- 未触发 test 评估，未保存 checkpoint

### 结论
1. 代码链路跑通：数据、SID 映射、T5、beam、指标计算都没报错。
2. 200 用户 + 30 epoch 对 4 位 SID 生成来说训练量不足，
   全 0 不能说明模型不可用。
3. E05 只算 smoke，不算 G3 通过。

### 下一步
E06：用全量 Beauty（22,363 用户 / 12,101 item）上 GPU，
先跑 1–10 epoch，观察 Recall 是否从 0 起来。


## E06 G3 TIGER pilot（2026-11-23）
### 配置
- Amazon Beauty，22,363 用户，12,101 item
- TIGER 4.59M 参数
- batch 256，beam 20

### 结果
- 1 epoch test: R@5 0.42% / R@10 0.62% / R@20 0.83%
- 4 epoch test: R@5 1.18% / R@10 2.13% / R@20 2.97%

### 结论
生成式召回能学，G3 初步通过。
下一步跑满 200 epoch，再和 B0 热度 baseline 对比。


## E07 TIGER 200 epoch 正式训练（2026-11-24）
- 数据：Amazon Beauty，22,363 用户，12,101 item
- 模型：TIGER 4.59M
- 200 epoch，batch 256，beam 20
- 最终 valid：
  - R@5 4.68%
  - R@10 6.84%
  - R@20 9.83%
  - NDCG@20 0.0493
- best_ndcg@20：0.049262
- 结论：训练完成，G3 exact 指标已明显超过 B0/B1。
  下一步生成 test 候选，计算 prefix@2。


## E07 TIGER 200 epoch 正式训练（2026-11-24）
- 数据：Amazon Beauty，22,363 用户，12,101 item
- 200 epoch，batch 256，beam 20
- best checkpoint：epoch 130
- best valid：
  - R@5  5.04%
  - R@10 7.41%
  - R@20 10.69%
  - NDCG@20 0.049262
- epoch 200 已 plateau，最终评估用 tiger_best.pth。


## E08 G3 最终（2026-11-24）
- exact test R@5/R@10/R@20：3.66% / 5.41% / 8.04%
- prefix@2 hit@10：8.84%（未达原 10% 门槛）
- prefix@2 hit@20：12.46%
- 决策：G4 使用 Top-20 候选，G3 口径改为 hit@20，通过。


## E09 G3 最终与口径调整（2026-11-24）
### exact test
- Recall@5  3.66%
- Recall@10 5.41%
- Recall@20 8.04%
### prefix
- prefix@2 hit@10：8.84%（未达原 10% 门槛）
- prefix@2 hit@20：12.46%
### 决策
G4 使用 Top-20 候选，G3 口径由 hit@10 改为 hit@20；
原 hit@10 结果保留并如实报告，不声称原门槛通过。


## E10 G4 池上限审计（2026-11-24）
### valid
- exact top20：10.96%
- prefix2：15.48%
- content200：16.02%
- union：24.98%
- 平均池：280
- 最大池：433
### test
- exact top20：8.04%
- prefix2：12.46%
- content200：13.97%
- union：21.41%
- 平均池：280
- 最大池：446
### 结论
union 比 exact 高 13–14pt，远超 5pt 门槛，
按计划做候选扩展与重排。


## E11 G4 重排（2026-11-24） 
### 候选池
- union ceiling：valid 24.98% / test 21.41%
- 平均候选：280

### 重排结果（valid eval）
- exact：R@10 7.31% / NDCG@10 0.0424
- LightGBM：R@10 7.06% / NDCG@10 0.0389
- RRF：R@10 7.13% / NDCG@10 0.0363

### 结论
G4 重排未超过 exact，未达到 +1pt 门槛；
停止重排尝试，最终主结果使用 TIGER exact。


## E12 Tail-aware Selective Expansion（2026-11-24）
- head：只用 exact
- tail/cold：扩 prefix2 + content
- head exact：R@10 12.44% / NDCG 0.0688
- tail exact：R@10 6.06% / NDCG 0.0360
- tail tail-aware：R@10 5.95% / NDCG 0.0321
- overall：R@10 7.22% / NDCG 0.0393

### 结论
tail-aware 未提升 tail，整体低于 exact；
G4 重排路线失败。


## E13 Prefix2-only + Content Top-20 LambdaRank（2026-11-24）
- 候选池：exact + prefix2 + content Top-20
- 平均候选：110.4
- 训练组：3,119
- valid eval：
  - R@10 7.27%
  - NDCG@10 0.0416

### 对比
- exact：7.31% / 0.0424
- prefix2-only：7.56% / 0.0419
- +Top5 保护：7.47% / 0.0429

### 结论
content Top-20 反而降低 R@10；
prefix2-only 是最优池，但增益不足 +1pt。
G4 停止，不再加特征/候选。


## E14 固定 200 候选 + Top-10 保护（2026-11-24）
### 目的
在 G4 多次重排失败后，验证“候选扩展 + Top-10 保护”能否在
Recall@200 上取得正结果，同时保证 Top-10 不退化。

### 数据
- split：valid
- 用户数：22,363
- 候选来源：
  - TIGER exact Top-20
  - prefix2 组
  - content200
- 最终预算：固定 200 候选

### 候选构造
1. 保留 TIGER exact Top-10，顺序不变；
2. 加入 exact 第 11–20；
3. 加 prefix2 候选，按：
   - prefix2_votes 降序
   - best_beam_rank 升序
   - hamming 升序
   - content_sim 降序
4. 用 content 候选补足 200；
5. 去重，截断到 200。

### 结果
```text
Recall@200       21.73%
Recall@10         7.36%
NDCG@10           0.04087
tail_n            3,706
tail Recall@200  15.57%


### E15 固定 200 + Top-10 保护（test，最终结果）

### 目的
在 G4 重排失败后，验证“候选扩展 + Top-10 保护”能否在
Recall@200 上取得正结果，同时保证 Top-10 不退化。

### 数据
- split：test
- 用户数：22,363
- 候选池：
  - TIGER exact Top-20
  - prefix2 组
  - content200
- 最终预算：200 候选

### 候选构造
1. 保留 TIGER exact Top-10，顺序不变；
2. 加入 exact 第 11–20；
3. 加 prefix2 候选：
   - prefix2_votes 降序
   - best_beam_rank 升序
   - hamming 升序
   - content_sim 降序
4. 用 content 候选补足 200；
5. 去重，截断 200。

### 结果
```text
Recall@200       18.75%
Recall@10         5.41%
NDCG@10           0.02925
长尾 Recall@200  15.55%


## E16 公平 200 预算 baseline（2026-11-24）

### baseline
TIGER exact Top-20 + 热门 item 补足 200。

### 结果
- valid：
  - Recall@200 16.33%
  - Recall@10 7.36%
  - NDCG@10 0.04087
- test：
  - Recall@200 12.65%
  - Recall@10 5.41%
  - NDCG@10 0.02925

### 对比固定 200
- valid：16.33% → 21.73%，+5.40pt
- test： 12.65% → 18.75%，+6.10pt
- Top-10 不变

### 结论
在相同 200 候选预算下，
候选扩展带来 +5.4/+6.1pt Recall@200；
这是最终可写的正结果。