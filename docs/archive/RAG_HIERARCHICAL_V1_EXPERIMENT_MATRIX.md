# SuperHermes RAG Hierarchical V1 最小实验矩阵

## 1. 目的

这份文稿只回答一个问题：

> 当前 Hierarchical v1 是否真的让 RAG 主链路更适合当前项目。

它不是完整实验平台规范，只定义当前阶段必须跑的最小对比。

本方案参考 DeepEval、Ragas、TruLens 的 RAG 评估思路，但当前阶段不直接引入新依赖。原因是项目目标是先把现有 RAG 稳定评估起来，而不是把评估框架平台化。

---

## 2. 评估设计依据

三个主流 RAG 评估框架给出的共同启发是：

| 框架 | 可借鉴的核心思路 | 对当前项目的落地方式 |
| --- | --- | --- |
| DeepEval | `Contextual Precision` 看检索排序质量；`Contextual Recall` 看检索上下文是否覆盖 expected output；`Contextual Relevancy` 看上下文是否相关；`Faithfulness` 和 `Answer Relevancy` 看生成质量 | 当前先用确定性指标近似 retrieval 层；小样本再接 LLM judge |
| Ragas | `Context Precision / Recall`、`Faithfulness`、`Answer Relevancy`、`Factual Correctness`、`Noise Sensitivity`；还支持 ID-based context precision | 当前优先使用 root/anchor/keyword/ID 口径；后续再接 LLM-based 指标 |
| TruLens | RAG Triad：`context relevance`、`groundedness`、`answer relevance` | 作为最终 E2E 质量框架，但本轮先只覆盖 retrieval 和低置信度 gate |

官方参考：

- DeepEval：`Contextual Precision / Recall / Relevancy`、`Faithfulness`、`Answer Relevancy`，见 https://www.confident-ai.com/docs/metrics/single-turn/contextual-precision-metric。
- Ragas：`Context Precision / Recall`、`ID Based Context Precision / Recall`、`Faithfulness`、`Answer Relevancy`、`Factual Correctness`、`Noise Sensitivity`，见 https://docs.ragas.io/en/stable/concepts/metrics/。
- TruLens：RAG Triad，即 `context relevance`、`groundedness`、`answer relevance`，见 https://www.trulens.org/getting_started/core_concepts/rag_triad/。

因此当前评估分两层：

1. **Retrieval-first 确定性评估**
   - 用于 A0 / A1 / B1。
   - 成本低、稳定、可重复。
   - 主要判断 chunking、retrieval_text、rerank、structure re-rank 是否改好。

2. **Small-sample judge 评估**
   - 用于 B1 成立后的少量 E2E 样本。
   - 对齐 DeepEval/Ragas/TruLens 的生成质量维度。
   - 不作为本轮大规模实验前置条件。

这样划分的原因是：RAG 能力不能被一个总分证明。检索实验回答“证据有没有找对、排前”；E2E judge 回答“最终答案有没有基于证据、有没有答到问题”。如果把两者混成一个分数，很难定位问题。

---

## 3. 前置条件

### 3.1 评估口径先行

旧 `rag_tuning.jsonl` 的 `gold_chunk_ids` 基于旧 chunk 语义。reindex 后不能直接作为唯一指标。

实验前必须先完成以下之一：

1. 将旧 `gold_chunk_ids` 映射到新 chunk id。
2. 暂时改用 `expected_root_ids / expected_anchors / expected_keywords`。

当前更推荐第 2 种，因为改动更小，也更适合结构化 chunk。

### 3.2 必须记录索引变体

A0 和 A1 不是同一个索引上的简单开关，因为 `retrieval_text` 会影响 ingest 阶段的 embedding / sparse。

每次 run 必须记录：

```json
{
  "index_variant": "raw_text | title_context",
  "structure_rerank_enabled": true,
  "confidence_gate_enabled": true,
  "auto_merge_enabled": false,
  "auto_merge_called": false
}
```

### 3.3 已补齐的最小开关

为了让实验可执行，当前已补 3 个内部开关：

| 开关 | 层级 | 说明 |
| --- | --- | --- |
| `EVAL_RETRIEVAL_TEXT_MODE=raw|title_context` | reindex | 控制 ingest 时使用 raw text 还是 title-injected retrieval text |
| `STRUCTURE_RERANK_ENABLED=true|false` | query | 控制是否启用 B1 structure re-rank |
| `CONFIDENCE_GATE_ENABLED=true|false` | query | 控制是否启用 fallback gate |

这些开关只用于评估和调试，不作为公开 API。

---

## 4. A 组：主链路最小消融

### 4.1 目的

验证两件事：

1. `retrieval_text` 是否有价值。
2. B1 structure re-rank 是否有独立收益。

### 4.2 实验组

| ID | 索引变体 | retrieval_text | structure re-rank | fallback gate | 说明 |
| --- | --- | --- | --- | --- | --- |
| A0 | `raw_text` | 关闭 | 关闭 | 关闭 | raw text baseline |
| A1 | `title_context` | 开启 | 关闭 | 关闭 | 只验证 title-injected retrieval text |
| B1 | `title_context` | 开启 | 开启 | 关闭 | 验证 structure re-rank |

### 4.3 执行方式

1. 设置 `EVAL_RETRIEVAL_TEXT_MODE=raw`。
2. reindex，运行 A0。
3. 设置 `EVAL_RETRIEVAL_TEXT_MODE=title_context`。
4. reindex，运行 A1。
5. 复用 A1 索引，开启 `STRUCTURE_RERANK_ENABLED=true`，运行 B1。

### 4.4 指标

#### 一级指标：确定性 retrieval 指标

这些指标直接决定 RAG 是否具备“可用检索能力”：

- `root_hit@5`
- `anchor_hit@5`
- `keyword_hit@5`
- `MRR`
- `context_precision_id@5`

说明：

- `context_precision_id@5` 借鉴 Ragas 的 ID-based context precision：在 top-k 中，命中 expected root / anchor / mapped chunk 的比例。
- `MRR` 用于判断正确证据是否排在前面。
- `anchor_hit@5` 对条款/编号类文档尤其重要。

#### 二级指标：噪声与稳定性指标

- `legacy_chunk_hit@5`
- `irrelevant_context_ratio@5`
- `dominant_root_share`
- `avg_latency_ms`
- `p95_latency_ms`
- `error_rate`

说明：

- `legacy_chunk_hit@5` 仅作参考，除非已经完成新旧 chunk id 映射。
- 对民法典类样本，`anchor_hit@5` 比旧 exact chunk hit 更可信。
- `irrelevant_context_ratio@5` 用 expected root / anchor / keyword 的未命中比例近似上下文噪声；后续可替换为 DeepEval/Ragas 的 contextual relevancy。

### 4.5 判定规则

#### A1 是否值得保留

A1 相比 A0 应至少满足以下两项：

- `root_hit@5` 提升。
- `anchor_hit@5` 提升。
- `keyword_hit@5` 提升。
- `MRR` 不下降。
- `irrelevant_context_ratio@5` 不上升。

并且：

- `avg_latency_ms` 增长不超过 15%。
- `error_rate = 0`。

#### B1 是否值得保留

B1 相比 A1 应至少满足以下两项：

- `root_hit@5` 提升。
- `anchor_hit@5` 提升。
- `MRR` 提升。
- 排名前 1 的命中率提升。
- `context_precision_id@5` 提升。

并且：

- `avg_latency_ms` 增长不超过 20%。
- `error_rate = 0`。

### 4.6 样本级比较

除了平均值，报告必须输出 paired comparison：

| 字段 | 含义 |
| --- | --- |
| `wins` | 新策略比旧策略更早命中或从未命中变为命中 |
| `losses` | 新策略比旧策略更晚命中或从命中变为未命中 |
| `ties` | 两者表现相同 |

判断 B1 是否成立时，不能只看平均值；必须同时看 `wins > losses`。

---

## 5. G 组：fallback gate 最小实验

### 5.1 前提

只有当 B1 成立后，才运行 G 组。

G 组复用 B1 索引，不再 reindex。

### 5.2 实验组

| ID | `CONFIDENCE_GATE_ENABLED` | `LOW_CONF_TOP_MARGIN` | `LOW_CONF_ROOT_SHARE` | `LOW_CONF_TOP_SCORE` | `ENABLE_ANCHOR_GATE` |
| --- | --- | ---: | ---: | ---: | --- |
| G0 | false | - | - | - | false |
| G1 | true | 0.05 | 0.45 | 0.20 | true |
| G2 | true | 0.03 | 0.35 | 0.15 | true |
| G3 | true | 0.08 | 0.55 | 0.25 | true |

说明：

- G1 是当前推荐阈值。
- G2 是更松阈值，用于观察是否减少不必要 fallback。
- G3 是更紧阈值，用于观察是否能抓住更多高风险 query，但也可能提高延迟和误触发。

### 5.3 人工样本

G 组需要 10 到 15 条人工判断样本。

字段：

```json
{
  "sample_id": "...",
  "query": "...",
  "should_fallback": true,
  "reason": "anchor mismatch | weak ranking | enough evidence"
}
```

### 5.4 指标

- `fallback_trigger_rate`
- `fallback_precision`
- `fallback_recall`
- `post_fallback_root_hit@5`
- `post_fallback_anchor_hit@5`
- `p95_latency_ms`

### 5.5 判定规则

G1 相比 G0：

- 应提升高风险样本的 post-fallback 命中。
- `p95_latency_ms` 必须明显低于全部走 `current_pipeline`。

G1 相比 G2：

- `fallback_precision` 更高。
- `fallback_recall` 不应明显下降。

G1 相比 G3：

- `fallback_trigger_rate` 更低。
- `fallback_precision` 不应明显低于 G3。
- 高风险样本的 `fallback_recall` 不应明显低于 G3。

如果 G1 触发率过高，应优先调松：

1. `LOW_CONF_TOP_MARGIN`
2. `LOW_CONF_ROOT_SHARE`
3. `ENABLE_ANCHOR_GATE`

---

## 6. 小样本 E2E judge 评估

A 组和 G 组通过后，再跑小样本 E2E judge。它不是 A/B 主判断前置条件，而是验证“检索变好是否真的让回答变好”。

### 6.1 样本

从派生数据集中选 10 到 15 条：

- 编号/条款类。
- 手册/说明书类。
- 一般结构化文档。

### 6.2 指标

| 指标 | 参考框架 | 含义 | 本项目用途 |
| --- | --- | --- | --- |
| `answer_relevancy` | DeepEval / Ragas / TruLens | 回答是否直接回应问题 | 防止答非所问 |
| `faithfulness / groundedness` | DeepEval / Ragas / TruLens | 回答声明是否被检索上下文支持 | 衡量幻觉风险 |
| `factual_correctness` | Ragas | 回答是否与 reference answer 对齐 | 检查最终答案准确性 |
| `context_relevance` | TruLens / DeepEval / Ragas | 检索上下文是否与 query 相关 | 检查无关上下文污染 |

### 6.3 最小 judge 记录格式

每条样本至少记录：

```json
{
  "sample_id": "...",
  "query": "...",
  "reference_answer": "...",
  "retrieved_contexts": ["..."],
  "answer": "...",
  "answer_relevancy": 0.0,
  "faithfulness": 0.0,
  "factual_correctness": 0.0,
  "context_relevance": 0.0,
  "judge_reason": "..."
}
```

分数使用 0 到 1。`judge_reason` 必须简短说明扣分原因，避免只留下不可解释的数字。

### 6.4 执行策略

当前阶段不强制引入 DeepEval、Ragas 或 TruLens 依赖。

推荐顺序：

1. 先用当前轻量规则评估 retrieval。
2. 在 10 到 15 条样本上用现有 LLM 做 judge。
3. 如果后续需要平台化，再选一个框架接入。

框架选择建议：

- 如果要 pytest/CI 风格单元评估，优先 DeepEval。
- 如果要 RAG 指标体系和 ID-based context precision，优先 Ragas。
- 如果要 tracing、RAG Triad、线上观测，优先 TruLens。

---

## 7. 最小输出

每次 run 至少输出：

```text
.jbeval/reports/<run_id>/
```

必须包含：

- `summary.json`
- `summary.md`
- `results.jsonl`
- `config.json`

`config.json` 必须记录：

- 代码版本或 git diff 标识。
- index variant。
- reindex 时间。
- 关键环境变量。
- 是否启用 structure re-rank。
- 是否启用 confidence gate。
- 是否启用 auto_merge。
- 是否实际调用 auto_merge。
- paired comparison 的 wins/losses/ties。
- 如果跑了 judge，小样本 judge 指标。

---

## 8. 执行顺序

固定顺序：

1. 派生新评估口径。
2. 确认最小实验开关配置。
3. A0 reindex + run。
4. A1 reindex + run。
5. B1 run。
6. 判断主链路是否成立。
7. 若 B1 成立，运行 G0 / G1 / G2 / G3。
8. 固定默认门控参数。
9. 跑 10 到 15 条小样本 E2E judge。

---

## 9. 本轮不做

- `B1-lite`
- `B1-heavy`
- 大规模人工标注集
- Prompt / 模型层实验
- 线上 A/B
- 完整诊断平台

这些都属于下一阶段。
