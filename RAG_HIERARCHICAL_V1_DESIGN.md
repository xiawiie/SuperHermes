# SuperHermes RAG Hierarchical V1 项目计划

## 1. 结论

基于当前代码状态和已有 `rag_tuning.jsonl` 评估结果，Hierarchical v1 的总体方向是合理的，但计划需要进一步收敛。

已有评估暴露出的核心事实是：

- `hybrid_rerank` 延迟低，平均约 `635ms`，但 `hit@5 = 0.22`。
- `current_pipeline` 质量更高，`hit@5 = 0.32`、`article_hit@5 = 0.62`，但平均延迟约 `51s`。
- `auto_merge` 旧策略没有稳定提升，`hybrid_rerank_auto_merge` 反而低于 `hybrid_rerank`。

因此，当前最合理的路线不是继续默认走慢速 `current_pipeline`，也不是继续强化旧 `auto_merge`，而是：

> 用结构化 chunk、`retrieval_text`、structure re-rank 提升默认快路径质量；只在低置信度时进入慢回退。

这条路线符合当前项目目标：先满足 RAG 的有效使用，而不是建设完整评估平台或复杂诊断系统。

---

## 2. 当前计划是否合理

### 2.1 合理的部分

当前计划中以下设计应保留：

1. `text` 与 `retrieval_text` 分离
   - `text` 保留原文，用于展示和回答上下文。
   - `retrieval_text` 注入有效标题上下文，用于 embedding、sparse、rerank。

2. `generic / structured` 双 profile
   - 不为法律、手册、制度等文档各自建立独立 RAG 管线。
   - 只在 ingest 侧做轻量结构识别。

3. B1 structure re-rank
   - 继续使用：

   ```text
   root_score = max(leaf_rerank_score in same root)
   final_score = 0.7 * leaf_score + 0.3 * root_score
   same_root_cap = 2
   ```

   - 这比旧 `auto_merge` 更可控，也更贴合“正确结构附近已被命中，但 exact leaf 不稳定”的问题。

4. 低置信度 fallback gate
   - 慢回退只处理高风险 query。
   - 不再把 LangGraph rewrite / expanded retrieval 当成默认热路径。

5. 最小实验矩阵和最小诊断规范
   - 当前阶段只验证主链路是否更好。
   - 诊断先覆盖召回、排序、低置信度，不急着自动归因 Prompt / 模型问题。

### 2.2 需要修正的部分

当前计划中仍有 5 个需要修正的点：

1. 旧 `gold_chunk_ids` 不能直接用于新索引
   - reindex 后 chunk id 与旧评估集不再一致。
   - 实验前必须先采用 `root / anchor / expected_keywords` 口径，或做新旧 chunk id 映射。

2. A0 / A1 / B1 不是同一个索引上的简单开关
   - A0 如果要关闭 `retrieval_text`，必须用 raw text 重建索引。
   - A1 / B1 使用标题注入后的 `retrieval_text` 索引。
   - 因此实验报告必须记录 `index_variant`。

3. 当前代码已补最小实验切换开关
   - 已支持：
     - `EVAL_RETRIEVAL_TEXT_MODE=raw|title_context`
     - `STRUCTURE_RERANK_ENABLED=true|false`
     - `CONFIDENCE_GATE_ENABLED=true|false`

4. `auto_merge` 文档定位要和代码现实一致
   - 代码里 `_auto_merge_documents()` 和 `AUTO_MERGE_ENABLED` 仍存在。
   - 但当前成功检索路径没有调用 `_auto_merge_documents()`。
   - 因此它应被定义为残留兼容逻辑，而不是 v1 默认策略。
   - 下一步应清理它，或在评估报告中显式记录其未启用。

5. 诊断要区分“当前可做”和“补 trace 后可做”
   - 当前 `rag_trace` 能判断低置信度和最终命中情况。
   - 当前代码已补候选阶段 trace：
     - `candidates_before_rerank`
     - `candidates_after_rerank`
     - `candidates_after_structure_rerank`
   - 后续评估脚本应消费这些字段并输出诊断分类。

---

## 3. 优化后的项目目标

本轮目标调整为 3 个可验收目标：

1. 建立一个可运行的默认快路径
   - `retrieval_text + rerank + structure re-rank + confidence gate`
   - 高置信度 query 不进入 LLM grader / rewrite。

2. 建立一个可执行的最小评估闭环
   - 能比较 A0 / A1 / B1。
   - 能比较 G0 / G1 / G2 / G3。
   - 能输出 `summary.json`、`summary.md`、`results.jsonl`。

3. 建立一个最小故障诊断闭环
   - 能将失败 case 归为：
     - `recall_miss`
     - `ranking_miss`
     - `low_confidence`
     - `mixed_failure`
   - 在 trace 不足时明确返回 `insufficient_trace`，不伪装成准确归因。

4. 引入更科学的 E2E 小样本评估
   - 参考 DeepEval / Ragas / TruLens 的通用指标体系。
   - 大规模实验仍以确定性 retrieval 指标为主。
   - 小样本再评估 answer relevancy、faithfulness/groundedness、factual correctness。
   - 不用单一总分判断 RAG 好坏，而是分别判断检索、上下文质量、生成忠实性和答案相关性。

---

## 4. 优化后的实施计划

### Phase 0：对齐评估口径

这是后续实验的前置条件。

必须完成：

- 给 `rag_tuning.jsonl` 增加或派生以下字段：
  - `expected_root_ids`
  - `expected_anchors`
  - `expected_keywords`
- 暂时不要把旧 `gold_chunk_ids` 当成唯一硬指标。
- 评估报告中同时保留：
  - `legacy_chunk_hit@5`，仅作参考
  - `root_hit@5`
  - `anchor_hit@5`
  - `keyword_hit@5`

通过条件：

- 每条样本至少有一种新口径可评估。
- 民法典类样本必须能抽出 `expected_anchor`。

### Phase 1：最小实验开关（已补齐）

当前已补 3 个开关，避免实验只能停留在文档里。

| 开关 | 用途 |
| --- | --- |
| `EVAL_RETRIEVAL_TEXT_MODE=raw|title_context` | 控制 ingest 时 embedding / sparse 使用 raw text 还是 title-injected retrieval text |
| `STRUCTURE_RERANK_ENABLED=true|false` | 控制是否应用 B1 |
| `CONFIDENCE_GATE_ENABLED=true|false` | 控制是否触发 fallback |

说明：

- `EVAL_RETRIEVAL_TEXT_MODE` 是 reindex 级开关。
- `STRUCTURE_RERANK_ENABLED` 和 `CONFIDENCE_GATE_ENABLED` 是 query 级开关。
- 这些开关是实验支撑，不应暴露成公开 API。

### Phase 2：运行最小主链路实验

执行顺序：

1. A0：raw text index，关闭 structure re-rank，关闭 fallback。
2. A1：title_context index，关闭 structure re-rank，关闭 fallback。
3. B1：复用 A1 index，开启 structure re-rank，关闭 fallback。

判断方式：

- A1 相比 A0：证明 `retrieval_text` 是否值得。
- B1 相比 A1：证明 structure re-rank 是否值得。

主要指标：

- `root_hit@5`
- `anchor_hit@5`
- `keyword_hit@5`
- `context_precision_id@5`
- `MRR`
- `irrelevant_context_ratio@5`
- `avg_latency_ms`
- `error_rate`

通过条件：

- A1 或 B1 至少在 `root_hit@5 / anchor_hit@5 / MRR` 中两项提升。
- B1 延迟相对 A1 增长不超过 20%。
- `error_rate = 0`。

### Phase 3：运行最小门控实验

在 B1 成立后运行。

实验组：

- G0：关闭 confidence gate。
- G1：当前推荐阈值。
- G2：更松阈值。
- G3：更紧阈值。

判断方式：

- G1 相比 G0：证明 fallback gate 是否有必要。
- G1 相比 G2：证明当前阈值是否比更松策略更稳。
- G1 相比 G3：证明当前阈值是否比更紧策略更克制。

主要指标：

- `fallback_trigger_rate`
- `fallback_precision`
- `fallback_recall`
- `post_fallback_root_hit@5`
- `p95_latency_ms`

通过条件：

- G1 能抓住明显高风险 query。
- G1 的 `p95_latency_ms` 明显低于全部走 `current_pipeline`。
- G1 不应让 fallback 触发率失控。
- G1 相比 G2 不应明显漏掉高风险样本。
- G1 相比 G3 不应明显降低 fallback precision。

### Phase 4：补最小诊断函数

当前已实现：

```python
DiagnosticResult
classify_failure(...)
```

当前优先支持离线评估模式。

必须能输出：

- 失败分类
- 失败阶段
- 结构化证据
- 可执行建议

不在本阶段自动诊断：

- Prompt 问题
- 模型能力问题

这两类等上下文与 prompt 证据链完整后再做。

### Phase 5：小样本 E2E judge

在 A0 / A1 / B1 和 G0 / G1 / G2 / G3 通过后，再选 10 到 15 条样本做 E2E judge。

评估维度：

- `answer_relevancy`
- `faithfulness / groundedness`
- `factual_correctness`
- `context_relevance`

最小输出：

- 每条样本记录 `retrieved_contexts`、最终 `answer`、`reference_answer`。
- 每个 judge 分数必须给出简短原因。
- 报告按文档类型聚合，不只看总平均值。

说明：

- 当前阶段不强制接入 DeepEval、Ragas 或 TruLens。
- 先复用现有 LLM 做轻量 judge。
- 后续如果要框架化，再按目标选择：
  - CI/pytest 风格：DeepEval。
  - RAG 指标与 ID-based context precision：Ragas。
  - tracing 与 RAG Triad：TruLens。

---

## 5. 当前文档分工

本项目计划只保留高层决策和阶段计划。

具体执行细节放在两份配套文档：

- [RAG_HIERARCHICAL_V1_EXPERIMENT_MATRIX.md](C:/Users/goahe/Desktop/Project/SuperHermes/RAG_HIERARCHICAL_V1_EXPERIMENT_MATRIX.md)
  - 负责实验分组、指标、运行顺序、判定规则。

- [RAG_DIAGNOSTIC_SPEC.md](C:/Users/goahe/Desktop/Project/SuperHermes/RAG_DIAGNOSTIC_SPEC.md)
  - 负责诊断输入、输出、分类规则、trace 要求。

---

## 6. 不纳入当前阶段的内容

以下内容现在不做：

- 新增第三种 profile。
- 完整前端诊断页。
- 线上 A/B。
- Prompt / 模型层自动归因。
- 大规模人工标注平台。
- 默认引入更强或新的外部模型服务。
- 正式 staging/swap 索引切换系统。

这些都可以后续做，但不应该阻塞当前“先让 RAG 稳定有效使用”的目标。

---

## 7. 最终判断

当前计划的方向是合理的，但必须按上述方式收敛后执行。

最关键的调整是：

1. 先修评估口径，再谈指标提升。
2. 承认 A0/A1 是不同索引变体，不要误写成同索引开关。
3. 补实验开关，让矩阵能真正运行。
4. 把诊断范围限定在当前 trace 能支撑的层级。
5. 把旧 `auto_merge` 正式标记为残留兼容逻辑，并避免在评估口径中混入它。

按这个计划推进，下一步最有价值的工程动作是：

1. 基于派生后的 `.jbeval/datasets/rag_tuning_derived.jsonl` 跑 A0 / A1 / B1。
2. 在评估报告中消费候选阶段 trace 和 `classify_failure()`。
3. 根据 A 组结果再决定是否进入 G0 / G1 / G2 / G3。
