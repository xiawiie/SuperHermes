# SuperHermes RAG 诊断规范（V1 最小可用版）

## 1. 目的

这份规范只解决当前阶段最重要的问题：

> 当 RAG 结果不对时，快速判断问题更像出在召回、排序，还是低置信度门控。

本阶段不自动诊断 Prompt 和模型能力问题，因为当前系统还没有稳定记录最终 prompt、最终 context 和引用证据链。强行自动归因会制造不可靠结论。

诊断框架参考 TruLens 的 RAG Triad：

- context relevance：检索上下文是否相关。
- groundedness：回答是否被上下文支撑。
- answer relevance：回答是否回应问题。

当前 V1 先落地第一层，也就是 context relevance / retrieval quality；groundedness 和 answer relevance 放到小样本 E2E judge。

这个边界很重要：V1 诊断只能证明“检索证据链是否足够好”，不能单独证明最终 RAG 回答质量。最终回答质量必须结合 small-sample judge 的 faithfulness / groundedness、answer relevance、factual correctness。

---

## 2. V1 诊断范围

V1 覆盖：

- `recall_miss`
- `ranking_miss`
- `low_confidence`
- `mixed_failure`
- `ok`
- `insufficient_trace`

V1 不覆盖：

- `prompt_issue`
- `model_limit`

V1 只在小样本 judge 报告中观察：

- `answer_relevancy`
- `faithfulness / groundedness`
- `factual_correctness`

---

## 3. 统一输出：`DiagnosticResult`

```python
DiagnosticResult = {
    "category": "ok | recall_miss | ranking_miss | low_confidence | mixed_failure | insufficient_trace",
    "failed_stage": "none | recall | rerank | structure_rerank | confidence_gate | unknown",
    "evidence": dict,
    "suggestions": list[str],
}
```

要求：

- `evidence` 必须是结构化字段。
- `suggestions` 必须是可执行动作。
- trace 不足时必须返回 `insufficient_trace`，不要伪装成精确归因。

---

## 4. 诊断输入

建议函数签名：

```python
def classify_failure(
    query: str,
    rag_trace: dict,
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
) -> DiagnosticResult:
    ...
```

### 4.1 离线评估模式

离线评估应优先提供：

- `expected_root_ids`
- `expected_anchors`
- `expected_keywords`

`expected_chunk_ids` 只有在完成新旧 chunk id 映射后才作为强指标。

### 4.2 线上运行模式

线上没有标准答案时，只能输出候选归因。

线上模式可以可靠识别：

- `low_confidence`
- `insufficient_trace`

线上模式不能仅靠 trace 断言：

- 真实召回失败
- 真实排序失败

---

## 5. Trace 字段要求

### 5.1 当前已可使用字段

当前系统已有或应保持稳定的字段：

- `retrieved_chunks`
- `top_score`
- `top_margin`
- `dominant_root_share`
- `dominant_root_support`
- `fallback_required`
- `confidence_reasons`
- `query_anchors`
- `anchor_match`
- `retrieval_stage`
- `rewrite_strategy`

这些字段足够支持：

- low confidence 判断
- 最终结果是否命中 root / anchor / keyword
- 基础 case 复核

### 5.2 要严格区分召回和排序，必须补的字段

如果要把 `recall_miss` 和 `ranking_miss` 区分清楚，还必须补候选阶段 trace：

- `candidates_before_rerank`
- `candidates_after_rerank`
- `candidates_after_structure_rerank`

每个候选只需要记录轻量字段：

```json
{
  "chunk_id": "...",
  "root_chunk_id": "...",
  "anchor_id": "...",
  "text_preview": "...",
  "score": 0.0,
  "rerank_score": 0.0,
  "final_score": 0.0
}
```

不要在 trace 中默认写完整正文，避免日志过大。

---

## 6. 分类规则

### 6.1 `insufficient_trace`

满足以下任一条件：

- `rag_trace` 为空。
- 缺少 `retrieved_chunks`。
- 离线模式要区分召回/排序，但缺少候选阶段 trace。

建议动作：

- 补齐 `rag_trace` 字段。
- 在评估模式下打开候选阶段 trace。

### 6.2 `low_confidence`

满足以下任一条件：

- `fallback_required = true`
- `confidence_reasons` 非空

建议动作：

- 检查 `top_margin`。
- 检查 `dominant_root_share`。
- 检查 `anchor_match`。
- 判断门控是否过松或过紧。

### 6.3 `recall_miss`

离线模式下，如果候选阶段 trace 存在，并且以下任一成立：

- `expected_root_ids` 没出现在 `candidates_before_rerank`。
- `expected_anchors` 没出现在 `candidates_before_rerank`。
- 已映射的新 `expected_chunk_ids` 没出现在 `candidates_before_rerank`。

则判定为：

- `category = recall_miss`
- `failed_stage = recall`

建议动作：

- 调整 chunking 边界。
- 优化 `retrieval_text`。
- 调大 dense/sparse candidate 范围。
- 检查 embedding / sparse 信号。

### 6.4 `ranking_miss`

离线模式下，如果 expected root / anchor / chunk 已出现在 `candidates_before_rerank`，但没有进入最终 `retrieved_chunks`，则判定为排序问题。

细分：

- 出现在 `candidates_before_rerank`，但未出现在 `candidates_after_rerank`：
  - `failed_stage = rerank`
- 出现在 `candidates_after_rerank`，但未出现在 `candidates_after_structure_rerank` 或最终 top-k：
  - `failed_stage = structure_rerank`

建议动作：

- 调整 `STRUCTURE_RERANK_ROOT_WEIGHT`。
- 调整 `SAME_ROOT_CAP`。
- 调整 rerank candidate 范围。
- 检查标题注入是否产生噪声。

### 6.5 `mixed_failure`

同时存在召回和排序问题时返回。

例如：

- expected anchor 未被召回。
- 同时 top-k 也被错误 root 聚集。

建议动作：

- 先修召回，再修排序。

### 6.6 `ok`

离线模式下满足以下任一条件：

- 最终 `retrieved_chunks` 命中 expected root。
- 最终 `retrieved_chunks` 命中 expected anchor。
- 最终 `retrieved_chunks` 覆盖 expected keywords。
- 已完成 chunk id 映射时，最终命中 expected chunk。

线上模式下只有人工确认正确时才返回 `ok`。

---

## 7. 最小伪代码

```python
def classify_failure(query, rag_trace, expected_chunk_ids=None,
                     expected_root_ids=None, expected_anchors=None,
                     expected_keywords=None):
    if not rag_trace or "retrieved_chunks" not in rag_trace:
        return insufficient_trace_result("missing_rag_trace")

    if rag_trace.get("fallback_required") or rag_trace.get("confidence_reasons"):
        return low_confidence_result(rag_trace)

    final_hits = match_expected(
        rag_trace.get("retrieved_chunks", []),
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
        expected_anchors=expected_anchors,
        expected_keywords=expected_keywords,
    )
    if final_hits:
        return ok_result(final_hits)

    needs_offline_diagnosis = expected_chunk_ids or expected_root_ids or expected_anchors or expected_keywords
    if not needs_offline_diagnosis:
        return insufficient_trace_result("missing_ground_truth")

    before = rag_trace.get("candidates_before_rerank")
    after_rerank = rag_trace.get("candidates_after_rerank")
    after_structure = rag_trace.get("candidates_after_structure_rerank")
    if before is None or after_rerank is None or after_structure is None:
        return insufficient_trace_result("missing_candidate_stage_trace")

    if not match_expected(before, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords):
        return recall_miss_result(...)

    if not match_expected(after_rerank, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords):
        return ranking_miss_result(stage="rerank")

    if not match_expected(after_structure, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords):
        return ranking_miss_result(stage="structure_rerank")

    return mixed_failure_result(...)
```

---

## 8. 建议映射表

| category | 主要含义 | 优先动作 |
| --- | --- | --- |
| `recall_miss` | 正确 root / anchor / chunk 没进入候选 | 调 chunking、`retrieval_text`、candidate 范围 |
| `ranking_miss` | 召回到了，但排序或结构重排没保住 | 调 rerank 范围、root weight、same root cap |
| `low_confidence` | 快路径自己判断不稳 | 调 margin、root share、anchor gate |
| `mixed_failure` | 多层问题叠加 | 先修召回，再修排序 |
| `insufficient_trace` | 当前证据不足以归因 | 补 trace 或补 ground truth |

---

## 9. 与通用 RAG 评估指标的映射

| 通用指标 | DeepEval / Ragas / TruLens 对应概念 | 当前 V1 对应字段 |
| --- | --- | --- |
| Context Precision | Contextual Precision / Context Precision | `context_precision_id@5`、`MRR` |
| Context Recall | Contextual Recall / Context Recall | `root_hit@5`、`anchor_hit@5`、`keyword_hit@5` |
| Context Relevance | Contextual Relevancy / Context Relevance | `irrelevant_context_ratio@5`、后续 LLM judge |
| Groundedness | Faithfulness / Groundedness | 后续小样本 judge |
| Answer Relevance | Answer Relevancy / Answer Relevance | 后续小样本 judge |
| Factual Correctness | Factual Correctness | 后续小样本 judge |

V1 的诊断结果应进入 `results.jsonl`，用于解释 A0 / A1 / B1 / G0 / G1 / G2 / G3 的失败样本，而不是替代这些实验指标。

---

## 10. 当前已落地

当前阶段已经落地：

1. 定义 `DiagnosticResult`。
2. 实现 `classify_failure()` 最小版本。
3. 在评估模式下补候选阶段 trace。

仍待评估脚本消费：

4. 在评估结果中输出诊断分类。
5. 在评估结果中输出 `context_precision_id@5`、`irrelevant_context_ratio@5`、paired comparison。

---

## 11. 后续再做

- `/diagnose` HTTP 接口。
- 前端诊断页。
- Prompt / 模型层自动归因。
- 自动聚合报告。
- 大规模 case 标注系统。
