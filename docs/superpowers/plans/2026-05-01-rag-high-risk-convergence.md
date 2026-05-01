# RAG High-Risk Convergence Plan

> 状态感知修订版，基于 `codex/rag-convergence-20260501` 当前实现。本文用于指导后续收敛，不要求一次性全部实现。默认策略是：先冻结契约和可观测性，再通过 feature flag 小步改变行为。

## 结论

原方案的方向是对的：它抓住了四个真正会拖垮 RAG 系统可维护性和质量稳定性的点。

- 两套路由式 rerank 必须收敛，否则每次调权都会产生双倍维护成本。
- fallback 二次完整检索必须预算化，否则低置信度查询会在最需要快的时候变得最慢。
- Agent 路径和直接路径必须统一上下文契约，否则用户不知道为什么“有附件”和“无附件”答案质量不同。
- 错误必须结构化进入 trace，否则系统会继续出现“结果可用但不知道降级了”的隐性失败。

但原方案也有三个需要修正的地方：

- 它把一些当前分支已经完成的工作仍写成待办，例如 `retrieve_documents()` 拆分、runtime config、trace builder、layered 收为候选策略、共享 RAG 文本格式。
- 它新增模块偏多，容易在第一轮之后继续扩大范围。现在应优先复用已有模块：`runtime_config.py`、`trace.py`、`types.py`、`formatting.py`、`utils.py`。
- 它没有把 Deep Mode 纳入严格边界。Deep 不能只是“更多检索 + 最终综合”，必须先有 evidence pack、citation verifier、降级语义和质量闸门。

因此修订后的执行策略是：

1. 保持当前第一轮成果，不回滚。
2. 第二轮先做 citation verifier 和 fallback candidate-only，避免盲目启用 Deep。
3. Deep Mode 只作为 gated orchestrator 引入，不默认改变回答路径。
4. Agent/直接路径先统一执行契约，再决定是否统一策略。

## 当前状态

### 已完成

- `backend/rag/runtime_config.py` 已集中读取 RAG runtime 配置，支持 fake env 测试。
- `backend/rag/utils.py` 的 `retrieve_documents()` 已从巨型函数收敛为门面和阶段函数。
- `backend/rag/layered_rerank.py` 已改为接收显式 `LayeredRerankConfig`，不再在核心逻辑 import-time 绑定环境变量。
- layered 路径不再拥有独立 CrossEncoder 后处理，最终 rerank 统一经过 `_rerank_documents()`。
- `backend/rag/trace.py` 已提供 `build_retrieval_meta()`、`build_initial_rag_trace()`、`merge_expanded_rag_trace()`、`append_stage_error()`。
- `backend/rag/formatting.py` 已统一 RAG chunk 文本格式。
- Agent 工具路径和附件直接路径已统一记录 `context_delivery_mode`。
- `StructuredTool.invoke()` 下 `rag_trace` 和单轮工具调用计数已通过 mutable holder 修复。

### 部分完成

- 错误处理：检索层已覆盖 embedding、hybrid、scoped/global、layered、rerank 等关键失败路径，但 Agent、LLM synthesis、Deep、citation 还没有统一错误矩阵。
- Agent/直接路径：上下文格式和 trace 边界已统一，但执行策略仍不同。
- fallback latency：已有 deadline、并行 executor、complex 双路并行，但 expanded retrieval 仍是完整 `retrieve_documents()`。

### 未完成

- citation verifier 运行时化。
- Deep Mode active synthesis。
- fallback 第二阶段 candidate-only。
- Agent/直接路径统一执行策略。
- layered 文件和命名进一步清理，避免“layered rerank”继续误导。

## 原方案评价

### 优势

原方案最大的优点是方向正确，且符合“先收敛，再优化”的工程顺序。它没有一上来删掉 layered，也没有贸然启用 Deep，而是先把公共入口、错误语义、postprocess、fallback budget 和 EvidenceBundle 列为中间层。

它还正确识别了两个关键事实：

- layered 的核心价值不是另一套 rerank，而是候选召回和候选压缩。
- fallback 的问题不是“第二次检索一定错”，而是第二次检索必须受预算、超时和收益约束。

### 劣势

原方案的问题是实施颗粒度偏大，尤其是 `contracts.py`、`postprocess.py`、`errors.py`、`backend/chat/evidence.py` 同时出现，会让第二轮变成又一次大搬家。当前分支已经有 `types.py`、`trace.py`、`formatting.py` 和阶段化 `utils.py`，继续新增太多平行模块会让抽象反而分散。

另一个问题是 Deep 缺席。高风险 RAG 收敛如果不规定 Deep 的边界，后面很容易把 Deep 加成一个独立黑盒：自己拆 query、自己检索、自己综合、自己返回答案。那会重新制造一套和标准 RAG 并行的路径，等于把现在正在解决的问题复制到 Deep 里。

### 需要修改的判断

原方案中的 `EvidenceBundle` 是对的，但位置应调整。它不应该先放在 `backend/chat/evidence.py`，因为 evidence 是 RAG 产物，不只是 chat 层格式。建议放在 `backend/rag/evidence.py` 或 `backend/rag/citations.py`，chat 层只负责 delivery。

原方案中的 `StageError` 扩展也是对的，但不一定要立即迁移到 `backend/rag/errors.py`。当前已有 `backend/rag/types.py` 和 `backend/rag/trace.py`，短期内先在这两个文件里增强字段，避免多文件迁移。

原方案中的 `postprocess.py` 可以推迟。当前 `_finish_retrieval_pipeline()` 已经统一后处理，先让它稳定；当 `utils.py` 第二轮继续瘦身时再迁移。

## 修订原则

### 原则 1：默认行为不变

任何会改变用户答案路径的能力都必须 feature flag 控制。包括：

- `RAG_FALLBACK_CANDIDATE_ONLY`
- `RAG_UNIFIED_EXECUTION_ENABLED`
- `RAG_DEEP_SHADOW`
- `RAG_DEEP_ACTIVE`
- `RAG_CITATION_VERIFY_ENABLED`

原因：RAG 的质量退化常常不是测试立刻能看出来的。默认行为稳定比“理论上更先进”更重要。

优势：可回滚、可对照、便于逐步评估。

劣势：短期配置项会增加，需要清楚标记实验开关和默认值。

### 原则 2：Deep 不拥有独立检索栈

Deep Mode 只能调用标准 RAG 的候选召回、postprocess、trace、citation verifier，不能复制一套新的 retrieval/rerank/confidence。

原因：否则 Deep 会成为第三套并行路径，维护成本和行为差异都会继续扩大。

优势：Deep 质量问题能落回同一套检索 trace 调试。

劣势：Deep 早期能力会受标准 RAG 边界约束，不能快速堆复杂功能。

### 原则 3：citation verifier 先于 Deep active

Deep 主动综合必须先过 citation verifier。没有 verifier 时，Deep 最多 shadow 运行，不直接返回答案。

原因：Deep 的最大风险不是召回少，而是“综合得很像对的，但引用和证据不受约束”。

优势：减少幻觉答案进入主路径。

劣势：第一版 verifier 只能做确定性引用校验，不能完全证明 claim faithfulness。

### 原则 4：性能优化先减少 CrossEncoder 次数

fallback 的主要延迟瓶颈通常是 rerank，而不是重新 embedding 本身。第二次 query 不同，不能强行复用第一轮 embedding。优化应优先避免第二轮完整 rerank。

优势：收益明确，能用单测证明 `_rerank_documents()` 调用次数减少。

劣势：candidate-only 合并可能影响排序，需要 eval 对照。

## 修订后阶段计划

## Phase 1：继续清理 layered rerank 命名和残留

### 目标

把 layered 明确降级为 candidate strategy，避免未来开发者误以为它仍是一套后处理 rerank。

### 修改

- 保留 `backend/rag/layered_rerank.py`，但文档和函数命名改成 candidate prefilter 语义。
- 清理或标注旧 L2/L3 常量，确认它们只影响候选数量和候选保护，不影响最终 postprocess。
- 在 trace 中使用 `candidate_strategy=layered_split`，而不是暗示 `rerank_strategy=layered`。
- 保留 `LAYERED_RERANK_ENABLED` 兼容，但在 runtime config 中标记为 `layered_candidate_enabled`。

### 为什么这样做

删除 layered 会丢掉已有实验能力；保留原名又会继续误导。最稳妥的是“保留实现，改清边界”。

### 优势

- 维护者只需要改一套 rerank。
- layered 仍可作为候选压缩策略评测。
- 后续 fallback candidate-only 可以复用 layered 的候选策略。

### 劣势

- 文件名 `layered_rerank.py` 短期仍不准确。
- 完全改名会影响 import 和历史上下文，建议单独小提交处理。

### 验收

- `LAYERED_RERANK_ENABLED=true/false` 都经过同一个 `_finish_retrieval_pipeline()`。
- layered 失败 fallback 到 standard hybrid 时保留 scoped mode。
- 测试覆盖 sparse 缺失、split 失败、L1 candidate drop trace。

## Phase 2：运行时 citation verifier

### 目标

把离线 answer eval 中的 citation 思路变成运行时可复用能力，但第一版只做确定性校验。

### 新增模块

`backend/rag/citations.py`

核心结构：

```python
CitationRef:
    ref_id: str
    chunk_id: str | None
    filename: str
    page_number: int | str | None
    section_path: str | None

CitationVerifierResult:
    valid: bool
    cited_refs: list[str]
    unknown_refs: list[str]
    missing_required_refs: list[str]
    citation_error: str | None
```

核心函数：

```python
build_citation_refs(docs) -> list[CitationRef]
format_evidence_with_refs(docs) -> str
verify_citations(answer, refs) -> CitationVerifierResult
```

### 为什么这样做

Deep 和普通 RAG 都需要知道答案是否引用了真实证据。先做确定性 ref 校验，可以避免引入新 LLM judge，也不会增加依赖。

### 优势

- 可单测，稳定。
- 不依赖模型质量。
- 可同时服务 Deep、Agent 工具、评测脚本。
- 为后续 claim-level verifier 留接口。

### 劣势

- 只能证明引用存在，不能证明每个 claim 都被证据支持。
- 需要推动 prompt 使用 `[C1]` 这类稳定 ref，否则 verifier 只能做弱检查。

### 验收

- 答案引用 `[C1]` 时必须能映射到真实 chunk。
- 引用 `[C99]` 必须标记 unknown。
- filename/page 引用不匹配时必须进入 `citation_verifier` trace。
- verifier 不通过时默认不改用户答案，只写 trace；Deep active 时才触发 retry 或降级。

## Phase 3：fallback candidate-only

### 目标

解决 fallback 触发后的两次完整检索延迟。第二轮 expanded retrieval 默认不再完整 rerank，而是只召回候选，合并后统一 rerank 一次。

### 当前问题

当前 `retrieve_expanded()` 仍会调用完整 `retrieve_documents()`。这意味着：

```text
initial: embedding -> Milvus -> rerank -> confidence
expanded: embedding -> Milvus -> rerank -> confidence
```

最坏情况下 CrossEncoder 执行两次。

### 新增能力

在 `backend/rag/utils.py` 或后续 `backend/rag/retrieval_session.py` 中增加内部函数：

```python
retrieve_candidate_pool(query, context_files, budget, session) -> CandidateRetrievalResult
finish_retrieval_pipeline(query, candidates, ...)
```

fallback 新路径：

```text
initial full retrieval
if fallback required:
  rewrite query
  expanded candidate retrieval only
  merge initial docs/candidates + expanded candidates
  final rerank once
```

### 为什么这样做

expanded query 的 embedding 仍然需要重新算，但第二轮不一定需要先做一次完整 CE rerank。真正需要的是 expanded candidates 和 initial evidence 一起竞争最终 top_k。

### 优势

- 直接减少 CrossEncoder 调用次数。
- 保留第二轮 query 带来的召回增益。
- trace 能明确展示节省了哪一步。

### 劣势

- 合并候选后排序分布可能变化。
- candidate pool 太大时最终 rerank 仍可能慢。
- 需要 careful budget，否则只是把慢点换到最终 rerank。

### 风险控制

- 默认关闭：`RAG_FALLBACK_CANDIDATE_ONLY=false`。
- deadline 不足时直接返回 initial。
- 对 expanded candidate 设置独立上限：`RAG_FALLBACK_EXPANDED_CANDIDATE_K`。
- trace 记录：
  - `fallback_second_pass_mode`
  - `expanded_candidate_count`
  - `final_rerank_input_count`
  - `fallback_saved_rerank`
  - `expanded_retrieval_skipped_reason`

### 验收

- 单测证明 candidate-only 模式下 fallback 不会调用 `_rerank_documents()` 两次。
- complex 策略仍并行 HyDE 和 step-back candidate retrieval。
- K2/default smoke 不退化。
- fallback eval 至少报告 latency、hit、MRR、fallback_helped、fallback_hurt。

## Phase 4：Agent 和直接路径统一执行契约

### 目标

让有附件和无附件路径都经过同一个 RAG turn preparation 层，但不立即强行改变默认行为。

### 新增模块

`backend/chat/rag_execution.py`

核心结构：

```python
RagExecutionPolicy:
    NO_RAG
    OPTIONAL_TOOL
    FORCED_PRELOAD

RagTurnRequest:
    user_text
    context_files
    stream

RagTurnContext:
    policy
    docs
    context
    rag_trace
    delivery_mode
```

### 策略

默认行为保持：

- 有 `context_files`：`FORCED_PRELOAD`，通过 `system_message` 注入。
- 无 `context_files`：`OPTIONAL_TOOL`，Agent 自主调用。

实验行为：

- `RAG_UNIFIED_EXECUTION_ENABLED=true` 时，对明显文档问题先走轻量 intent 判断。
- 判定为文档问题：`FORCED_PRELOAD`。
- 普通聊天、写作、翻译、代码解释：`NO_RAG` 或 `OPTIONAL_TOOL`。

### 为什么这样做

如果直接把所有无附件请求都改成预检索，会导致过检索、延迟上升和回答变僵。先统一契约，再逐步统一策略，风险小很多。

### 优势

- trace 统一。
- streaming 和 non-streaming 逻辑可复用。
- 后续产品层可以清楚解释“为什么这次检索了”。

### 劣势

- 默认行为短期仍有差异。
- intent 判断可能误判，需要观测和白名单。

### 验收

- 有附件 non-stream、附件 stream、无附件 tool、无附件不检索四类路径都有测试。
- 每条路径都写入：
  - `retrieval_policy`
  - `context_delivery_mode`
  - `context_format_version`
  - `rag_trace`
- `context_files` 仍不可逃逸。

## Phase 5：统一错误语义

### 目标

把错误从“异常字符串”提升为“产品可解释状态”。

### 扩展 StageError

当前 `StageError` 可以继续保留兼容字段：

```python
stage: str
error: str
fallback_to: str | None
```

新增可选字段：

```python
error_code: str
severity: "info" | "warning" | "error"
recoverable: bool
user_visible: bool
```

### 错误矩阵

| 场景 | error_code | fallback_to | recoverable | user_visible |
|---|---|---|---:|---:|
| query plan 失败 | QUERY_PLAN_FAILED | raw_query | true | false |
| sparse embedding 失败 | SPARSE_EMBEDDING_FAILED | dense_retrieve | true | false |
| dense embedding 失败 | DENSE_EMBEDDING_FAILED | attached_context_fallback 或 failed | 视情况 | true when failed |
| hybrid retrieve 失败 | HYBRID_RETRIEVE_FAILED | dense_retrieve | true | false |
| scoped retrieve 失败 | SCOPED_RETRIEVE_FAILED | global_only | true | false |
| global retrieve 失败 | GLOBAL_RETRIEVE_FAILED | scoped_only | true | false |
| layered retrieve 失败 | LAYERED_RETRIEVE_FAILED | standard_hybrid | true | false |
| rerank 失败 | RERANK_FAILED | ranked_candidates | true | false |
| citation 校验失败 | CITATION_VERIFY_FAILED | retry_or_standard | true | false |
| Deep synthesis 失败 | DEEP_SYNTHESIS_FAILED | standard_rag | true | true only if no answer |

### 为什么这样做

用户不一定要看到所有内部错误，但系统必须知道结果是否降级、是否仍可信、是否应该展示“不足以回答”。

### 优势

- trace 可用于质量分析。
- UI 可以选择展示用户可见错误。
- eval 可以统计 recoverable/unrecoverable failure。

### 劣势

- 初期字段会比现在多。
- 需要保持向后兼容，避免前端 schema 崩。

### 验收

- 所有失败路径都有 `stage_errors`。
- 所有 `stage_errors` 至少保持旧字段，新增字段可选。
- public schema 接受扩展字段。

## Phase 6：Deep Mode 严格边界

### 总体判断

Deep Mode 不能直接作为“更强回答模式”接入。它必须先是一个可观测、可降级、可验证的 orchestrator。

如果当前或旧分支存在 `deep_mode.py`，不建议在原半成品上补几行生成逻辑。可以使用 `backend/rag/deep_mode.py` 这个文件名，但应该重建边界。

### Deep Mode 不应该做什么

- 不应该重新实现 Milvus 检索。
- 不应该重新实现 rerank。
- 不应该绕过 standard RAG trace。
- 不应该在没有 citation verifier 的情况下返回 active answer。
- 不应该默认替代 Standard 模式。

### Deep Mode 应该只做什么

```text
question
-> plan_subqueries
-> collect evidence through standard RAG
-> build evidence pack
-> synthesize with citation refs
-> verify citations
-> return answer or downgrade
```

### 建议结构

`backend/rag/deep_mode.py`

```python
DeepModeRequest:
    question: str
    context_files: tuple[str, ...]
    max_subqueries: int
    mode: "shadow" | "active"

SubqueryEvidence:
    subquery: str
    docs: list[dict]
    coverage: dict
    retrieval_meta: dict

DeepModeResult:
    executed: bool
    final_answer: str | None
    evidence_pack: list[SubqueryEvidence]
    citations: list[dict]
    verifier: dict
    fallback_to_standard: bool
    trace: dict
```

### Shadow 模式

`RAG_DEEP_SHADOW=true`

- 执行 query decomposition。
- 收集 subquery evidence。
- 生成 evidence coverage trace。
- 不改变用户答案。

### Active 模式

`RAG_DEEP_ACTIVE=true`

仅在满足以下条件时启用：

- 问题被判定为多跳、多文档、多约束综合。
- citation verifier 可用。
- evidence coverage 达到最低阈值。
- fallback deadline 足够。

### 降级规则

- subquery 规划失败：standard RAG。
- 任一子查询检索失败：保留成功 evidence，trace 标记 partial coverage。
- synthesis 失败：standard RAG。
- citation verifier 失败：最多 retry 一次；仍失败则 standard RAG 或返回证据不足。

### 为什么这样做

Deep 的价值来自“结构化分解 + 证据覆盖 + 可验证综合”，不是来自盲目多检索。没有边界的 Deep 会放大延迟、幻觉和维护成本。

### 优势

- 可以证明 Deep 是否真的帮助。
- 失败时有清晰降级。
- 不会复制第四套路由。
- 后续可以逐步增强 coverage 和 verifier。

### 劣势

- 初期实现看起来保守。
- active Deep 不会马上覆盖所有复杂问题。
- 需要额外 eval 数据集衡量 Deep 是否值得默认开启。

### 验收

- shadow Deep 不改变最终答案。
- active Deep 必须返回非空 `final_answer` 或明确 `fallback_to_standard=true`。
- 所有 final answer 引用必须通过 citation verifier。
- trace 包含：
  - `deep_executed`
  - `deep_mode`
  - `subqueries`
  - `evidence_coverage`
  - `citation_verifier`
  - `fallback_to_standard`

## Phase 7：验证矩阵

### 单测

必须覆盖：

- layered candidate strategy true/false 共享 postprocess。
- rerank 失败降级到原排序并写 structured stage error。
- sparse embedding 失败 dense fallback。
- dense embedding 失败 failed meta 或 attachment fallback。
- fallback candidate-only 减少 rerank 次数。
- Agent tool path trace 回传。
- direct attachment path trace delivery。
- citation verifier valid/invalid/unknown refs。
- Deep shadow 不改变答案。
- Deep active 引用校验失败后 retry 或降级。

### 集成测试

运行：

```powershell
uv run pytest tests/ -q
uv run python -m compileall backend scripts
uv run ruff check backend/ scripts/ tests/
```

### RAG smoke

至少保留：

```powershell
uv run python scripts/evaluate_rag_matrix.py --dataset-profile smoke --documents-dir "C:\Users\goahe\Desktop\Project\doc" --limit 1 --top-k 5 --mode retrieval --variants K2 --skip-reindex --skip-coverage-check
```

### 专项 eval

新增或扩展评测维度：

- fallback 触发率
- fallback helped/hurt
- candidate-only latency delta
- citation validity
- Deep shadow coverage
- Deep active answer quality

## 推荐提交顺序

1. `Clarify layered candidate strategy boundaries`
2. `Add deterministic RAG citation verifier`
3. `Budget expanded fallback with candidate-only retrieval`
4. `Unify chat RAG execution policy contracts`
5. `Extend structured RAG stage errors`
6. `Add gated Deep Mode evidence orchestration`

每个提交必须能单独回滚。Deep active 必须最后进入，而且默认关闭。

## 最终建议

这份方案值得继续推进，但应该从“大而全的改造计划”改成“状态感知、feature-flag 驱动、先 verifier 后 Deep”的计划。

最优先的下一步不是 Deep，也不是 Agent 路径统一，而是 citation verifier。它是 Deep active 的前置安全网，也是后续回答质量评估的基础。

第二优先是 fallback candidate-only，因为它直接对应用户可感知延迟，而且可以通过 `_rerank_documents()` 调用次数和 smoke/eval 明确验证收益。

第三优先才是 Agent/直接路径统一策略。当前已经统一了格式和 trace，继续推进执行策略前应先保留默认行为。

Deep Mode 可以直接使用 `backend/rag/deep_mode.py` 作为文件，但不应沿用旧半成品写法。它必须是薄 orchestrator，不拥有独立检索栈，不默认 active，不绕过 citation verifier。

## 执行记录（2026-05-01）

本轮已按 feature flag 优先、默认行为不变的原则完成代码落地：

- 前置 `StageError` 最小结构化语义：`error_code`、`severity`、`recoverable`、`user_visible`。
- 将 layered 明确标记为 `candidate_strategy=layered_split`，最终 rerank 仍统一走 shared pipeline。
- 新增运行时 citation verifier：默认关闭，仅在 `RAG_CITATION_VERIFY_ENABLED=true` 时写入 trace。
- 新增 fallback candidate-only 路径：默认关闭，仅在 `RAG_FALLBACK_CANDIDATE_ONLY=true` 时第二轮只召回候选，合并后统一 rerank。
- 新增 `backend/chat/rag_execution.py`，统一 Agent 工具路径和附件直接路径的 RAG 执行契约与 trace 字段。
- 新增 gated Deep Mode orchestrator：不拥有独立检索栈，active 必须同时满足 `RAG_DEEP_ACTIVE=true` 和 citation verifier 可用，否则降级到 standard。

本轮验证：

```powershell
python -m py_compile backend\chat\agent.py backend\chat\tools.py backend\chat\rag_execution.py backend\rag\citations.py backend\rag\deep_mode.py backend\rag\formatting.py backend\rag\layered_rerank.py backend\rag\pipeline.py backend\rag\runtime_config.py backend\rag\trace.py backend\rag\types.py backend\rag\utils.py
python -m pytest -q tests
git diff --check
```

结果：`344 passed`，仅有第三方 `pkg_resources` deprecation warnings 和 Git 的 LF/CRLF 工作区提示。
