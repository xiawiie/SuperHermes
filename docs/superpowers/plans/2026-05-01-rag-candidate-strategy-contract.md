# RAG Candidate Strategy Contract

## 结论

RAG 保留多种候选生成方案，但最终 rerank 只能有一个共享出口。

当前约定是：

```text
standard / scoped candidate strategy
layered candidate strategy
  -> finish_retrieval_pipeline
  -> shared rerank_documents
  -> shared structure rerank / confidence / trace
```

这不是删除 layered。Layered 仍然负责 L0 split retrieve、L1 文件感知预筛、候选保护和候选压缩。收敛的只是最终 rerank、trace、错误语义和后处理出口。

## 为什么不保留两套完整 rerank

保留两种候选方案是有价值的，因为它们解决的问题不同：

- `global_hybrid` / `scoped_hybrid` 适合稳定、简单、可解释的标准召回。
- `layered_split` 适合更复杂的候选扩展、文件级聚合和候选保护。

但保留两套最终 rerank 会让系统重新分叉：

- CrossEncoder 参数和缓存要维护两份。
- 分数融合和 top-k 截断可能产生不可见差异。
- rerank 失败处理、trace 字段、评测指标会漂移。
- 质量变化难以归因：无法判断是候选集变了，还是 rerank 逻辑变了。

因此候选策略可以多样，最终 rerank 出口必须统一。

## Trace 契约

每条候选路径必须写入以下字段：

```text
candidate_strategy
candidate_strategy_family
candidate_strategy_version
rerank_strategy
rerank_contract_version
candidate_strategy_fallback_from  # 仅在候选策略降级时出现
```

当前稳定值：

```text
candidate_strategy_version = candidate-strategy-v1
rerank_contract_version = shared-rerank-v1
rerank_strategy = shared_pipeline
```

候选策略：

```text
global_hybrid   -> standard family
scoped_hybrid   -> standard family
layered_split   -> layered family
dense_fallback  -> standard family, with fallback_from when applicable
```

## 行为边界

允许：

- 新增候选生成策略。
- 调整 layered 的 L0/L1 候选筛选、quota、保护规则。
- 为候选添加额外特征，供 shared rerank 或 score fusion 使用。

不允许：

- 在 layered 内部重新引入独立 CrossEncoder 最终排序。
- 让某个候选策略绕过 `finish_retrieval_pipeline` 后处理。
- 使用同一个 trace 字段同时表示候选策略和最终 rerank 策略。

## 验收标准

- Standard、Scoped、Layered 都必须有 `candidate_strategy` trace。
- Layered 完整检索必须只调用共享 `_rerank_documents()` 一次。
- Candidate-only 检索可以跳过 rerank，但必须显式标记候选策略。
- 新增候选策略时必须补对应 trace 测试。

