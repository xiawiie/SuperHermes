# RAG 标注数据集质量评估

## 推荐结论

正式 benchmark 优先使用 `rag_doc_gold_v2.jsonl`。

`rag_doc_gold_v1.jsonl` 继续保留为全量覆盖/烟测集；`rag_doc_gold_v2.jsonl` 是更严格的硬指标集，适合做主评估。

## v1 状态

`rag_doc_gold_v1.jsonl` 的价值是覆盖广，252 条样本覆盖 84 个源文件，类型分布接近均衡。但它不适合作为强 benchmark：

- 精确重复答案较多，模板化手册会导致 37 组重复答案。
- 少量 PDF 抽取噪声会形成不自然问题。
- 字段更偏兼容旧评估脚本，缺少 qrels/positive contexts 等标准化结构。

## v2 优化结果

`rag_doc_gold_v2.jsonl` 基于 v1 进一步过滤和标准化：

- 样本数：125
- 重复 ID：0
- 重复答案：0
- 覆盖源文件：60
- 类型分布：
  - operation：45
  - specification：42
  - troubleshooting：36
  - safety：2
- 带结构锚点样本：50
- 平均 source coverage：0.983
- 最低 source coverage：0.727
- 平均质量分：6.086

删除/过滤内容：

- 非知识库 xlsx：3 条
- 重复答案：全部去重
- 抽取噪声、版权/术语表/乱码片段
- 不自然问题模板，如“如何的...”“如何2）...”
- 关键词无法在答案中命中的样本
- 源文覆盖不足样本

## v2 标准字段

v2 兼容旧字段，同时新增更标准的 benchmark 字段：

- `query`
- `reference_answer`
- `expected_files`
- `expected_pages`
- `expected_page_refs`
- `gold_doc_ids`
- `positive_contexts`
- `relevance_judgments`
- `hard_negative_files`
- `expected_keyword_policy`
- `metadata`
- `quality_checks`

## 评估口径建议

主指标：

1. `file_hit_at_5`
2. `page_hit_at_5`
3. `keyword_required_hit_at_5`
4. `anchor_hit_at_5`
5. `mrr`
6. `context_precision_id_at_5`

不要再只看“任意关键词命中”。`expected_keyword_policy.min_match` 默认要求至少命中 3 个关键词，能显著降低虚假命中。

## 剩余风险

- v2 是更硬的集，但覆盖文件从 84 降到 60，适合作为主 benchmark，不适合替代全量 smoke test。
- PDF 抽取仍可能有少量断词、空格和表格顺序问题。
- 还没有补真实 `root_chunk_id`，需要重建索引后再做 chunk/root id 映射。
