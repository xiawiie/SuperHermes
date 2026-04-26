# Phase 0 基线快照与契约冻结（2026-04-26）

## 1. 功能基线

- `pytest tests -q`：**230 passed, 1 warning**（jiеба 第三方警告，不影响）
- 快速门禁（60 测试）：**60 passed in ~9s**
- 前端语法：`node --check frontend/script.js` **通过**

## 2. 质量基线

- `uv run ruff check backend scripts tests`：**All checks passed**
- `uv run python -m compileall backend scripts`：**通过**

## 3. 性能预算（来自 latest full-gold 报告）

基线报告路径：`.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/summary.json`

### 3.1 关键指标阈值（不得下降）

| 指标 | GB0 基线 | GS3 推荐基线 | V3Q 质量天花板 |
|---|---|---|---|
| File@5 | 0.744 | 0.920 | 0.992 |
| Chunk@5 | 0.000 | 0.655 | 0.724 |
| MRR | 0.296 | 0.708 | 0.704 |

### 3.2 延迟预算（P50 不超预算）

| 变体 | P50 ms | P95 ms |
|---|---|---|
| GB0 | 939 | 1061 |
| GS3 | 1144 | ~1500 |
| V3Q | 3938 | ~5000 |

预算规则：任何重构不得使推荐基线（GS3）的 P50/P95 劣化超过 10%。

## 4. API 契约快照

### 4.1 认证路由（/auth/*）

- `POST /auth/register` → `AuthResponse(access_token, username, role)`
- `POST /auth/login` → `AuthResponse`
- `GET /auth/me` → `CurrentUserResponse(username, role)`

### 4.2 会话路由（/sessions/*）

- `GET /sessions` → `SessionListResponse(sessions[])`
- `GET /sessions/{id}` → `SessionMessagesResponse(messages[])`
- `PATCH /sessions/{id}` → `SessionRenameResponse`
- `DELETE /sessions/{id}` → `SessionDeleteResponse`

### 4.3 聊天路由

- `POST /chat` → `ChatResponse(response)` [同步]
- `POST /chat/stream` → SSE 流式输出 [异步]

### 4.4 文档路由

- `GET /documents` → `DocumentListResponse(documents[])`
- `POST /documents/upload` → `DocumentUploadResponse`
- `DELETE /documents/{filename}` → `DocumentDeleteResponse`

## 5. CLI 契约快照

主评测脚本：`scripts/evaluate_rag_matrix.py`

关键参数：
- `--dataset-profile`：frozen/gold/smoke/natural/custom
- `--variants`：逗号分隔变体名
- `--skip-reindex`：跳过重建索引
- `--limit`：限制样本数
- `--run-id`：运行标识

输出结构：
- `.jbeval/reports/<run-id>/summary.json`
- `.jbeval/reports/<run-id>/summary.md`
- `.jbeval/reports/<run-id>/results.jsonl`
- `.jbeval/reports/<run-id>/variant-*.jsonl`

## 6. 评测产物字段快照

关键输出字段（summary.json 中每个变体）：
- `rows`, `file_hit_at_5`, `file_page_hit_at_5`, `chunk_hit_at_5`, `root_hit_at_5`
- `mrr`, `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`
- `error_rate`, `fallback_trigger_rate`
- `rewrite_strategy_distribution`

## 7. 错误路径快照

- 401：认证过期（前端触发登出）
- 404：会话/文档不存在
- 429：上游模型限流
- 500：内部服务异常（统一包装错误消息）

## 8. 配置默认值快照

关键环境变量默认值（来自 `backend/rag_utils.py` 与 `.env.example`）：
- `LEAF_RETRIEVE_LEVEL`：3
- `MILVUS_RRF_K`：60
- `MILVUS_SEARCH_EF`：64
- `RERANK_TOP_N`：0
- `AUTO_MERGE_ENABLED`：true
- `CONFIDENCE_GATE_ENABLED`：false
- `QUERY_PLAN_ENABLED`：false

## 9. Smoke 评测阻塞项登记

- 状态：在当前环境中 smoke 评测命令长时间无输出（疑似 Milvus/索引连接问题）
- 阻塞命令：`uv run python scripts/evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --skip-reindex --limit 1`
- 重试条件：Milvus 服务恢复、索引可用后重新执行
- 影响：不影响代码重构验证（测试/质量门禁已覆盖功能验证）

## 10. 阶段验收结论

- Phase 0 所有快照已采集并固化。
- 阻塞项已登记，不影响进入 Phase 1。
