# SuperHermes 代码瘦身设计

> 日期: 2026-04-26
> 状态: 待审批

## 一、审计结论

经过对全项目源码的逐文件审计，确认以下问题及其严重程度：

### P0 — 必须立即修复

#### 1. utils.py 检索流水线三重重复 (~140 行)

`retrieve_documents()` 中存在三段几乎相同的 rerank→structure_rerank→confidence→trace 构建逻辑：

| 位置 | 行范围 | 说明 |
|------|--------|------|
| `_finish_retrieval_pipeline()` | 191-297 | 已抽象版本，含 try/except 错误处理 |
| 标准路径 (non-scoped) | 998-1036 | 手写重复，retrieval_mode = "hybrid"/"hybrid_boosted" |
| Dense fallback 路径 | 1067-1107 | 手写重复，retrieval_mode = "dense_fallback"/"dense_fallback_boosted" |

三者的差异仅在于：
- `retrieval_mode` 字符串不同 (hybrid / hybrid_boosted / dense_fallback / dense_fallback_boosted)
- 标准路径和 fallback 路径最后多一行 `rerank_meta.update(global_trace)`
- fallback 路径的 `hybrid_error` 非空
- `_finish_retrieval_pipeline` 有 try/except 兜底，另外两段没有

修复方案：扩展 `_finish_retrieval_pipeline()` 接受 `retrieval_mode`、`extra_trace`（已有）、`hybrid_error` 参数，让标准路径和 fallback 路径都调用它。

**预计减少：~120 行**

#### 2. 环境变量 + 模型初始化散布在 6 个文件

| 文件 | ARK_API_KEY / API_KEY | MODEL | BASE_URL | load_dotenv() | init_chat_model() |
|------|----------------------|-------|----------|---------------|-------------------|
| `chat/agent.py` | ARK_API_KEY | MODEL | BASE_URL | yes | 1 次 |
| `chat/tools.py` | — | — | — | yes | — |
| `rag/pipeline.py` | ARK_API_KEY | MODEL, FAST_MODEL, GRADE_MODEL | BASE_URL | yes | 3 次 |
| `rag/utils.py` | ARK_API_KEY | MODEL | BASE_URL | yes | 1 次 |
| `evaluation/answer_eval.py` | ARK_API_KEY \|\| OPENAI_API_KEY | ANSWER_MODEL, JUDGE_MODEL | BASE_URL \|\| OPENAI_BASE_URL | yes | 2 次 |
| `infra/embedding.py` | — | — | — | yes | — |

共 **8 处 `load_dotenv()`**、**6 处 `init_chat_model()`**、**5 处读取同一组 ARK env vars**。

修复方案：创建 `backend/config.py` 集中读取所有环境变量，各模块从 config 导入。`load_dotenv()` 只在入口 (`application/main.py`) 调用一次。

**预计减少：~40 行 + 消除配置不一致风险**

### P1 — 应该修复

#### 3. 完全死代码

| 文件 | 原因 |
|------|------|
| `backend/rag/runtime/config.py` | 定义了 env_bool/env_float/env_int，零导入 |
| `backend/rag/runtime/__init__.py` | 空文件 |
| `backend/rag/rules.py` | 定义了 ANCHOR_PATTERN + extract_anchor_id + normalize_title，零导入 |

**预计减少：~65 行**

#### 4. ANCHOR_PATTERN 正则重复定义

`utils.py` 第 110 行定义了 `_ANCHOR_PATTERN`，与 `rules.py` 中定义的完全相同。删除 `rules.py` 后，保留 `utils.py` 中的版本。

#### 5. _env_bool() 重复定义

`pipeline.py` 第 17 行和 `utils.py` 第 55 行各定义了一个 `_env_bool()`，逻辑完全相同。集中到 `config.py` 后删除。

#### 6. services/chat_service.py 纯透传层

```python
def run_chat(...): return chat_with_agent(...)
def run_chat_stream(...): return chat_with_agent_stream(...)
```

`routers/chat.py` 可以直接导入 `chat/agent.py`。

**预计减少：17 行 + 1 个文件**

#### 7. routers/__init__.py 无用重导出

`api.py` 直接从子模块导入，不经过 `__init__.py`。该文件的重导出从未被使用。

#### 8. 项目.md 与 README.md 内容重复

`项目.md`(346 行) 与 `README.md`(541 行) 大量内容重复（架构图、目录结构、API 速览等）。`项目.md` 是早期设计文档，应删除。

### P2 — 建议修复

#### 9. utils.py 中的纯透传包装函数

以下函数只是直接委托给 retrieval.py/rerank.py，没有增加任何逻辑：

| 函数 | 行数 | 问题 |
|------|------|------|
| `_weighted_rrf_merge()` | 137-141 | 纯透传，调用处可直接用底层函数 |
| `_annotate_scope_scores()` | 187-188 | 纯透传 |
| `_dedupe_docs()` | 711-712 | 纯透传 |

以下函数绑定了模块级常量，有一定存在价值（但也可以改为直接传参）：

| 函数 | 行数 | 绑定的常量 |
|------|------|-----------|
| `_apply_filename_boost()` | 144-152 | DOC_SCOPE_MATCH_BOOST, MILVUS_RRF_K 等 |
| `_apply_heading_lexical_scoring()` | 155-164 | HEADING_LEXICAL_WEIGHT, MILVUS_RRF_K |
| `_rerank_rrf_score()` | 167-168 | MILVUS_RRF_K |
| `_apply_rerank_score_fusion()` | 171-184 | RERANK_FUSION_*_WEIGHT, RERANK_SCORE_FUSION_ENABLED |

建议：保留绑定常量的包装函数（它们有实际价值），只消除 3 个纯透传函数。

**预计减少：~12 行**

#### 10. filename 去重模式重复

`tools.py:set_rag_context_files()` (行 42-50) 和 `agent.py:_normalize_context_files()` (行 114-123) 实现了完全相同的去重逻辑。应提取为共享工具函数。

#### 11. tools.py 全局可变状态

5 个全局变量管理请求级状态，并发请求可能互相污染：

```python
_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_CONTEXT_FILES_THIS_TURN = []
_RAG_STEP_QUEUE = None
_RAG_STEP_LOOP = None
```

修复方案：使用 `contextvars.Context` 替代全局变量，确保每个 asyncio Task 有独立状态。

#### 12. elapsed_ms 公开别名冗余

`utils.py` 第 346 行 `elapsed_ms = _elapsed_ms` 给私有函数起了公开别名。`pipeline.py` 通过 import 使用 `elapsed_ms`。直接将原函数命名为 `elapsed_ms` 即可。

### P3 — 低优先级

#### 13. frontend/script.js 1129 行单文件

前端是 Vue 3 CDN Options API，无组件拆分。这是架构层面的问题，不适合在代码瘦身中解决。

## 二、修复计划

### 阶段 A：删除死代码 + 文件清理（风险极低）

1. 删除 `backend/rag/runtime/config.py`
2. 删除 `backend/rag/runtime/__init__.py`
3. 删除 `backend/rag/runtime/` 目录
4. 删除 `backend/rag/rules.py`
5. 删除 `backend/services/chat_service.py`
6. 更新 `routers/chat.py` 直接导入 `chat/agent.py`
7. 删除 `backend/routers/__init__.py`（或改为空文件）
8. 删除根目录 `项目.md`

**验证：** 运行 `pytest` 确保零回归

### 阶段 B：集中环境变量（风险低）

1. 创建 `backend/config.py`，集中读取所有环境变量
2. 在 `application/main.py` 入口调用一次 `load_dotenv()`
3. 各模块从 `backend.config` 导入配置
4. 删除各文件中的 `load_dotenv()` 调用和重复的 `os.getenv()`
5. 集中 `_env_bool()` 到 `config.py`

**验证：** 运行 `pytest` + 手动启动服务测试

### 阶段 C：消除 utils.py 检索流水线重复（风险中）

1. 扩展 `_finish_retrieval_pipeline()` 签名，增加参数：
   - `retrieval_mode: str` — 指定返回的 mode 字符串
   - `hybrid_error: str | None = None`
   - `extra_trace` 已有，保留
2. 让标准路径和 fallback 路径都调用 `_finish_retrieval_pipeline()`
3. fallback 路径在调用前自行完成 dense-only 检索，将结果传给 `_finish_retrieval_pipeline()`

**验证：** 运行 `pytest` + 对比重构前后的 RAG trace 输出

### 阶段 D：小型清理（风险低）

1. 内联 3 个纯透传包装函数
2. 统一 filename 去重逻辑到共享函数
3. 将 `_elapsed_ms` 直接命名为 `elapsed_ms`，删除别名
4. 更新 `docs/ARCHITECTURE.md`

**验证：** 运行 `pytest`

### 阶段 E（可选）：并发安全改进

1. 将 `tools.py` 全局变量迁移到 `contextvars.Context`

**验证：** 运行 `pytest` + 并发压力测试

## 三、预计效果

| 阶段 | 减少行数 | 减少文件数 |
|------|---------|-----------|
| A: 死代码清理 | ~80 行 | 5 个文件 |
| B: 集中配置 | ~40 行 | +1 (config.py) |
| C: 流水线去重 | ~120 行 | 0 |
| D: 小型清理 | ~25 行 | 0 |
| **总计** | **~265 行** | **净减 4 个文件** |

utils.py 将从 1154 行降至约 970 行，仍在可接受范围内（主要由检索逻辑的固有复杂度决定）。
