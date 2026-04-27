# SuperHermes 代码瘦身 P0+P1 设计

> 日期: 2026-04-27
> 状态: 已审批
> 范围: P0（流水线去重 + 配置集中化）+ P1（死代码清理）

## 背景

深度审计发现 utils.py 有 ~140 行检索流水线重复代码，环境变量读取散布在 6 个文件，3 个模块完全无导入（死代码）。详见 `docs/2026-04-26-codebase-slim-design.md`。

## 阶段 A：死代码清理

纯删除操作，不改变运行逻辑。

### 删除文件

| 文件 | 原因 |
|------|------|
| `backend/rag/runtime/config.py` | env_bool/env_float/env_int，全项目零导入 |
| `backend/rag/runtime/__init__.py` | 空文件 |
| `backend/rag/rules.py` | ANCHOR_PATTERN + extract_anchor_id，全项目零导入 |
| `backend/services/chat_service.py` | 纯透传层，唯一调用者 routers/chat.py |
| 根目录 `项目.md` | 与 README.md 大量重复 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `backend/routers/chat.py` | `from backend.services.chat_service import run_chat, run_chat_stream` → `from backend.chat.agent import chat_with_agent as run_chat, chat_with_agent_stream as run_chat_stream`。注意：旧 chat_service 做了 `context_files or []` 转换，但 routers/chat.py 调用时已传 `request.context_files or []`，所以删除透传层安全无副作用 |

### 验证

`pytest` 全量通过，零回归。

## 阶段 B：配置集中化

### 新建文件

**`backend/config.py`** — 集中**跨文件共享的**环境变量读取和 `load_dotenv()` 调用。

**范围边界：** 只迁移在 2 个以上文件中重复读取的环境变量。模块自用的配置（如 utils.py 中 ~35 个 RAG 调优参数）留在原模块，不集中——保持配置与使用位置相近。

```python
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM（agent.py, pipeline.py, utils.py, answer_eval.py 共享）---
ARK_API_KEY = os.getenv("ARK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
FAST_MODEL = os.getenv("FAST_MODEL")
GRADE_MODEL = os.getenv("GRADE_MODEL")

# --- Evaluation ---
ANSWER_EVAL_GENERATION_MODEL = os.getenv("ANSWER_EVAL_GENERATION_MODEL")
ANSWER_EVAL_JUDGE_MODEL = os.getenv("ANSWER_EVAL_JUDGE_MODEL")

# --- Milvus（embedding.py, milvus_client.py, query_plan.py 共享）---
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings_collection")

# --- Text Mode（embedding.py, loader.py 共享）---
EVAL_RETRIEVAL_TEXT_MODE = os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context")

# --- External ---
AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default

def env_float(name: str, default: float = 0.0) -> float:
    val = os.getenv(name, "").strip()
    return float(val) if val else default

def env_int(name: str, default: int = 0) -> int:
    val = os.getenv(name, "").strip()
    return int(val) if val else default
```

### 修改文件

| 文件 | 删除内容 | 导入替换 |
|------|---------|---------|
| `chat/agent.py` | `load_dotenv()` + `API_KEY = os.getenv(...)` + `MODEL = ...` + `BASE_URL = ...` | `from backend.config import ARK_API_KEY as API_KEY, MODEL, BASE_URL` |
| `chat/tools.py` | `load_dotenv()` + `AMAP_*` 读取 | `from backend.config import AMAP_WEATHER_API, AMAP_API_KEY` |
| `rag/pipeline.py` | `load_dotenv()` + `_env_bool()` + env 读取 | `from backend.config import ...` + 使用 `env_bool` |
| `rag/utils.py` | `load_dotenv()` + `_env_bool()` + env 读取 | `from backend.config import ...` + 使用 `env_bool` |
| `infra/embedding.py` | `load_dotenv()` + env 读取 | `from backend.config import MILVUS_COLLECTION, EVAL_RETRIEVAL_TEXT_MODE` |
| `infra/db/database.py` | `load_dotenv()` | 依赖 config 模块已触发 load_dotenv（无需显式导入） |
| `infra/vector_store/milvus_client.py` | `load_dotenv()` + MILVUS_COLLECTION 读取 | `from backend.config import MILVUS_COLLECTION` |
| `evaluation/answer_eval.py` | `load_dotenv()` + env 读取 | `from backend.config import ...`（含 OPENAI_API_KEY fallback） |
| `rag/query_plan.py` | MILVUS_COLLECTION 读取 | `from backend.config import MILVUS_COLLECTION` |
| `documents/loader.py` | EVAL_RETRIEVAL_TEXT_MODE 读取 | `from backend.config import EVAL_RETRIEVAL_TEXT_MODE` |

注意：`load_dotenv()` 在 `config.py` 模块加载时执行一次即可。所有导入 config 模块的文件自动获得环境变量。模块自用的配置（如 utils.py 中 35 个 RAG 调优参数、security/auth.py 中 JWT 配置）保持原位不动。

### 验证

`pytest` 全量通过 + 手动启动服务确认正常。

## 阶段 C：流水线去重

### 改动范围

只改 `backend/rag/utils.py`。

### 步骤 1：扩展 `_finish_retrieval_pipeline` 签名

新增 2 个参数：

```python
def _finish_retrieval_pipeline(
    query: str,
    search_query: str,
    retrieved: list[dict],
    top_k: int,
    candidate_k: int,
    timings: Dict[str, float],
    stage_errors: list[Dict[str, str]],
    total_start: float,
    extra_trace: dict | None = None,
    query_plan: QueryPlan | None = None,
    context_files: list[str] | None = None,
    base_filter: str | None = None,
    retrieval_mode: str = "hybrid",        # 新增
    hybrid_error: str | None = None,        # 新增
) -> Dict[str, Any]:
```

### 步骤 2：更新函数内部

- 行 225：**删除** `"hybrid_scoped" if (...) else "hybrid"` 条件判断，改为直接赋值参数 `rerank_meta["retrieval_mode"] = retrieval_mode`
- 行 236：`hybrid_error` 改为使用参数 `rerank_meta["hybrid_error"] = hybrid_error`
- except 块（行 263）：`"hybrid_error": None` 改为 `"hybrid_error": hybrid_error`
- except 块中的 `"retrieval_mode": "failed"` 保持不变（错误场景统一用 "failed"）

### 步骤 2.5：更新 scoped 路径调用（行 934）

scoped 路径当前依赖函数内部自动计算 mode。扩展后需要显式传入：

```python
scope_mode = "hybrid_scoped" if scope_trace.get("scope_filter_applied") else "hybrid"
return _finish_retrieval_pipeline(
    ...,
    retrieval_mode=scope_mode,
)
```

### 步骤 3：替换标准路径（原 998-1036 行）

将手写的 rerank→structure→confidence→trace 替换为：

```python
retrieval_mode = "hybrid_boosted" if global_trace.get("filename_boost_applied") else "hybrid"
return _finish_retrieval_pipeline(
    query, search_query, retrieved, top_k,
    candidate_k, timings, stage_errors, total_start,
    extra_trace=global_trace,
    context_files=context_files,
    retrieval_mode=retrieval_mode,
)
```

### 步骤 4：替换 fallback 路径（原 1067-1107 行）

将手写的 rerank→structure→confidence→trace 替换为：

```python
retrieval_mode = "dense_fallback_boosted" if global_trace.get("filename_boost_applied") else "dense_fallback"
return _finish_retrieval_pipeline(
    query, search_query, retrieved, top_k,
    candidate_k, timings, stage_errors, total_start,
    extra_trace=global_trace,
    context_files=context_files,
    retrieval_mode=retrieval_mode,
    hybrid_error=hybrid_error,
)
```

### 副作用

标准路径和 fallback 路径原本没有 try/except 保护 rerank 部分，统一调用 `_finish_retrieval_pipeline` 后自动获得 try/except 兜底。这是行为改进，不是回归。

### 验证

`pytest` + 对比重构前后的 RAG trace JSON，确保字段和值完全一致。

## 执行顺序

1. 阶段 A → `pytest` → 验证通过
2. 阶段 B → `pytest` → 验证通过
3. 阶段 C → `pytest` → 验证通过
4. code-review → 确认无问题
5. 进入 P2

## 预计效果

| 指标 | 数值 |
|------|------|
| 减少行数 | ~240 行（含阶段 A 删除 ~80 行 + 阶段 B 净减 ~40 行 + 阶段 C 减 ~120 行） |
| 减少文件 | 净减 4 个（删 5 文件 + 新建 1 config.py） |
| utils.py 行数 | 1154 → ~1030 |
