# RAG High-Risk Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛 SuperHermes RAG 中四个高风险问题：两套 rerank 路径并存、fallback 二次完整检索延迟、附件路径与 Agent 工具路径行为不一致、错误处理语义不统一。

**Architecture:** 保留 `retrieve_documents()` 和 `run_rag_graph()` 作为外部兼容入口，内部引入类型化检索契约、统一候选召回接口、统一 postprocess 出口、请求级检索会话、统一 EvidenceBundle 和结构化 StageError。`layered_rerank.py` 只保留 L0/L1/L2/L3 所需的候选选择和参数决策能力，不再拥有独立后处理语义。

**Tech Stack:** Python 3.12, Pydantic, LangGraph, LangChain tools, Milvus, sentence_transformers CrossEncoder, pytest/unittest. 不新增依赖。

---

## Context And Evidence

当前高风险点对应代码位置：

- `backend/rag/utils.py:180` 的 `_finish_retrieval_pipeline()` 同时处理 layered 和 standard 后处理。
- `backend/rag/utils.py:864` 的 `retrieve_documents()` 仍把 QueryPlan、embedding、Milvus、Layered/Standard 分支、fallback、trace 拼装放在一个函数中。
- `backend/rag/utils.py:938`、`backend/rag/utils.py:1147` 通过 `_LAYERED_RERANK_ENABLED` 切换 `split_retrieve()` 与 `hybrid_retrieve()`，后续候选处理语义仍有差异。
- `backend/rag/pipeline.py:220` 的 `retrieve_initial()` 第一次调用 `retrieve_documents()`。
- `backend/rag/pipeline.py:516` 的 `retrieve_expanded()` 在 fallback 后再次调用 `retrieve_documents()`，`complex` 模式虽并行执行两路扩展检索，但仍是完整检索流水线。
- `backend/chat/agent.py:277` 的 `_retrieve_attached_context()` 在有 `context_files` 时直接运行 `run_rag_graph()`。
- `backend/chat/agent.py:342` 有附件时直接调用 `model_instance.invoke()`，`backend/chat/agent.py:344` 无附件时调用 `agent_instance.invoke()`。
- `backend/chat/tools.py:133` 的 `search_knowledge_base()` 是 Agent 工具路径入口，也调用 `run_rag_graph()`。
- `backend/rag/types.py:7` 的 `StageError` 只有 `stage/error/fallback_to`，无法表达严重级别、是否恢复、是否用户可见。

## Cleanup Plan

这是架构收敛和 refactor 工作。实施时必须遵守以下顺序：

1. 先写回归测试，锁住当前可接受行为：检索结果数量、trace 字段、fallback 路由、附件路径保存 trace、rerank 失败降级。
2. 每次只收敛一个边界：契约、postprocess、fallback session、chat evidence、error semantics 分开提交。
3. 优先删除重复分支，避免新增第三套路径。
4. 不改变默认用户行为，除非测试和 trace 明确证明新行为更一致。
5. 不新增依赖，不引入 version-suffixed 生产模块名。

## Design Judgment

### Decision 1: Layered Becomes Candidate Strategy

`LAYERED_RERANK_ENABLED` 当前实际上切换了候选集构造和部分后处理语义。更合适的边界是：

```text
RetrievalStrategy:
  standard_hybrid -> CandidateSet
  scoped_hybrid -> CandidateSet
  layered_split -> CandidateSet

UnifiedPostprocess:
  CandidateSet -> rerank -> structure_rerank -> confidence -> RetrievalResult
```

Layered 只决定候选来自哪里、CE 输入 K 和 root cap 建议值，不再决定 trace 字段、错误格式或最终排序出口。

### Decision 2: 二次检索不是简单禁止，而是预算化

改写后的 query 不同，第二次 embedding 和 Milvus 检索有时是必要的。优化目标不是强行复用错误的 embedding，而是：

- 同一 `run_rag_graph()` 内复用 Milvus manager、embedding service、reranker runtime、cache key 逻辑和 request cache。
- 当扩展 query 与原 query 等价或扩展为空时，跳过第二次检索。
- 当 fallback deadline 不足以完成第二次完整 CE 时，返回 initial retrieval，并把原因写入 trace。
- 给 expanded retrieval 使用更明确的 budget，例如 `expanded_candidate_k`、`expanded_rerank_input_cap`，默认保持当前质量，只有在 deadline 压力下收缩。

### Decision 3: 统一答案层 EvidenceBundle

附件路径和 Agent 工具路径不应分别拼 SystemMessage 和 Tool 返回字符串。统一做法是：

```text
RAG graph -> EvidenceBundle -> formatter -> model/agent message
```

有 `context_files` 时仍确定性检索；无 `context_files` 时，对明显依赖文档的问题也可以确定性检索，其余问题继续由 Agent 决定是否调用工具。这样既减少行为分裂，也避免所有普通聊天都强制检索。

### Decision 4: 错误要成为产品语义，不只是异常字符串

rerank 失败返回原排序可以接受，但不能静默。embedding 失败也不应只靠外层大异常兜底。统一错误结构必须能回答：

- 哪个阶段失败？
- 是否已降级？
- 降级到什么？
- 结果是否还能用于回答？
- 是否应该暴露给用户或只进入 trace？

## Acceptance Criteria

- `LAYERED_RERANK_ENABLED=true/false` 都经过同一个 postprocess 函数，测试能证明 rerank failure、structure rerank、confidence trace 字段一致。
- `retrieve_expanded()` 在扩展 query 为空、等价或 deadline 不足时不再启动第二次完整检索，并在 trace 中记录 `expanded_retrieval_skipped_reason`。
- `complex` fallback 继续并行两路扩展检索，新增测试证明 wall time 小于两次串行总和。
- 有附件和无附件但显式文档问题都通过同一个 EvidenceBundle formatter 注入上下文。
- RAG trace 中的 `stage_errors` 每项包含 `stage`、`error_code`、`severity`、`recoverable`、`fallback_to`、`user_visible`。
- rerank 失败、hybrid 失败、dense fallback 成功、embedding 失败且无 fallback 四类场景都有测试。
- 所有新增默认配置保持现有生产行为：`RAG_FALLBACK_ENABLED=false` 时不触发改写检索。

## File Structure

### New Files

- `backend/rag/contracts.py`  
  定义 `RetrievalRequest`、`RetrievalPlan`、`CandidateSet`、`RetrievalResult`、`RetrievalSession`、`RetrievalBudget`。

- `backend/rag/postprocess.py`  
  统一 `rerank -> structure_rerank -> confidence -> meta` 出口，从 `utils.py` 迁移 `_finish_retrieval_pipeline()` 的主体。

- `backend/rag/errors.py`  
  定义结构化 `StageError`、错误码、severity、fallback 语义。

- `backend/chat/evidence.py`  
  定义 `EvidenceBundle`、`build_evidence_bundle()`、`format_evidence_for_model()`、`should_force_knowledge_retrieval()`。

### Modified Files

- `backend/rag/utils.py`  
  保留 `retrieve_documents()` 兼容入口，内部改为构造 request/session/plan，调用候选召回和统一 postprocess。

- `backend/rag/layered_rerank.py`  
  保留 L1 candidate 和 adaptive CE 逻辑，移除或停止输出与最终 trace 语义冲突的字段。

- `backend/rag/rerank.py`  
  保持 CrossEncoder 运行逻辑，补充结构化错误 meta 和 request/session cache 可观测字段。

- `backend/rag/pipeline.py`  
  在 `RAGState` 中传递 `RetrievalSession` 或 session id，给 `retrieve_expanded()` 增加 budget/skip 逻辑。

- `backend/rag/types.py`  
  将现有 `StageError` 迁移到 `backend/rag/errors.py`，保留兼容 import 或重导出。

- `backend/contracts/schemas.py`  
  扩展 `RagTrace` 对新 `stage_errors` 字段和 evidence 字段的 schema 支持。

- `backend/chat/agent.py`  
  附件路径和强制知识检索路径使用同一个 EvidenceBundle formatter。

- `backend/chat/tools.py`  
  `search_knowledge_base()` 使用 EvidenceBundle formatter 返回检索证据，trace 存储保持一致。

### Tests

- `tests/test_rag_contracts.py`
- `tests/test_rag_postprocess_convergence.py`
- `tests/test_rag_fallback_budget.py`
- `tests/test_chat_evidence_bundle.py`
- `tests/test_rag_error_semantics.py`
- 修改现有 `tests/test_rag_observability.py`
- 修改现有 `tests/test_rag_pipeline.py`

---

## Phase 1: Type Contracts And Error Semantics

### Task 1: Add Retrieval Contracts

**Files:**
- Create: `backend/rag/contracts.py`
- Test: `tests/test_rag_contracts.py`

- [ ] **Step 1: Write contract tests**

Add `tests/test_rag_contracts.py`:

```python
from backend.rag.contracts import (
    CandidateSet,
    RetrievalBudget,
    RetrievalPlan,
    RetrievalRequest,
    RetrievalResult,
    RetrievalSession,
)


def test_retrieval_request_normalizes_context_files():
    req = RetrievalRequest(query=" q ", top_k=5, context_files=("a.pdf", "a.pdf", " b.pdf "))

    assert req.query == "q"
    assert req.context_files == ("a.pdf", "b.pdf")


def test_candidate_set_records_strategy_without_postprocess_fields():
    candidates = CandidateSet(
        strategy="layered_split",
        docs=[{"chunk_id": "c1", "text": "alpha"}],
        plan=RetrievalPlan(candidate_k=120, ce_input_k=80, ce_top_n=30),
        meta={"scope_mode": "none"},
    )

    assert candidates.strategy == "layered_split"
    assert candidates.plan.ce_input_k == 80
    assert "rerank_applied" not in candidates.meta


def test_retrieval_session_has_stable_request_cache_key():
    session = RetrievalSession(session_id="s1")
    req = RetrievalRequest(query="hello", top_k=5, context_files=())
    budget = RetrievalBudget(deadline_seconds=6.0)

    key1 = session.cache_key(req, budget)
    key2 = session.cache_key(req, budget)

    assert key1 == key2
    assert key1.startswith("retrieval-session:s1:")


def test_retrieval_result_exposes_trace_meta():
    result = RetrievalResult(
        docs=[{"chunk_id": "c1"}],
        meta={"retrieval_mode": "hybrid", "stage_errors": []},
    )

    assert result.as_dict()["docs"][0]["chunk_id"] == "c1"
    assert result.as_dict()["meta"]["retrieval_mode"] == "hybrid"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_rag_contracts.py -q
```

Expected before implementation: import failure for `backend.rag.contracts`.

- [ ] **Step 3: Implement contracts**

Create `backend/rag/contracts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal

from backend.shared.filename_utils import dedupe_filenames


RetrievalStrategy = Literal["standard_hybrid", "scoped_hybrid", "layered_split", "dense_fallback", "failed"]


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    context_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "top_k", max(1, int(self.top_k)))
        object.__setattr__(self, "context_files", tuple(dedupe_filenames(list(self.context_files))))


@dataclass(frozen=True)
class RetrievalPlan:
    candidate_k: int
    ce_input_k: int | None = None
    ce_top_n: int | None = None
    root_weight: float | None = None
    same_root_cap: int | None = None
    source: str = "default"
    applied: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_k": self.candidate_k,
            "ce_input_k": self.ce_input_k,
            "ce_top_n": self.ce_top_n,
            "root_weight": self.root_weight,
            "same_root_cap": self.same_root_cap,
            "source": self.source,
            "applied": self.applied,
        }


@dataclass(frozen=True)
class RetrievalBudget:
    deadline_seconds: float | None = None
    remaining_ms: float | None = None
    allow_expanded_retrieval: bool = True
    expanded_candidate_k: int | None = None
    expanded_rerank_input_cap: int | None = None


@dataclass
class RetrievalSession:
    session_id: str
    request_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def cache_key(self, request: RetrievalRequest, budget: RetrievalBudget) -> str:
        payload = {
            "query": request.query,
            "top_k": request.top_k,
            "context_files": request.context_files,
            "deadline_seconds": budget.deadline_seconds,
            "expanded_candidate_k": budget.expanded_candidate_k,
            "expanded_rerank_input_cap": budget.expanded_rerank_input_cap,
        }
        digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return f"retrieval-session:{self.session_id}:{digest}"


@dataclass(frozen=True)
class CandidateSet:
    strategy: RetrievalStrategy
    docs: list[dict[str, Any]]
    plan: RetrievalPlan
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    docs: list[dict[str, Any]]
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"docs": self.docs, "meta": self.meta}
```

- [ ] **Step 4: Run contract tests**

Run:

```powershell
python -m pytest tests/test_rag_contracts.py -q
```

Expected: all tests pass.

### Task 2: Add Structured Stage Errors

**Files:**
- Create: `backend/rag/errors.py`
- Modify: `backend/rag/types.py`
- Test: `tests/test_rag_error_semantics.py`

- [ ] **Step 1: Write error semantics tests**

Add `tests/test_rag_error_semantics.py`:

```python
from backend.rag.errors import StageError, stage_error


def test_stage_error_has_recovery_semantics():
    err = stage_error(
        stage="rerank",
        error=RuntimeError("ce down"),
        error_code="RERANK_FAILED",
        severity="warning",
        fallback_to="ranked_candidates",
        recoverable=True,
        user_visible=False,
    )

    payload = err.as_dict()

    assert payload["stage"] == "rerank"
    assert payload["error_code"] == "RERANK_FAILED"
    assert payload["severity"] == "warning"
    assert payload["fallback_to"] == "ranked_candidates"
    assert payload["recoverable"] is True
    assert payload["user_visible"] is False
    assert "ce down" in payload["error"]


def test_types_stage_error_remains_compatible():
    from backend.rag.types import StageError as CompatStageError

    payload = CompatStageError(stage="hybrid_retrieve", error="down", fallback_to="dense_retrieve").as_dict()

    assert payload["stage"] == "hybrid_retrieve"
    assert payload["fallback_to"] == "dense_retrieve"
    assert payload["severity"] == "error"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_rag_error_semantics.py -q
```

Expected before implementation: import failure for `backend.rag.errors`.

- [ ] **Step 3: Implement structured errors**

Create `backend/rag/errors.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class StageError:
    stage: str
    error: str
    error_code: str = "STAGE_FAILED"
    severity: Severity = "error"
    fallback_to: str | None = None
    recoverable: bool = False
    user_visible: bool = False

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage": self.stage,
            "error": self.error,
            "error_code": self.error_code,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "user_visible": self.user_visible,
        }
        if self.fallback_to:
            payload["fallback_to"] = self.fallback_to
        return payload


def stage_error(
    *,
    stage: str,
    error: Exception | str,
    error_code: str = "STAGE_FAILED",
    severity: Severity = "error",
    fallback_to: str | None = None,
    recoverable: bool = False,
    user_visible: bool = False,
) -> StageError:
    return StageError(
        stage=stage,
        error=str(error),
        error_code=error_code,
        severity=severity,
        fallback_to=fallback_to,
        recoverable=recoverable,
        user_visible=user_visible,
    )
```

Modify `backend/rag/types.py`:

```python
from backend.rag.errors import StageError

__all__ = ["StageError"]
```

- [ ] **Step 4: Run error tests**

Run:

```powershell
python -m pytest tests/test_rag_error_semantics.py -q
```

Expected: all tests pass.

---

## Phase 2: Rerank Path Convergence

### Task 3: Extract Unified Postprocess

**Files:**
- Create: `backend/rag/postprocess.py`
- Modify: `backend/rag/utils.py`
- Test: `tests/test_rag_postprocess_convergence.py`

- [ ] **Step 1: Write tests proving one postprocess path**

Add `tests/test_rag_postprocess_convergence.py`:

```python
from unittest.mock import patch

import backend.rag.utils as rag_utils


def _doc(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "text": chunk_id, "filename": "manual.pdf", "score": 0.9}


def test_standard_and_layered_use_same_postprocess_error_shape():
    docs = [_doc("c1"), _doc("c2")]

    def failing_rerank(query, docs, top_k):
        return docs[:top_k], {
            "rerank_enabled": True,
            "rerank_applied": False,
            "rerank_error": "ce down",
            "rerank_input_count": len(docs),
            "rerank_output_count": top_k,
        }

    with (
        patch("backend.rag.utils._rerank_documents", side_effect=failing_rerank),
        patch("backend.rag.utils._apply_structure_rerank", side_effect=lambda docs, top_k, **kw: (docs[:top_k], {})),
        patch("backend.rag.utils._evaluate_retrieval_confidence", return_value={"fallback_required": False}),
    ):
        standard = rag_utils._finish_retrieval_pipeline(
            query="q",
            search_query="q",
            retrieved=docs,
            top_k=1,
            candidate_k=2,
            timings={},
            stage_errors=[],
            total_start=0.0,
            retrieval_mode="hybrid",
        )
        layered = rag_utils._finish_retrieval_pipeline(
            query="q",
            search_query="q",
            retrieved=docs,
            top_k=1,
            candidate_k=2,
            timings={},
            stage_errors=[],
            total_start=0.0,
            extra_trace={"v3_layers": ["layered_split"]},
            retrieval_mode="layered_split",
        )

    assert standard["meta"]["stage_errors"][0]["error_code"] == "RERANK_FAILED"
    assert layered["meta"]["stage_errors"][0]["error_code"] == "RERANK_FAILED"
    assert standard["meta"]["stage_errors"][0]["fallback_to"] == "ranked_candidates"
    assert layered["meta"]["stage_errors"][0]["fallback_to"] == "ranked_candidates"


def test_postprocess_records_layered_plan_without_changing_error_contract():
    docs = [_doc("c1")]

    with (
        patch("backend.rag.utils._rerank_documents", return_value=(docs, {"rerank_enabled": False, "rerank_applied": False, "rerank_error": None})),
        patch("backend.rag.utils._apply_structure_rerank", side_effect=lambda docs, top_k, **kw: (docs[:top_k], {"structure_rerank_applied": True})),
        patch("backend.rag.utils._evaluate_retrieval_confidence", return_value={"fallback_required": False}),
    ):
        result = rag_utils._finish_retrieval_pipeline(
            query="q",
            search_query="q",
            retrieved=docs,
            top_k=1,
            candidate_k=1,
            timings={},
            stage_errors=[],
            total_start=0.0,
            extra_trace={"retrieval_plan": {"source": "layered_split"}},
            retrieval_mode="layered_split",
        )

    assert result["meta"]["retrieval_plan"]["source"] == "layered_split"
    assert result["meta"]["stage_errors"] == []
```

- [ ] **Step 2: Run convergence tests and verify current behavior**

Run:

```powershell
python -m pytest tests/test_rag_postprocess_convergence.py -q
```

Expected before implementation: first test fails because current `_stage_error()` does not emit `error_code`.

- [ ] **Step 3: Create `postprocess.py`**

Move the shared logic from `backend/rag/utils.py:_finish_retrieval_pipeline()` into `backend/rag/postprocess.py` as `finish_retrieval_pipeline()`. The function must accept dependencies as callables so tests can patch it without importing CrossEncoder:

```python
def finish_retrieval_pipeline(
    *,
    query: str,
    retrieved: list[dict],
    top_k: int,
    candidate_k: int,
    timings: dict[str, float],
    stage_errors: list[dict],
    total_start: float,
    retrieval_mode: str,
    config: dict,
    rerank_documents,
    apply_structure_rerank,
    evaluate_confidence,
    candidate_trace,
    elapsed_ms,
    extra_trace: dict | None = None,
    plan: dict | None = None,
) -> dict:
    stage_start = time.perf_counter()
    reranked, rerank_meta = rerank_documents(query=query, docs=retrieved, top_k=top_k)
    timings["rerank_ms"] = elapsed_ms(stage_start)

    if rerank_meta.get("rerank_error"):
        stage_errors.append(
            stage_error(
                stage="rerank",
                error=str(rerank_meta["rerank_error"]),
                error_code="RERANK_FAILED",
                severity="warning",
                fallback_to="ranked_candidates",
                recoverable=True,
                user_visible=False,
            ).as_dict()
        )

    stage_start = time.perf_counter()
    reranked_docs, structure_meta = apply_structure_rerank(docs=reranked, top_k=top_k)
    timings["structure_rerank_ms"] = elapsed_ms(stage_start)

    stage_start = time.perf_counter()
    confidence_meta = evaluate_confidence(query=query, docs=reranked_docs)
    timings["confidence_ms"] = elapsed_ms(stage_start)
    timings["total_retrieve_ms"] = elapsed_ms(total_start)

    meta = {
        **rerank_meta,
        **structure_meta,
        **confidence_meta,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "stage_errors": stage_errors,
        "timings": timings,
        "retrieval_plan": plan or {},
        "candidate_count_before_rerank": len(retrieved),
        "candidate_count_after_rerank": len(reranked),
        "candidate_count_after_structure_rerank": len(reranked_docs),
        "candidates_before_rerank": candidate_trace(retrieved),
        "candidates_after_rerank": candidate_trace(reranked),
        "candidates_after_structure_rerank": candidate_trace(reranked_docs),
    }
    if extra_trace:
        meta.update(extra_trace)
    return {"docs": reranked_docs, "meta": meta}
```

The implementation must:

- Call `rerank_documents()` exactly once.
- Convert any `rerank_error` into a `stage_error` payload with `error_code="RERANK_FAILED"`, `severity="warning"`, `fallback_to="ranked_candidates"`, `recoverable=True`, and `user_visible=False`.
- Call `apply_structure_rerank()` exactly once.
- Call `evaluate_confidence()` exactly once.
- Add `candidates_before_rerank`, `candidates_after_rerank`, `candidates_after_structure_rerank`.
- Merge `extra_trace` after core meta is created.

- [ ] **Step 4: Keep `_finish_retrieval_pipeline()` as compatibility wrapper**

In `backend/rag/utils.py`, keep `_finish_retrieval_pipeline()` but reduce it to a wrapper that calls `finish_retrieval_pipeline()` with existing config and helper functions. This keeps existing tests and imports stable.

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_rag_postprocess_convergence.py tests/test_rag_observability.py tests/test_layered_rerank.py -q
```

Expected: all tests pass.

### Task 4: Convert Layered To Candidate Strategy

**Files:**
- Modify: `backend/rag/utils.py`
- Modify: `backend/rag/layered_rerank.py`
- Test: `tests/test_rag_postprocess_convergence.py`

- [ ] **Step 1: Add tests that Layered only changes CandidateSet**

Append to `tests/test_rag_postprocess_convergence.py`:

```python
def test_layered_flag_changes_candidate_strategy_not_postprocess(monkeypatch):
    docs = [_doc("c1")]
    calls = {"postprocess": 0}

    def fake_finish(**kwargs):
        calls["postprocess"] += 1
        return {"docs": kwargs["retrieved"], "meta": {"retrieval_mode": kwargs["retrieval_mode"], "stage_errors": []}}

    monkeypatch.setattr(rag_utils, "_LAYERED_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_utils._embedding_service, "get_embeddings", lambda queries: [[0.1, 0.2]])
    monkeypatch.setattr(rag_utils._embedding_service, "get_sparse_embedding", lambda query: {1: 0.5})
    monkeypatch.setattr(rag_utils._milvus_manager, "split_retrieve", lambda *args, **kwargs: docs)
    monkeypatch.setattr(rag_utils, "_finish_retrieval_pipeline", lambda **kwargs: fake_finish(**kwargs))

    result = rag_utils.retrieve_documents("q", top_k=1)

    assert calls["postprocess"] == 1
    assert result["meta"]["retrieval_mode"] in {"layered_split", "hybrid"}
```

- [ ] **Step 2: Make candidate retrieval functions explicit**

In `backend/rag/utils.py`, extract these private helpers:

```python
def _retrieve_layered_candidates(
    search_query: str,
    dense_embedding: list[float],
    sparse_embedding: dict,
    filter_expr: str,
    query_plan: QueryPlan,
    candidate_k: int,
) -> CandidateSet:
    retrieved = _milvus_manager.split_retrieve(
        dense_embedding,
        sparse_embedding,
        dense_top_k=_L0_DENSE_TOP_K,
        sparse_top_k=_L0_SPARSE_TOP_K,
        search_ef=MILVUS_SEARCH_EF,
        sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
        filter_expr=filter_expr,
    )
    plan = RetrievalPlan(candidate_k=candidate_k, source="layered_split")
    trace = {"semantic_query": search_query, "query_plan": query_plan.to_dict()}
    return CandidateSet(strategy="layered_split", docs=retrieved, plan=plan, meta=trace)


def _retrieve_standard_candidates(
    search_query: str,
    dense_embedding: list[float],
    sparse_embedding: dict,
    filter_expr: str,
    candidate_k: int,
) -> CandidateSet:
    retrieved = _milvus_manager.hybrid_retrieve(
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=candidate_k,
        rrf_k=MILVUS_RRF_K,
        search_ef=MILVUS_SEARCH_EF,
        sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
        filter_expr=filter_expr,
    )
    plan = RetrievalPlan(candidate_k=candidate_k, source="standard_hybrid")
    trace = {"semantic_query": search_query}
    return CandidateSet(strategy="standard_hybrid", docs=retrieved, plan=plan, meta=trace)


def _retrieve_scoped_candidates(
    search_query: str,
    scoped_docs: list[dict],
    global_docs: list[dict],
    candidate_k: int,
) -> CandidateSet:
    retrieved = _retrieval_weighted_rrf_merge(
        [(scoped_docs, 1.0 - DOC_SCOPE_GLOBAL_RESERVE_WEIGHT), (global_docs, DOC_SCOPE_GLOBAL_RESERVE_WEIGHT)],
        rrf_k=MILVUS_RRF_K,
    )
    plan = RetrievalPlan(candidate_k=candidate_k, source="scoped_hybrid")
    trace = {"semantic_query": search_query, "scoped_candidate_count": len(scoped_docs), "global_candidate_count": len(global_docs)}
    return CandidateSet(strategy="scoped_hybrid", docs=retrieved, plan=plan, meta=trace)
```

Each helper may still live in `utils.py` for the first commit. Do not move all retrieval code at once.

- [ ] **Step 3: Remove postprocess decisions from Layered branch**

The `if _LAYERED_RERANK_ENABLED` branches may set:

- candidate docs
- `retrieval_mode`
- `retrieval_plan`
- `v3_layers`
- `scope_filter_applied`
- CE input/top N suggestions

The branch must not set final rerank meta, confidence meta, or final candidate traces. Those belong to unified postprocess.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_rag_postprocess_convergence.py tests/test_layered_rerank.py tests/test_rag_observability.py -q
```

Expected: all tests pass.

---

## Phase 3: Fallback Latency And Request Session

### Task 5: Add Request-Level Retrieval Session

**Files:**
- Modify: `backend/rag/contracts.py`
- Modify: `backend/rag/pipeline.py`
- Modify: `backend/rag/utils.py`
- Test: `tests/test_rag_fallback_budget.py`

- [ ] **Step 1: Write fallback session tests**

Add `tests/test_rag_fallback_budget.py`:

```python
import time
from unittest.mock import patch

import backend.rag.pipeline as rag_pipeline


def test_retrieve_expanded_skips_when_expanded_query_matches_initial():
    initial_docs = [{"chunk_id": "c1", "text": "initial", "filename": "a.pdf"}]
    state = {
        "question": "same query",
        "docs": initial_docs,
        "context": "initial",
        "context_files": [],
        "expansion_type": "step_back",
        "expanded_query": "same query",
        "rag_trace": {"query": "same query"},
        "fallback_deadline": time.perf_counter() + 5,
    }

    with patch("backend.rag.pipeline.retrieve_documents", side_effect=AssertionError("second retrieval should be skipped")):
        result = rag_pipeline.retrieve_expanded(state)

    assert result["docs"] == initial_docs
    assert result["rag_trace"]["expanded_retrieval_skipped_reason"] == "expanded_query_equivalent"
    assert result["rag_trace"]["retrieval_stage"] == "initial"


def test_retrieve_expanded_skips_when_deadline_cannot_cover_second_retrieval():
    initial_docs = [{"chunk_id": "c1", "text": "initial", "filename": "a.pdf"}]
    state = {
        "question": "q",
        "docs": initial_docs,
        "context": "initial",
        "context_files": [],
        "expansion_type": "step_back",
        "expanded_query": "expanded q",
        "rag_trace": {},
        "fallback_deadline": time.perf_counter() + 0.001,
    }

    with patch("backend.rag.pipeline.retrieve_documents", side_effect=AssertionError("second retrieval should be skipped")):
        result = rag_pipeline.retrieve_expanded(state)

    assert result["docs"] == initial_docs
    assert result["rag_trace"]["expanded_retrieval_skipped_reason"] == "fallback_deadline_exhausted"
    assert result["rag_trace"]["fallback_returned_initial"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_rag_fallback_budget.py -q
```

Expected before implementation: first test calls `retrieve_documents()` and fails.

- [ ] **Step 3: Add query equivalence helper**

In `backend/rag/pipeline.py`:

```python
def _normalize_retrieval_query(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _should_skip_expanded_retrieval(state: RAGState, rag_trace: dict, fallback_deadline: float) -> str | None:
    if time.perf_counter() >= fallback_deadline:
        return "fallback_deadline_exhausted"
    original = _normalize_retrieval_query(state.get("question"))
    expanded = _normalize_retrieval_query(state.get("expanded_query"))
    if not expanded:
        return "expanded_query_empty"
    if expanded == original:
        return "expanded_query_equivalent"
    return None
```

- [ ] **Step 4: Use skip helper at start of `retrieve_expanded()`**

Before starting expanded retrieval jobs:

```python
skip_reason = _should_skip_expanded_retrieval(state, rag_trace, fallback_deadline)
if skip_reason:
    rag_trace["expanded_retrieval_skipped_reason"] = skip_reason
    rag_trace["fallback_returned_initial"] = True
    return _fallback_to_initial_retrieval(state, rag_trace, expanded_start)
```

- [ ] **Step 5: Add session id to RAGState and trace**

Generate a session id once in `run_rag_graph()`:

```python
retrieval_session_id = f"rag-{int(graph_start * 1000000)}"
```

Store it in `RAGState` and `rag_trace["retrieval_session_id"]`. Pass it through `retrieve_initial()` and `retrieve_expanded()` as an optional `session_id` argument to `retrieve_documents()` after Task 6 changes its signature.

- [ ] **Step 6: Run fallback budget tests**

Run:

```powershell
python -m pytest tests/test_rag_fallback_budget.py tests/test_rag_pipeline.py -q
```

Expected: all tests pass.

### Task 6: Add Expanded Retrieval Budget

**Files:**
- Modify: `backend/rag/utils.py`
- Modify: `backend/rag/pipeline.py`
- Test: `tests/test_rag_fallback_budget.py`

- [ ] **Step 1: Add budget propagation test**

Append to `tests/test_rag_fallback_budget.py`:

```python
def test_retrieve_expanded_passes_budget_to_second_retrieval():
    state = {
        "question": "q",
        "docs": [],
        "context": "",
        "context_files": ["manual.pdf"],
        "expansion_type": "step_back",
        "expanded_query": "expanded q",
        "rag_trace": {},
        "fallback_deadline": time.perf_counter() + 5,
    }
    captured = {}

    def fake_retrieve(query, top_k=5, context_files=None, retrieval_budget=None, retrieval_session_id=None):
        captured["query"] = query
        captured["budget"] = retrieval_budget
        captured["session_id"] = retrieval_session_id
        return {"docs": [{"chunk_id": "c1", "text": "expanded", "filename": "manual.pdf"}], "meta": {"timings": {}, "stage_errors": []}}

    with patch("backend.rag.pipeline.retrieve_documents", side_effect=fake_retrieve):
        result = rag_pipeline.retrieve_expanded(state)

    assert captured["query"] == "expanded q"
    assert captured["budget"] is not None
    assert captured["budget"].allow_expanded_retrieval is True
    assert result["rag_trace"]["retrieval_stage"] == "expanded"
```

- [ ] **Step 2: Extend `retrieve_documents()` signature compatibly**

In `backend/rag/utils.py`:

```python
def retrieve_documents(
    query: str,
    top_k: int = 5,
    context_files: list[str] | None = None,
    retrieval_budget: RetrievalBudget | None = None,
    retrieval_session_id: str | None = None,
) -> Dict[str, Any]:
    request = RetrievalRequest(query=query, top_k=top_k, context_files=tuple(context_files or ()))
    budget = retrieval_budget or RetrievalBudget()
    result = _retrieve_documents_with_contracts(
        request=request,
        budget=budget,
        retrieval_session_id=retrieval_session_id,
    )
    return result.as_dict()
```

Existing callers remain valid.

- [ ] **Step 3: Apply budget only to expanded retrieval**

If `retrieval_budget.expanded_candidate_k` is set, use it instead of `_effective_candidate_k(top_k)`. If `expanded_rerank_input_cap` is set, pass it through plan metadata and cap CE input in `_rerank_documents()` via runtime config. Do not change default behavior when budget fields are absent.

- [ ] **Step 4: Record budget trace**

Add to meta:

```python
"retrieval_session_id": retrieval_session_id,
"retrieval_budget": {
    "allow_expanded_retrieval": retrieval_budget.allow_expanded_retrieval,
    "expanded_candidate_k": retrieval_budget.expanded_candidate_k,
    "expanded_rerank_input_cap": retrieval_budget.expanded_rerank_input_cap,
} if retrieval_budget else None,
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_rag_fallback_budget.py tests/test_rag_observability.py -q
```

Expected: all tests pass.

---

## Phase 4: Agent And Direct Path Consistency

### Task 7: Introduce EvidenceBundle

**Files:**
- Create: `backend/chat/evidence.py`
- Modify: `backend/chat/tools.py`
- Test: `tests/test_chat_evidence_bundle.py`

- [ ] **Step 1: Write EvidenceBundle tests**

Add `tests/test_chat_evidence_bundle.py`:

```python
from backend.chat.evidence import EvidenceBundle, format_evidence_for_model, should_force_knowledge_retrieval


def test_format_evidence_for_model_is_shared_for_tools_and_attachments():
    bundle = EvidenceBundle(
        docs=[{"filename": "manual.pdf", "page_number": 2, "text": "alpha"}],
        context_files=("manual.pdf",),
        rag_trace={"retrieval_stage": "initial"},
    )

    text = format_evidence_for_model(bundle)

    assert "manual.pdf" in text
    assert "Page 2" in text
    assert "alpha" in text


def test_force_retrieval_for_context_files():
    assert should_force_knowledge_retrieval("hello", ("manual.pdf",)) is True


def test_force_retrieval_for_explicit_document_question():
    assert should_force_knowledge_retrieval("根据文档说明保修期", ()) is True
    assert should_force_knowledge_retrieval("写一首短诗", ()) is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_chat_evidence_bundle.py -q
```

Expected before implementation: import failure for `backend.chat.evidence`.

- [ ] **Step 3: Implement EvidenceBundle**

Create `backend/chat/evidence.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from backend.shared.filename_utils import dedupe_filenames


_DOC_INTENT_RE = re.compile(r"(根据文档|知识库|政策|产品说明|规格|合同|资料|文档|manual|policy|spec)", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceBundle:
    docs: list[dict[str, Any]]
    context_files: tuple[str, ...] = ()
    rag_trace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rag_result(cls, rag_result: dict, context_files: list[str] | None = None) -> "EvidenceBundle":
        return cls(
            docs=list(rag_result.get("docs") or []),
            context_files=tuple(dedupe_filenames(context_files)),
            rag_trace=dict(rag_result.get("rag_trace") or {}),
        )


def should_force_knowledge_retrieval(user_text: str, context_files: tuple[str, ...] | list[str]) -> bool:
    if context_files:
        return True
    return bool(_DOC_INTENT_RE.search(user_text or ""))


def format_evidence_for_model(bundle: EvidenceBundle) -> str:
    if not bundle.docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, doc in enumerate(bundle.docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text") or doc.get("retrieval_text") or ""
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
```

- [ ] **Step 4: Update `search_knowledge_base()` to use shared formatter**

In `backend/chat/tools.py`, replace local formatting loop with:

```python
from backend.chat.evidence import EvidenceBundle, format_evidence_for_model

bundle = EvidenceBundle.from_rag_result(rag_result, context_files=get_rag_context_files())
if bundle.rag_trace:
    _set_last_rag_context({"rag_trace": bundle.rag_trace})
return format_evidence_for_model(bundle)
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_chat_evidence_bundle.py -q
```

Expected: all tests pass.

### Task 8: Use EvidenceBundle In Agent Direct Path

**Files:**
- Modify: `backend/chat/agent.py`
- Test: `tests/test_chat_evidence_bundle.py`

- [ ] **Step 1: Add tests for consistent forced retrieval**

Append to `tests/test_chat_evidence_bundle.py`:

```python
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage

import backend.chat.agent as chat_agent


def test_retrieved_context_instruction_uses_shared_evidence_text():
    messages = [HumanMessage(content="根据文档回答")]
    instruction_messages = chat_agent._with_retrieved_context_instruction(
        messages,
        ["manual.pdf"],
        "Retrieved Chunks:\n[1] manual.pdf (Page 1):\nalpha",
    )

    assert isinstance(instruction_messages[-2], SystemMessage)
    assert "Retrieved Chunks" in instruction_messages[-2].content
    assert "manual.pdf" in instruction_messages[-2].content


def test_should_force_knowledge_retrieval_can_route_non_attachment_doc_question():
    from backend.chat.evidence import should_force_knowledge_retrieval

    assert should_force_knowledge_retrieval("按照政策说明", []) is True
```

- [ ] **Step 2: Route explicit document questions through deterministic evidence path**

In `chat_with_agent()`:

```python
from backend.chat.evidence import EvidenceBundle, format_evidence_for_model, should_force_knowledge_retrieval

force_retrieval = should_force_knowledge_retrieval(user_text, context_files)
if force_retrieval:
    rag_result = _retrieve_attached_context(user_text, context_files)
    bundle = EvidenceBundle.from_rag_result(rag_result, context_files=context_files)
    attached_rag_trace = bundle.rag_trace
    agent_messages = _with_retrieved_context_instruction(
        messages,
        context_files,
        format_evidence_for_model(bundle),
    )
else:
    agent_messages = _with_context_file_instruction(messages, context_files)
```

When `force_retrieval` is true, call `model_instance.invoke(agent_messages)` to keep deterministic behavior. When false, keep `agent_instance.invoke()`.

- [ ] **Step 3: Mirror the same logic in `chat_with_agent_stream()`**

Apply the same `force_retrieval` branch before `_agent_worker()`. If `force_retrieval` is true, stream from `model_instance.astream(agent_messages)`. If false, stream from `agent_instance.astream(...)`.

- [ ] **Step 4: Run chat evidence tests**

Run:

```powershell
python -m pytest tests/test_chat_evidence_bundle.py tests/test_application_entrypoints.py -q
```

Expected: all tests pass.

---

## Phase 5: Embedding And Retrieval Failure Fallbacks

### Task 9: Make Embedding Failure Explicit And Recoverable Where Possible

**Files:**
- Modify: `backend/rag/utils.py`
- Modify: `backend/infra/vector_store/milvus_client.py`
- Test: `tests/test_rag_error_semantics.py`

- [ ] **Step 1: Add tests for embedding fallback trace**

Append to `tests/test_rag_error_semantics.py`:

```python
from unittest.mock import patch

import backend.rag.utils as rag_utils


def test_embedding_failure_returns_failed_trace_with_structured_error():
    with (
        patch.object(rag_utils._embedding_service, "get_embeddings", side_effect=RuntimeError("dense embed down")),
        patch.object(rag_utils._embedding_service, "get_sparse_embedding", side_effect=RuntimeError("sparse embed down")),
    ):
        result = rag_utils.retrieve_documents("query", top_k=1)

    assert result["docs"] == []
    assert result["meta"]["retrieval_mode"] == "failed"
    assert result["meta"]["stage_errors"]
    assert result["meta"]["stage_errors"][0]["error_code"] in {"EMBEDDING_FAILED", "STAGE_FAILED"}
    assert result["meta"]["stage_errors"][0]["recoverable"] is False


def test_hybrid_failure_dense_fallback_is_recoverable():
    docs = [{"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9}]

    with (
        patch.object(rag_utils, "RERANK_MODEL", ""),
        patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
        patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
        patch.object(rag_utils._milvus_manager, "hybrid_retrieve", side_effect=RuntimeError("hybrid down")),
        patch.object(rag_utils._milvus_manager, "dense_retrieve", return_value=docs),
        patch("backend.rag.utils._apply_structure_rerank", side_effect=lambda docs, top_k, **kw: (docs[:top_k], {})),
        patch("backend.rag.utils._evaluate_retrieval_confidence", return_value={"fallback_required": False}),
    ):
        result = rag_utils.retrieve_documents("query", top_k=1)

    err = result["meta"]["stage_errors"][0]
    assert err["stage"] == "hybrid_retrieve"
    assert err["fallback_to"] == "dense_retrieve"
    assert err["recoverable"] is True
```

- [ ] **Step 2: Replace `_stage_error()` with structured helper**

In `backend/rag/utils.py`, update `_stage_error()`:

```python
def _stage_error(
    stage: str,
    error: str,
    fallback_to: str | None = None,
    *,
    error_code: str = "STAGE_FAILED",
    severity: str = "error",
    recoverable: bool | None = None,
    user_visible: bool = False,
) -> Dict[str, object]:
    from backend.rag.errors import stage_error

    return stage_error(
        stage=stage,
        error=error,
        error_code=error_code,
        severity=severity,
        fallback_to=fallback_to,
        recoverable=bool(fallback_to) if recoverable is None else recoverable,
        user_visible=user_visible,
    ).as_dict()
```

- [ ] **Step 3: Use error codes at critical stages**

Update calls:

- embedding failure: `error_code="EMBEDDING_FAILED"`, `recoverable=False` unless a fallback path succeeds.
- hybrid failure with dense fallback: `error_code="HYBRID_RETRIEVE_FAILED"`, `fallback_to="dense_retrieve"`, `recoverable=True`.
- rerank failure: `error_code="RERANK_FAILED"`, `fallback_to="ranked_candidates"`, `severity="warning"`, `recoverable=True`.
- final retrieve failure: `error_code="RETRIEVE_FAILED"`, `recoverable=False`, `user_visible=True`.

- [ ] **Step 4: Add limited metadata fallback for attached files**

When both dense and sparse embedding fail but `context_files` is present, call `retrieve_context_documents(context_files, limit_per_file=8)` and return:

```python
"retrieval_mode": "attached_context_fallback",
"stage_errors": [
    {
        "stage": "embedding",
        "error_code": "EMBEDDING_FAILED",
        "fallback_to": "attached_context_fallback",
        "recoverable": True,
        "user_visible": False,
    }
]
```

Do not claim keyword fallback unless a real keyword index exists.

- [ ] **Step 5: Run error tests**

Run:

```powershell
python -m pytest tests/test_rag_error_semantics.py tests/test_rag_observability.py -q
```

Expected: all tests pass.

---

## Phase 6: Trace Schema And Compatibility

### Task 10: Extend Public RagTrace Schema

**Files:**
- Modify: `backend/contracts/schemas.py`
- Test: `tests/test_api_routes.py`
- Test: `tests/test_rag_trace.py`

- [ ] **Step 1: Add schema test**

Append to `tests/test_rag_trace.py`:

```python
from backend.contracts.schemas import RagTrace


def test_rag_trace_accepts_structured_stage_errors():
    trace = RagTrace(
        tool_used=True,
        tool_name="search_knowledge_base",
        stage_errors=[
            {
                "stage": "rerank",
                "error": "ce down",
                "error_code": "RERANK_FAILED",
                "severity": "warning",
                "fallback_to": "ranked_candidates",
                "recoverable": True,
                "user_visible": False,
            }
        ],
    )

    assert trace.stage_errors[0]["error_code"] == "RERANK_FAILED"
```

- [ ] **Step 2: Add schema fields**

In `backend/contracts/schemas.py`, extend `RagTrace`:

```python
    stage_errors: Optional[List[dict]] = None
    timings: Optional[dict] = None
    retrieval_session_id: Optional[str] = None
    retrieval_budget: Optional[dict] = None
    expanded_retrieval_skipped_reason: Optional[str] = None
    evidence_source: Optional[str] = None
```

- [ ] **Step 3: Run schema and API tests**

Run:

```powershell
python -m pytest tests/test_rag_trace.py tests/test_api_routes.py -q
```

Expected: all tests pass.

---

## Phase 7: Verification Matrix

### Task 11: Run Targeted Regression Suite

**Files:**
- No code changes.

- [ ] **Step 1: Run core RAG unit tests**

Run:

```powershell
python -m pytest tests/test_rag_contracts.py tests/test_rag_error_semantics.py tests/test_rag_postprocess_convergence.py tests/test_rag_fallback_budget.py tests/test_chat_evidence_bundle.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run existing RAG regression tests**

Run:

```powershell
python -m pytest tests/test_rag_observability.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_layered_rerank.py tests/test_rerank_pair_enrichment.py tests/test_scoped_global_rrf.py tests/test_rag_trace.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run import and API smoke tests**

Run:

```powershell
python -m pytest tests/test_import_contract.py tests/test_application_entrypoints.py tests/test_api_routes.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run static compile check**

Run:

```powershell
python -m compileall backend tests
```

Expected: no syntax errors.

### Task 12: Manual Trace Inspection

**Files:**
- No code changes.

- [ ] **Step 1: Inspect standard retrieval trace**

Run a local call with `LAYERED_RERANK_ENABLED=false` and a document query. Verify trace contains:

```text
retrieval_mode
retrieval_plan
rerank_applied
candidates_before_rerank
candidates_after_rerank
candidates_after_structure_rerank
stage_errors
timings
```

- [ ] **Step 2: Inspect layered retrieval trace**

Run the same local call with `LAYERED_RERANK_ENABLED=true`. Verify:

```text
retrieval_plan.source = layered_split
postprocess field names match standard path
stage_errors item shape matches standard path
```

- [ ] **Step 3: Inspect fallback skip**

Force `expanded_query` equal to original query in a unit or local graph call. Verify:

```text
expanded_retrieval_skipped_reason = expanded_query_equivalent
fallback_returned_initial = true
retrieval_stage = initial
```

## Commit Strategy

Use small commits in this order:

1. `Add typed RAG retrieval contracts`
2. `Structure RAG stage error semantics`
3. `Unify RAG postprocess after candidate retrieval`
4. `Treat layered rerank as a candidate strategy`
5. `Budget expanded RAG fallback retrieval`
6. `Share evidence formatting across chat paths`
7. `Expose structured RAG trace fields`

Each commit must use the repository Lore commit protocol with `Constraint`, `Rejected`, `Confidence`, `Scope-risk`, `Tested`, and `Not-tested` trailers where useful.

## Rollback Plan

- If postprocess convergence changes result ordering unexpectedly, revert commits 3 and 4 only; contracts and error schema can remain.
- If fallback budget reduces answer quality, set the budget fields to disabled defaults and keep trace/schema changes.
- If deterministic non-attachment document retrieval causes over-retrieval, narrow `_DOC_INTENT_RE` and keep EvidenceBundle sharing for attachments/tools.
- If public API schema changes break clients, keep internal trace fields but make public `RagTrace` accept `extra="allow"` or reduce newly exposed fields to optional dicts.

## Remaining Risks

- The plan does not implement full keyword fallback because the current stack has sparse vector retrieval but no standalone full-text index contract.
- Layered candidate quality still depends on existing L1/L2 heuristics; convergence removes duplicated behavior but does not by itself prove quality improvement.
- Agent path consistency improves for explicit document questions and attachments; purely implicit private-knowledge questions may still rely on Agent tool choice unless a separate intent classifier is introduced.
- Latency gains from fallback budgeting depend on production query distribution and deadline values; gold-set and online trace review are still required before changing defaults aggressively.
