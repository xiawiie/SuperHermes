from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Literal, TypedDict, List, Optional
import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from rag_utils import retrieve_context_documents, retrieve_documents, step_back_expand, generate_hypothetical_document
from tools import emit_rag_step

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
GRADE_MODEL = os.getenv("GRADE_MODEL", "gpt-4.1")
RAG_FALLBACK_TIMEOUT_SECONDS = float(os.getenv("RAG_FALLBACK_TIMEOUT_SECONDS", "6"))
RAG_FALLBACK_WORKERS = int(os.getenv("RAG_FALLBACK_WORKERS", "4"))
RAG_FALLBACK_ENABLED = _env_bool("RAG_FALLBACK_ENABLED", False)
RAG_FALLBACK_USE_FAST_MODEL = _env_bool("RAG_FALLBACK_USE_FAST_MODEL", True)

FAST_MODEL = os.getenv("FAST_MODEL")
FAST_MODEL_ENABLED = RAG_FALLBACK_USE_FAST_MODEL and bool(FAST_MODEL) and FAST_MODEL != MODEL

_grader_model = None
_router_model = None
_fast_model = None
_fallback_executor = ThreadPoolExecutor(max_workers=max(1, RAG_FALLBACK_WORKERS), thread_name_prefix="rag-fallback")


def _get_fallback_model_name() -> str:
    """Return the model name used for fallback LLM calls, for tracing."""
    if FAST_MODEL_ENABLED:
        return FAST_MODEL
    return GRADE_MODEL or MODEL or "unknown"


def _get_fast_model():
    """Lazily initialize the FAST_MODEL for fallback LLM calls."""
    global _fast_model
    if not FAST_MODEL_ENABLED or not API_KEY:
        return None
    if _fast_model is None:
        _fast_model = init_chat_model(
            model=FAST_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _fast_model


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _trace_timings(rag_trace: dict) -> dict:
    timings = dict(rag_trace.get("timings") or {})
    rag_trace["timings"] = timings
    return timings


def _trace_stage_errors(rag_trace: dict) -> list:
    stage_errors = list(rag_trace.get("stage_errors") or [])
    rag_trace["stage_errors"] = stage_errors
    return stage_errors


def _append_stage_error(rag_trace: dict, stage: str, error: str, fallback_to: str | None = None):
    item = {"stage": stage, "error": error}
    if fallback_to:
        item["fallback_to"] = fallback_to
    _trace_stage_errors(rag_trace).append(item)


def _prefixed_stage_errors(prefix: str, errors: list[dict]) -> list[dict]:
    prefixed = []
    for item in errors or []:
        next_item = dict(item)
        next_item["stage"] = f"{prefix}_{next_item.get('stage', 'unknown')}"
        prefixed.append(next_item)
    return prefixed


def _fallback_deadline(start: float) -> float:
    return start + max(0.001, RAG_FALLBACK_TIMEOUT_SECONDS)


def _remaining_fallback_seconds(deadline: float) -> float:
    return max(0.001, deadline - time.perf_counter())


def _submit_with_context(fn: Callable[[], object]):
    return _fallback_executor.submit(fn)


def _await_with_deadline(future, deadline: float, rag_trace: dict, stage: str, fallback_to: str):
    try:
        return future.result(timeout=_remaining_fallback_seconds(deadline))
    except TimeoutError:
        future.cancel()
        rag_trace["fallback_timed_out"] = True
        rag_trace["fallback_returned_initial"] = True
        _append_stage_error(rag_trace, stage, f"timed out after {RAG_FALLBACK_TIMEOUT_SECONDS:.3f}s", fallback_to)
        return None
    except Exception as exc:
        _append_stage_error(rag_trace, stage, str(exc), fallback_to)
        return None


def _get_grader_model():
    global _grader_model
    if FAST_MODEL_ENABLED:
        return _get_fast_model()
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _grader_model


def _get_router_model():
    global _router_model
    if FAST_MODEL_ENABLED:
        return _get_fast_model()
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _router_model


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Return JSON only, for example {{\"binary_score\":\"yes\"}} or {{\"binary_score\":\"no\"}}. "
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """Choose a query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    context_files: List[str]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    fallback_started_at: Optional[float]
    fallback_deadline: Optional[float]
    rag_trace: Optional[dict]


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)


def _dedupe_docs(docs: List[dict]) -> List[dict]:
    deduped = []
    seen = set()
    for item in docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _fallback_to_initial_retrieval(state: RAGState, rag_trace: dict, expanded_start: float) -> RAGState:
    docs = state.get("docs") or []
    context = state.get("context") or _format_docs(docs)
    timings = _trace_timings(rag_trace)
    timings["expanded_retrieve_ms"] = _elapsed_ms(expanded_start)
    rag_trace.update({
        "retrieved_chunks": docs,
        "expanded_retrieved_chunks": [],
        "retrieval_stage": "initial",
        "fallback_timed_out": True,
        "fallback_returned_initial": True,
        "context_chars": len(context),
        "retrieved_chunk_count": len(docs),
        "final_context_chunk_count": len(docs),
    })
    return {"docs": docs, "context": context, "rag_trace": rag_trace}


def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]
    context_files = state.get("context_files") or []
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
    retrieved = retrieve_documents(query, top_k=5, context_files=context_files)
    results = retrieved.get("docs", [])
    attached_docs = []
    attached_meta = {}
    if context_files:
        attached = retrieve_context_documents(context_files, limit_per_file=8)
        attached_docs = attached.get("docs", [])
        attached_meta = attached.get("meta", {})
        results = _dedupe_docs(attached_docs + results)
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
    retrieve_timings = dict(retrieve_meta.get("timings") or {})
    retrieve_stage_errors = list(retrieve_meta.get("stage_errors") or [])
    emit_rag_step(
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    emit_rag_step("✅", f"检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "attached_context_chunks": attached_docs,
        "context_files": context_files,
        "retrieval_stage": "initial",
        "rerank_enabled": retrieve_meta.get("rerank_enabled"),
        "rerank_applied": retrieve_meta.get("rerank_applied"),
        "rerank_model": retrieve_meta.get("rerank_model"),
        "rerank_endpoint": retrieve_meta.get("rerank_endpoint"),
        "rerank_error": retrieve_meta.get("rerank_error"),
        "rerank_input_count": retrieve_meta.get("rerank_input_count"),
        "rerank_output_count": retrieve_meta.get("rerank_output_count"),
        "rerank_input_cap": retrieve_meta.get("rerank_input_cap"),
        "rerank_input_device_tier": retrieve_meta.get("rerank_input_device_tier"),
        "rerank_cache_enabled": retrieve_meta.get("rerank_cache_enabled"),
        "rerank_cache_hit": retrieve_meta.get("rerank_cache_hit"),
        "retrieval_mode": retrieve_meta.get("retrieval_mode"),
        "candidate_k": retrieve_meta.get("candidate_k"),
        "leaf_retrieve_level": retrieve_meta.get("leaf_retrieve_level"),
        "auto_merge_enabled": retrieve_meta.get("auto_merge_enabled"),
        "auto_merge_applied": retrieve_meta.get("auto_merge_applied"),
        "auto_merge_threshold": retrieve_meta.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": retrieve_meta.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": retrieve_meta.get("auto_merge_steps"),
        "structure_rerank_enabled": retrieve_meta.get("structure_rerank_enabled"),
        "structure_rerank_applied": retrieve_meta.get("structure_rerank_applied"),
        "structure_rerank_root_weight": retrieve_meta.get("structure_rerank_root_weight"),
        "same_root_cap": retrieve_meta.get("same_root_cap"),
        "dominant_root_id": retrieve_meta.get("dominant_root_id"),
        "dominant_root_share": retrieve_meta.get("dominant_root_share"),
        "dominant_root_support": retrieve_meta.get("dominant_root_support"),
        "confidence_gate_enabled": retrieve_meta.get("confidence_gate_enabled"),
        "fallback_required": retrieve_meta.get("fallback_required"),
        "confidence_reasons": retrieve_meta.get("confidence_reasons"),
        "top_margin": retrieve_meta.get("top_margin"),
        "top_score": retrieve_meta.get("top_score"),
        "anchor_match": retrieve_meta.get("anchor_match"),
        "query_anchors": retrieve_meta.get("query_anchors"),
        "candidates_before_rerank": retrieve_meta.get("candidates_before_rerank"),
        "candidates_after_rerank": retrieve_meta.get("candidates_after_rerank"),
        "candidates_after_structure_rerank": retrieve_meta.get("candidates_after_structure_rerank"),
        "attached_context_count": attached_meta.get("attached_context_count", 0),
        "timings": retrieve_timings,
        "stage_errors": retrieve_stage_errors,
        "context_chars": len(context),
        "retrieved_chunk_count": len(results),
        "final_context_chunk_count": len(results),
    }
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    stage_start = time.perf_counter()
    rag_trace = state.get("rag_trace", {}) or {}

    # When fallback is disabled, short-circuit: always go to generate_answer
    fallback_required_raw = rag_trace.get("fallback_required")
    if not RAG_FALLBACK_ENABLED:
        rag_trace.update({
            "grade_score": "skipped_fallback_disabled",
            "grade_route": "generate_answer",
            "rewrite_needed": False,
            "fallback_required_raw": fallback_required_raw,
            "fallback_executed": False,
            "fallback_disabled": bool(fallback_required_raw),
            "graph_path": "linear_initial_only",
            "fallback_llm_model": _get_fallback_model_name(),
        })
        _trace_timings(rag_trace)["grader_ms"] = _elapsed_ms(stage_start)
        return {"route": "generate_answer", "rag_trace": rag_trace}

    if rag_trace.get("fallback_required") is False:
        rag_trace.update({
            "grade_score": "skipped_fast_path",
            "grade_route": "generate_answer",
            "rewrite_needed": False,
            "fallback_required_raw": fallback_required_raw,
            "fallback_executed": False,
            "fallback_disabled": False,
            "fallback_llm_model": _get_fallback_model_name(),
        })
        _trace_timings(rag_trace)["grader_ms"] = _elapsed_ms(stage_start)
        return {"route": "generate_answer", "rag_trace": rag_trace}
    if rag_trace.get("fallback_required") is True:
        rag_trace.update({
            "grade_score": "fallback_triggered",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
            "fallback_required_raw": fallback_required_raw,
            "fallback_executed": True,
            "fallback_disabled": False,
            "fallback_llm_model": _get_fallback_model_name(),
        })
        _trace_timings(rag_trace)["grader_ms"] = _elapsed_ms(stage_start)
        return {"route": "rewrite_question", "rag_trace": rag_trace}

    grader = _get_grader_model()
    emit_rag_step("📊", "正在评估文档相关性...")
    if not grader:
        grade_update = {
            "grade_score": "unknown",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        _trace_timings(rag_trace)["grader_ms"] = _elapsed_ms(stage_start)
        return {"route": "rewrite_question", "rag_trace": rag_trace}
    question = state["question"]
    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    try:
        response = grader.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = (response.binary_score or "").strip().lower()
        grade_error = None
    except Exception as exc:
        score = "unknown"
        grade_error = str(exc)
    route = "generate_answer" if score == "yes" else "rewrite_question"
    if grade_error and context:
        route = "generate_answer"
    if route == "generate_answer":
        emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score}")
    grade_update = {
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
        "fallback_required_raw": fallback_required_raw,
        "fallback_executed": route == "rewrite_question",
        "fallback_disabled": False,
        "fallback_llm_model": _get_fallback_model_name(),
    }
    if grade_error:
        grade_update["grade_error"] = grade_error
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    if grade_error:
        _append_stage_error(rag_trace, "grade_documents", grade_error, "generate_answer" if context else "rewrite_question")
    _trace_timings(rag_trace)["grader_ms"] = _elapsed_ms(stage_start)
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    rewrite_start = time.perf_counter()
    fallback_started_at = float(state.get("fallback_started_at") or rewrite_start)
    fallback_deadline = float(state.get("fallback_deadline") or _fallback_deadline(fallback_started_at))
    question = state["question"]
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace["fallback_timeout_seconds"] = RAG_FALLBACK_TIMEOUT_SECONDS
    emit_rag_step("✏️", "正在重写查询...")
    router = _get_router_model()
    strategy = "step_back"
    router_ms = 0.0
    router_error = None
    if router:
        prompt = (
            "请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
            "Return JSON only, for example {\"strategy\":\"step_back\"}.\n"
            f"用户问题：{question}"
        )
        router_start = time.perf_counter()

        def _invoke_router():
            return router.with_structured_output(RewriteStrategy).invoke(
                [{"role": "user", "content": prompt}]
            )

        decision = _await_with_deadline(
            _submit_with_context(_invoke_router),
            fallback_deadline,
            rag_trace,
            "rewrite_router",
            "initial_retrieval",
        )
        router_ms = _elapsed_ms(router_start)
        if decision is None:
            if rag_trace.get("fallback_timed_out"):
                strategy = "timeout"
            else:
                router_error = "rewrite_router_failed"
                strategy = "step_back"
        else:
            strategy = decision.strategy

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""
    stepback_llm_ms = 0.0
    hyde_llm_ms = 0.0

    futures = {}
    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        stepback_start = time.perf_counter()
        futures["step_back"] = (stepback_start, _submit_with_context(lambda: step_back_expand(question)))

    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE 假设性文档生成中...")
        hyde_start = time.perf_counter()
        futures["hyde"] = (hyde_start, _submit_with_context(lambda: generate_hypothetical_document(question)))

    if "step_back" in futures:
        start, future = futures["step_back"]
        step_back = _await_with_deadline(future, fallback_deadline, rag_trace, "stepback_llm", "initial_retrieval")
        stepback_llm_ms = _elapsed_ms(start)
        if isinstance(step_back, dict):
            step_back_question = step_back.get("step_back_question", "")
            step_back_answer = step_back.get("step_back_answer", "")
            expanded_query = step_back.get("expanded_query", question)

    if "hyde" in futures:
        start, future = futures["hyde"]
        hyde_doc = _await_with_deadline(future, fallback_deadline, rag_trace, "hyde_llm", "initial_retrieval")
        hyde_llm_ms = _elapsed_ms(start)
        if isinstance(hyde_doc, str):
            hypothetical_doc = hyde_doc

    if rag_trace.get("fallback_timed_out"):
        strategy = "timeout"

    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
    })
    timings = _trace_timings(rag_trace)
    timings["rewrite_router_ms"] = router_ms
    timings["stepback_llm_ms"] = stepback_llm_ms
    timings["hyde_llm_ms"] = hyde_llm_ms
    timings["rewrite_question_ms"] = _elapsed_ms(rewrite_start)
    if router_error:
        _append_stage_error(rag_trace, "rewrite_router", router_error, "step_back")

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "fallback_started_at": fallback_started_at,
        "fallback_deadline": fallback_deadline,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    expanded_start = time.perf_counter()
    strategy = state.get("expansion_type") or "step_back"
    context_files = state.get("context_files") or []
    rag_trace = state.get("rag_trace", {}) or {}
    fallback_deadline = float(state.get("fallback_deadline") or _fallback_deadline(expanded_start))
    if strategy == "timeout" or rag_trace.get("fallback_timed_out"):
        return _fallback_to_initial_retrieval(state, rag_trace, expanded_start)
    emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    results: List[dict] = []
    rerank_applied_any = False
    rerank_enabled_any = False
    rerank_model = None
    rerank_endpoint = None
    rerank_errors = []
    rerank_input_count = 0
    rerank_output_count = 0
    rerank_input_cap = None
    rerank_input_device_tier = None
    rerank_cache_enabled_any = False
    rerank_cache_hit_any = False
    retrieval_mode = None
    candidate_k = None
    leaf_retrieve_level = None
    auto_merge_enabled = None
    auto_merge_applied = False
    auto_merge_threshold = None
    auto_merge_replaced_chunks = 0
    auto_merge_steps = 0
    expanded_timings: dict[str, float] = {}
    expanded_stage_errors: list[dict] = []
    precomputed_retrievals: dict[str, dict] = {}

    if strategy == "complex":
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        expanded_query = state.get("expanded_query") or state["question"]
        jobs = {
            "hyde": _submit_with_context(lambda: retrieve_documents(hypothetical_doc, top_k=5, context_files=context_files)),
            "step_back": _submit_with_context(lambda: retrieve_documents(expanded_query, top_k=5, context_files=context_files)),
        }
        for key, future in jobs.items():
            retrieved = _await_with_deadline(future, fallback_deadline, rag_trace, f"{key}_retrieve", "initial_retrieval")
            if retrieved is None:
                return _fallback_to_initial_retrieval(state, rag_trace, expanded_start)
            precomputed_retrievals[key] = retrieved

    if strategy in ("hyde", "complex"):
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        if "hyde" in precomputed_retrievals:
            retrieved_hyde = precomputed_retrievals["hyde"]
        else:
            future = _submit_with_context(lambda: retrieve_documents(hypothetical_doc, top_k=5, context_files=context_files))
            retrieved_hyde = _await_with_deadline(future, fallback_deadline, rag_trace, "hyde_retrieve", "initial_retrieval")
            if retrieved_hyde is None:
                return _fallback_to_initial_retrieval(state, rag_trace, expanded_start)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        hyde_timings = hyde_meta.get("timings") or {}
        expanded_timings["expanded_hyde_retrieve_ms"] = float(hyde_timings.get("total_retrieve_ms") or 0.0)
        expanded_stage_errors.extend(_prefixed_stage_errors("hyde", hyde_meta.get("stage_errors") or []))
        emit_rag_step(
            "🧱",
            "HyDE 三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(hyde_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(hyde_meta.get("rerank_enabled"))
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
        rerank_input_count += int(hyde_meta.get("rerank_input_count") or 0)
        rerank_output_count += int(hyde_meta.get("rerank_output_count") or 0)
        rerank_input_cap = rerank_input_cap if rerank_input_cap is not None else hyde_meta.get("rerank_input_cap")
        rerank_input_device_tier = rerank_input_device_tier or hyde_meta.get("rerank_input_device_tier")
        rerank_cache_enabled_any = rerank_cache_enabled_any or bool(hyde_meta.get("rerank_cache_enabled"))
        rerank_cache_hit_any = rerank_cache_hit_any or bool(hyde_meta.get("rerank_cache_hit"))
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or hyde_meta.get("retrieval_mode")
        candidate_k = candidate_k or hyde_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or hyde_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else hyde_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(hyde_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or hyde_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(hyde_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(hyde_meta.get("auto_merge_steps") or 0)

    if strategy in ("step_back", "complex"):
        expanded_query = state.get("expanded_query") or state["question"]
        if "step_back" in precomputed_retrievals:
            retrieved_stepback = precomputed_retrievals["step_back"]
        else:
            future = _submit_with_context(lambda: retrieve_documents(expanded_query, top_k=5, context_files=context_files))
            retrieved_stepback = _await_with_deadline(future, fallback_deadline, rag_trace, "stepback_retrieve", "initial_retrieval")
            if retrieved_stepback is None:
                return _fallback_to_initial_retrieval(state, rag_trace, expanded_start)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        step_timings = step_meta.get("timings") or {}
        expanded_timings["expanded_stepback_retrieve_ms"] = float(step_timings.get("total_retrieve_ms") or 0.0)
        expanded_stage_errors.extend(_prefixed_stage_errors("stepback", step_meta.get("stage_errors") or []))
        emit_rag_step(
            "🧱",
            "Step-back 三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(step_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(step_meta.get("rerank_enabled"))
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
        rerank_input_count += int(step_meta.get("rerank_input_count") or 0)
        rerank_output_count += int(step_meta.get("rerank_output_count") or 0)
        rerank_input_cap = rerank_input_cap if rerank_input_cap is not None else step_meta.get("rerank_input_cap")
        rerank_input_device_tier = rerank_input_device_tier or step_meta.get("rerank_input_device_tier")
        rerank_cache_enabled_any = rerank_cache_enabled_any or bool(step_meta.get("rerank_cache_enabled"))
        rerank_cache_hit_any = rerank_cache_hit_any or bool(step_meta.get("rerank_cache_hit"))
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or step_meta.get("retrieval_mode")
        candidate_k = candidate_k or step_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or step_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else step_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(step_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or step_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(step_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(step_meta.get("auto_merge_steps") or 0)

    deduped = []
    seen = set()
    for item in results:
        key = (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 扩展阶段可能合并了多路召回（如 hyde + step_back），
    # 这里统一重排展示名次，避免出现 1,2,3,4,5,4,5 这类重复名次。
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    emit_rag_step("✅", f"扩展检索完成，共 {len(deduped)} 个片段")
    rag_trace = state.get("rag_trace", {}) or {}
    timings = _trace_timings(rag_trace)
    timings.update(expanded_timings)
    timings["expanded_retrieve_ms"] = _elapsed_ms(expanded_start)
    _trace_stage_errors(rag_trace).extend(expanded_stage_errors)
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "context_files": context_files,
        "retrieval_stage": "expanded",
        "rerank_enabled": rerank_enabled_any,
        "rerank_applied": rerank_applied_any,
        "rerank_model": rerank_model,
        "rerank_endpoint": rerank_endpoint,
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
        "rerank_input_count": rerank_input_count,
        "rerank_output_count": rerank_output_count,
        "rerank_input_cap": rerank_input_cap,
        "rerank_input_device_tier": rerank_input_device_tier,
        "rerank_cache_enabled": rerank_cache_enabled_any,
        "rerank_cache_hit": rerank_cache_hit_any,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "leaf_retrieve_level": leaf_retrieve_level,
        "auto_merge_enabled": auto_merge_enabled,
        "auto_merge_applied": auto_merge_applied,
        "auto_merge_threshold": auto_merge_threshold,
        "auto_merge_replaced_chunks": auto_merge_replaced_chunks,
        "auto_merge_steps": auto_merge_steps,
        "structure_rerank_enabled": state.get("rag_trace", {}).get("structure_rerank_enabled"),
        "structure_rerank_applied": state.get("rag_trace", {}).get("structure_rerank_applied"),
        "structure_rerank_root_weight": state.get("rag_trace", {}).get("structure_rerank_root_weight"),
        "same_root_cap": state.get("rag_trace", {}).get("same_root_cap"),
        "dominant_root_id": state.get("rag_trace", {}).get("dominant_root_id"),
        "dominant_root_share": state.get("rag_trace", {}).get("dominant_root_share"),
        "dominant_root_support": state.get("rag_trace", {}).get("dominant_root_support"),
        "confidence_gate_enabled": state.get("rag_trace", {}).get("confidence_gate_enabled"),
        "fallback_required": state.get("rag_trace", {}).get("fallback_required"),
        "confidence_reasons": state.get("rag_trace", {}).get("confidence_reasons"),
        "top_margin": state.get("rag_trace", {}).get("top_margin"),
        "top_score": state.get("rag_trace", {}).get("top_score"),
        "anchor_match": state.get("rag_trace", {}).get("anchor_match"),
        "query_anchors": state.get("rag_trace", {}).get("query_anchors"),
        "candidates_before_rerank": state.get("rag_trace", {}).get("candidates_before_rerank"),
        "candidates_after_rerank": state.get("rag_trace", {}).get("candidates_after_rerank"),
        "candidates_after_structure_rerank": state.get("rag_trace", {}).get("candidates_after_structure_rerank"),
        "context_chars": len(context),
        "retrieved_chunk_count": len(deduped),
        "final_context_chunk_count": len(deduped),
    })
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)
    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str, context_files: list[str] | None = None) -> dict:
    graph_start = time.perf_counter()
    result = rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "context_files": context_files or [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "fallback_started_at": None,
        "fallback_deadline": None,
        "rag_trace": None,
    })
    rag_trace = result.get("rag_trace") or {}
    _trace_timings(rag_trace)["total_rag_graph_ms"] = _elapsed_ms(graph_start)
    result["rag_trace"] = rag_trace
    return result
