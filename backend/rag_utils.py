from collections import defaultdict
from typing import List, Tuple, Dict, Any
import os
import json
import re
import requests
from dotenv import load_dotenv

from milvus_client import MilvusManager
from embedding import embedding_service as _embedding_service
from parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))
STRUCTURE_RERANK_ROOT_WEIGHT = float(os.getenv("STRUCTURE_RERANK_ROOT_WEIGHT", "0.3"))
SAME_ROOT_CAP = int(os.getenv("SAME_ROOT_CAP", "2"))
STRUCTURE_RERANK_ENABLED = os.getenv("STRUCTURE_RERANK_ENABLED", "true").lower() != "false"
CONFIDENCE_GATE_ENABLED = os.getenv("CONFIDENCE_GATE_ENABLED", "true").lower() != "false"
LOW_CONF_TOP_MARGIN = float(os.getenv("LOW_CONF_TOP_MARGIN", "0.05"))
LOW_CONF_ROOT_SHARE = float(os.getenv("LOW_CONF_ROOT_SHARE", "0.45"))
LOW_CONF_TOP_SCORE = float(os.getenv("LOW_CONF_TOP_SCORE", "0.20"))
ENABLE_ANCHOR_GATE = os.getenv("ENABLE_ANCHOR_GATE", "true").lower() != "false"

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

_stepback_model = None
_ANCHOR_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]|"
    r"\d+(?:\.\d+){1,4}|"
    r"[一二三四五六七八九十]+、|"
    r"[（(][一二三四五六七八九十0-9A-Za-z]+[)）]|"
    r"附录[A-Za-z0-9一二三四五六七八九十]+|"
    r"附件[0-9一二三四五六七八九十]+)"
)


def _doc_retrieval_text(doc: dict) -> str:
    return str(doc.get("retrieval_text") or doc.get("text") or "")


def _candidate_trace(docs: List[dict]) -> List[dict]:
    traced: List[dict] = []
    for doc in docs:
        traced.append(
            {
                "chunk_id": doc.get("chunk_id", ""),
                "root_chunk_id": doc.get("root_chunk_id", ""),
                "anchor_id": doc.get("anchor_id", ""),
                "filename": doc.get("filename", ""),
                "text_preview": _doc_retrieval_text(doc)[:240],
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
                "final_score": doc.get("final_score"),
            }
        )
    return traced


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 两段自动合并：L3->L2，再 L2->L1。
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _apply_structure_rerank(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not STRUCTURE_RERANK_ENABLED:
        limited = docs[:top_k]
        return limited, {
            "structure_rerank_enabled": False,
            "structure_rerank_applied": False,
            "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
            "same_root_cap": SAME_ROOT_CAP,
            "dominant_root_id": None,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
        }
    if not docs:
        return [], {
            "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
            "structure_rerank_applied": False,
            "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
            "same_root_cap": SAME_ROOT_CAP,
            "dominant_root_id": None,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
        }

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        if not root_id:
            root_id = f"__orphan__:{len(grouped)}"
        grouped[root_id].append(doc)

    root_scores: Dict[str, float] = {}
    for root_id, items in grouped.items():
        root_scores[root_id] = max(
            float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
            for item in items
        )

    scored_docs = []
    for doc in docs:
        leaf_score = float(doc.get("rerank_score", doc.get("score", 0.0)) or 0.0)
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        root_score = root_scores.get(root_id, leaf_score)
        final_score = (1.0 - STRUCTURE_RERANK_ROOT_WEIGHT) * leaf_score + STRUCTURE_RERANK_ROOT_WEIGHT * root_score
        enriched = dict(doc)
        enriched["leaf_score"] = leaf_score
        enriched["root_score"] = root_score
        enriched["final_score"] = final_score
        scored_docs.append(enriched)

    scored_docs.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    limited: List[dict] = []
    per_root: Dict[str, int] = defaultdict(int)
    for doc in scored_docs:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        if per_root[root_id] >= SAME_ROOT_CAP:
            continue
        limited.append(doc)
        per_root[root_id] += 1
        if len(limited) >= top_k:
            break

    root_total_scores: Dict[str, float] = defaultdict(float)
    for doc in limited:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        root_total_scores[root_id] += float(doc.get("final_score", 0.0) or 0.0)

    dominant_root_id = None
    dominant_root_share = 0.0
    dominant_root_support = 0
    total_score = sum(root_total_scores.values())
    if root_total_scores:
        dominant_root_id = max(root_total_scores, key=root_total_scores.get)
        dominant_root_support = per_root.get(dominant_root_id, 0)
        if total_score > 0:
            dominant_root_share = root_total_scores[dominant_root_id] / total_score

    return limited, {
        "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
        "structure_rerank_applied": True,
        "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
        "same_root_cap": SAME_ROOT_CAP,
        "dominant_root_id": dominant_root_id,
        "dominant_root_share": dominant_root_share,
        "dominant_root_support": dominant_root_support,
    }


def _extract_query_anchors(query: str) -> list[str]:
    if not query:
        return []
    return list(dict.fromkeys(_ANCHOR_PATTERN.findall(query)))


def _doc_matches_anchor(doc: dict, anchor: str) -> bool:
    if not anchor:
        return False
    for value in (
        doc.get("anchor_id"),
        doc.get("section_title"),
        doc.get("section_path"),
    ):
        if anchor and anchor in str(value or ""):
            return True
    prefix = _doc_retrieval_text(doc).split("\n", 1)[0]
    return anchor in prefix


def _evaluate_retrieval_confidence(query: str, docs: List[dict]) -> Dict[str, Any]:
    if not docs:
        return {
            "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
            "fallback_required": CONFIDENCE_GATE_ENABLED,
            "confidence_reasons": ["no_docs"] if CONFIDENCE_GATE_ENABLED else [],
            "top_margin": 0.0,
            "top_score": 0.0,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
            "anchor_match": False,
            "query_anchors": [],
        }

    top_score = float(docs[0].get("final_score", docs[0].get("rerank_score", docs[0].get("score", 0.0))) or 0.0)
    second_score = (
        float(docs[1].get("final_score", docs[1].get("rerank_score", docs[1].get("score", 0.0))) or 0.0)
        if len(docs) > 1
        else 0.0
    )
    top_margin = top_score - second_score

    root_total_scores: Dict[str, float] = defaultdict(float)
    root_supports: Dict[str, int] = defaultdict(int)
    for doc in docs[:5]:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        score = float(doc.get("final_score", doc.get("rerank_score", doc.get("score", 0.0))) or 0.0)
        root_total_scores[root_id] += score
        root_supports[root_id] += 1

    dominant_root_share = 0.0
    dominant_root_support = 0
    if root_total_scores:
        dominant_root_id = max(root_total_scores, key=root_total_scores.get)
        total = sum(root_total_scores.values())
        dominant_root_support = root_supports[dominant_root_id]
        dominant_root_share = (root_total_scores[dominant_root_id] / total) if total else 0.0

    anchors = _extract_query_anchors(query) if ENABLE_ANCHOR_GATE else []
    anchor_match = True
    if anchors:
        anchor_match = False
        for anchor in anchors:
            if any(_doc_matches_anchor(doc, anchor) for doc in docs[:2]):
                anchor_match = True
                break

    reasons: list[str] = []
    if anchors and not anchor_match:
        reasons.append("anchor_mismatch")
    if top_margin < LOW_CONF_TOP_MARGIN and dominant_root_share < LOW_CONF_ROOT_SHARE:
        reasons.append("weak_margin_and_root")
    if top_score < LOW_CONF_TOP_SCORE and top_margin < LOW_CONF_TOP_MARGIN:
        reasons.append("low_score_and_margin")

    if not CONFIDENCE_GATE_ENABLED:
        reasons = []

    return {
        "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
        "fallback_required": bool(reasons),
        "confidence_reasons": reasons,
        "top_margin": top_margin,
        "top_score": top_score,
        "dominant_root_share": dominant_root_share,
        "dominant_root_support": dominant_root_support,
        "anchor_match": anchor_match,
        "query_anchors": anchors,
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [_doc_retrieval_text(doc) for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            return docs_with_rank[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return docs_with_rank[:top_k], meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return docs_with_rank[:top_k], meta


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_filename_filter(filenames: list[str] | None) -> str:
    clean_files = []
    seen = set()
    for filename in filenames or []:
        name = (filename or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_files.append(_escape_milvus_string(name))
    if not clean_files:
        return ""
    quoted = ", ".join(f'"{filename}"' for filename in clean_files)
    return f"filename in [{quoted}]"


def _dedupe_docs(docs: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def retrieve_context_documents(filenames: list[str] | None, limit_per_file: int = 8) -> Dict[str, Any]:
    """Fetch representative indexed chunks directly from attached filenames."""
    clean_files = []
    seen = set()
    for filename in filenames or []:
        name = (filename or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_files.append(name)

    docs = []
    for filename in clean_files:
        filename_filter = _build_filename_filter([filename])
        if not filename_filter:
            continue
        rows = _milvus_manager.query(
            filter_expr=f"chunk_level == {LEAF_RETRIEVE_LEVEL} and {filename_filter}",
            output_fields=[
                "text",
                "retrieval_text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_role",
                "section_title",
                "section_path",
                "anchor_id",
                "chunk_idx",
            ],
            limit=limit_per_file,
        )
        rows.sort(key=lambda item: (item.get("page_number", 0), item.get("chunk_idx", 0)))
        docs.extend(rows)

    docs = _dedupe_docs(docs)
    for idx, item in enumerate(docs, 1):
        item["rrf_rank"] = idx
    return {
        "docs": docs,
        "meta": {
            "retrieval_mode": "attached_file_context",
            "candidate_k": limit_per_file * max(len(clean_files), 1),
            "context_files": clean_files,
            "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
            "attached_context_count": len(docs),
        },
    }


def retrieve_documents(query: str, top_k: int = 5, context_files: list[str] | None = None) -> Dict[str, Any]:
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    filename_filter = _build_filename_filter(context_files)
    if filename_filter:
        filter_expr = f"{filter_expr} and {filename_filter}"
    hybrid_error = None
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
        confidence_meta = _evaluate_retrieval_confidence(query=query, docs=reranked_docs)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta["context_files"] = context_files or []
        rerank_meta["hybrid_error"] = None
        rerank_meta["dense_error"] = None
        rerank_meta.update(structure_meta)
        rerank_meta.update(confidence_meta)
        rerank_meta["candidates_before_rerank"] = _candidate_trace(retrieved)
        rerank_meta["candidates_after_rerank"] = _candidate_trace(reranked)
        rerank_meta["candidates_after_structure_rerank"] = _candidate_trace(reranked_docs)
        return {"docs": reranked_docs, "meta": rerank_meta}
    except Exception as exc:
        hybrid_error = str(exc)
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
            confidence_meta = _evaluate_retrieval_confidence(query=query, docs=reranked_docs)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta["context_files"] = context_files or []
            rerank_meta["hybrid_error"] = hybrid_error
            rerank_meta["dense_error"] = None
            rerank_meta.update(structure_meta)
            rerank_meta.update(confidence_meta)
            rerank_meta["candidates_before_rerank"] = _candidate_trace(retrieved)
            rerank_meta["candidates_after_rerank"] = _candidate_trace(reranked)
            rerank_meta["candidates_after_structure_rerank"] = _candidate_trace(reranked_docs)
            return {"docs": reranked_docs, "meta": rerank_meta}
        except Exception as dense_exc:
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "hybrid_error": hybrid_error,
                    "dense_error": str(dense_exc),
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "context_files": context_files or [],
                    "structure_rerank_applied": False,
                    "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
                    "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
                    "same_root_cap": SAME_ROOT_CAP,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                    "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
                    "fallback_required": CONFIDENCE_GATE_ENABLED,
                    "confidence_reasons": ["retrieve_failed"] if CONFIDENCE_GATE_ENABLED else [],
                    "candidates_before_rerank": [],
                    "candidates_after_rerank": [],
                    "candidates_after_structure_rerank": [],
                },
            }
