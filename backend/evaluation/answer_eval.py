from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from backend.config import (
    ARK_API_KEY,
    OPENAI_API_KEY,
    BASE_URL,
    MODEL,
    FAST_MODEL,
    GRADE_MODEL,
    ANSWER_EVAL_GENERATION_MODEL,
    ANSWER_EVAL_JUDGE_MODEL,
)
from backend.shared.json_utils import extract_json_object

API_KEY = ARK_API_KEY or OPENAI_API_KEY
ANSWER_MODEL = ANSWER_EVAL_GENERATION_MODEL or MODEL
JUDGE_MODEL = ANSWER_EVAL_JUDGE_MODEL or FAST_MODEL or GRADE_MODEL or MODEL

_answer_model = None
_judge_model = None


from backend.rag.utils import elapsed_ms  # noqa: E402


def _get_model(kind: str):
    global _answer_model, _judge_model
    model_name = ANSWER_MODEL if kind == "answer" else JUDGE_MODEL
    if not API_KEY or not model_name:
        return None
    if kind == "answer":
        if _answer_model is None:
            _answer_model = init_chat_model(
                model=model_name,
                model_provider="openai",
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=0,
                stream_usage=True,
            )
        return _answer_model
    if _judge_model is None:
        _judge_model = init_chat_model(
            model=model_name,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _judge_model


def _doc_text(doc: dict) -> str:
    return str(doc.get("retrieval_text") or doc.get("text") or doc.get("text_preview") or "")


def _doc_page(doc: dict) -> str:
    page = doc.get("page_number")
    if page in (None, ""):
        page = doc.get("page_start")
    return str(page if page not in (None, "") else "N/A")


def format_context_for_answer(docs: list[dict], max_chars_per_doc: int = 1100) -> str:
    chunks: list[str] = []
    for idx, doc in enumerate((docs or [])[:5], 1):
        filename = str(doc.get("filename") or "unknown")
        page = _doc_page(doc)
        section = str(doc.get("section_path") or doc.get("section_title") or "")
        body = _doc_text(doc).strip()
        if len(body) > max_chars_per_doc:
            body = body[:max_chars_per_doc] + "\n...[truncated]"
        header = f"[{idx}] file={filename} page={page}"
        if section:
            header += f" section={section}"
        chunks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(chunks)


def _clamp_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _filename_aliases(filename: str) -> set[str]:
    raw = str(filename or "").strip()
    if not raw:
        return set()
    stem = Path(raw).stem
    return {raw, stem}


def citation_coverage(answer: str, expected_files: Any = None, expected_pages: Any = None) -> dict[str, Any]:
    answer_text = answer or ""
    files = _as_list(expected_files)
    pages = _as_list(expected_pages)

    matched_files = [
        filename
        for filename in files
        if any(alias and alias in answer_text for alias in _filename_aliases(filename))
    ]
    matched_pages = []
    for page in pages:
        escaped = re.escape(str(page))
        if re.search(rf"(?:p\.?\s*|page\s*|页\s*|第\s*){escaped}(?:\s*页)?", answer_text, flags=re.IGNORECASE):
            matched_pages.append(str(page))

    total = len(files) + len(pages)
    matched = len(matched_files) + len(matched_pages)
    score = (matched / total) if total else None
    return {
        "citation_coverage": score,
        "expected_file_count": len(files),
        "matched_file_count": len(matched_files),
        "matched_files": matched_files,
        "expected_page_count": len(pages),
        "matched_page_count": len(matched_pages),
        "matched_pages": matched_pages,
    }


def generate_grounded_answer(question: str, docs: list[dict]) -> dict[str, Any]:
    model = _get_model("answer")
    context = format_context_for_answer(docs)
    if not model:
        return {
            "answer": "",
            "answer_error": "answer_model_unavailable",
            "answer_model": ANSWER_MODEL,
            "answer_generation_ms": 0.0,
        }
    prompt = (
        "Answer the user question using only the provided retrieved context.\n"
        "If the context is insufficient, say so explicitly.\n"
        "Cite supporting evidence after claims using [filename p.page].\n"
        "Do not invent citations, filenames, pages, or facts.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )
    start = time.perf_counter()
    try:
        response = model.invoke(prompt)
        answer = str(getattr(response, "content", response) or "").strip()
        return {
            "answer": answer,
            "answer_error": None,
            "answer_model": ANSWER_MODEL,
            "answer_generation_ms": elapsed_ms(start),
        }
    except Exception as exc:
        return {
            "answer": "",
            "answer_error": str(exc),
            "answer_model": ANSWER_MODEL,
            "answer_generation_ms": elapsed_ms(start),
        }


def judge_answer(question: str, answer: str, docs: list[dict], reference_answer: str | None = None) -> dict[str, Any]:
    model = _get_model("judge")
    context = format_context_for_answer(docs, max_chars_per_doc=900)
    if not model:
        return {
            "judge_error": "judge_model_unavailable",
            "judge_model": JUDGE_MODEL,
            "faithfulness_score": None,
            "answer_relevance_score": None,
            "grounded_claims": None,
            "total_claims": None,
            "judge_ms": 0.0,
        }
    prompt = (
        "You are evaluating a RAG answer. Return JSON only.\n"
        "Definitions:\n"
        "- faithfulness_score: 0 to 1, whether answer claims are supported by retrieved context.\n"
        "- answer_relevance_score: 0 to 1, whether the answer directly and completely answers the question.\n"
        "- grounded_claims and total_claims: integer claim counts.\n"
        "- verdict: one of pass, partial, fail.\n\n"
        "JSON schema:\n"
        "{\"faithfulness_score\":0.0,\"answer_relevance_score\":0.0,"
        "\"grounded_claims\":0,\"total_claims\":0,\"verdict\":\"partial\",\"reason\":\"...\"}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer, if available:\n{reference_answer or ''}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer:\n{answer}\n"
    )
    start = time.perf_counter()
    try:
        response = model.invoke(prompt)
        payload = extract_json_object(str(getattr(response, "content", response) or ""))
        faithfulness = _clamp_score(payload.get("faithfulness_score"))
        relevance = _clamp_score(payload.get("answer_relevance_score"))
        return {
            "judge_error": None,
            "judge_model": JUDGE_MODEL,
            "faithfulness_score": faithfulness,
            "answer_relevance_score": relevance,
            "grounded_claims": payload.get("grounded_claims"),
            "total_claims": payload.get("total_claims"),
            "verdict": payload.get("verdict"),
            "reason": payload.get("reason"),
            "judge_ms": elapsed_ms(start),
        }
    except Exception as exc:
        return {
            "judge_error": str(exc),
            "judge_model": JUDGE_MODEL,
            "faithfulness_score": None,
            "answer_relevance_score": None,
            "grounded_claims": None,
            "total_claims": None,
            "judge_ms": elapsed_ms(start),
        }


def evaluate_answer_end_to_end(
    question: str,
    docs: list[dict],
    expected: dict[str, Any],
    reference_answer: str | None = None,
) -> dict[str, Any]:
    answer_result = generate_grounded_answer(question, docs)
    answer = answer_result.get("answer") or ""
    judge_result = judge_answer(question, answer, docs, reference_answer=reference_answer) if answer else {
        "judge_error": "answer_empty",
        "judge_model": JUDGE_MODEL,
        "faithfulness_score": None,
        "answer_relevance_score": None,
        "grounded_claims": None,
        "total_claims": None,
        "judge_ms": 0.0,
    }
    coverage = citation_coverage(
        answer,
        expected_files=expected.get("expected_files"),
        expected_pages=expected.get("expected_pages"),
    )
    total_ms = float(answer_result.get("answer_generation_ms") or 0.0) + float(judge_result.get("judge_ms") or 0.0)
    return {
        **answer_result,
        **judge_result,
        **coverage,
        "answer_eval_ms": round(total_ms, 3),
    }
