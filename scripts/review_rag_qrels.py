from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.shared.filename_normalization import normalize_filename_for_match  # noqa: E402
from scripts.rag_dataset_utils import alignment_score, load_jsonl, write_json, write_jsonl  # noqa: E402
from scripts.rag_qrels import attach_canonical_ids, normalized_text_hash  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_chunk_gold_v2.jsonl"
DEFAULT_POOL = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_chunk_pool_gold_tcf.jsonl"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / ".jbeval" / "qrel_reviews"
DEFAULT_OUTPUT = PROJECT_ROOT / ".jbeval" / "qrels" / "rag_chunk_gold_v2.1.jsonl"
QREL_VERSION = "v2.1"


class QrelJudge(Protocol):
    def judge(self, row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PoolIndex:
    by_chunk_id: dict[str, dict[str, Any]]
    by_root_id: dict[str, list[dict[str, Any]]]
    by_canonical_chunk_id: dict[str, dict[str, Any]]
    by_canonical_root_id: dict[str, list[dict[str, Any]]]
    by_filename: dict[str, list[dict[str, Any]]]


class NoopJudge:
    def judge(self, row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "qid": qid(row),
            "llm_verdict": "reject",
            "llm_confidence": 0.0,
            "llm_reasoning": "LLM review disabled.",
            "support_level": "no_support",
            "claim_coverage": 0.0,
            "new_chunk_id": None,
            "new_root_id": None,
        }


JUDGE_SYSTEM_PROMPT = """\
You are a RAG qrel reviewer. Your job is to judge whether a chunk of text provides \
sufficient evidence to support the core claims in the expected answer.

For each chunk candidate, evaluate:
1. support_level: full_support / partial_support / topical_only / no_support
2. claim_coverage: 0.0-1.0 鈥?what fraction of the core claims in expected_answer \
can be directly supported by this chunk?
3. llm_verdict: accept / reject / remap
   - accept: this chunk fully supports the answer (full_support, claim_coverage >= 0.80)
   - remap: a different chunk is better (provide new_chunk_id)
   - reject: no suitable chunk found

Be strict: "topical_only" means the chunk is about the same general topic but does \
not contain the specific evidence needed to answer the question. Do not accept \
topical_only chunks.

Respond in JSON only:
{
  "llm_verdict": "accept|reject|remap",
  "llm_confidence": 0.0,
  "llm_reasoning": "one sentence",
  "support_level": "full_support|partial_support|topical_only|no_support",
  "claim_coverage": 0.0,
  "best_chunk_index": 0,
  "new_chunk_id": null
}
"""


class LLMJudge:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GRADE_MODEL") or os.getenv("MODEL", "glm-5")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        if not self.api_key:
            return None
        from langchain.chat_models import init_chat_model
        self._model = init_chat_model(
            model=self.model,
            model_provider="openai",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0,
        )
        return self._model

    def judge(self, row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        question = str(row.get("question") or row.get("query") or "")
        expected_answer = str(row.get("expected_answer") or row.get("reference_answer") or "")
        source_excerpt = str(row.get("source_excerpt") or "")[:800]

        chunks_text = ""
        for idx, chunk in enumerate(candidates[:8]):
            text = str(chunk.get("text") or chunk.get("retrieval_text") or "")[:600]
            chunk_id = str(chunk.get("chunk_id") or chunk.get("canonical_chunk_id") or "")
            page = str(chunk.get("page_number") or chunk.get("page_start") or "?")
            section = str(chunk.get("section_path") or chunk.get("section_title") or "")
            chunks_text += f"\n[chunk {idx}] id={chunk_id} page={page} section={section}\n{text}\n"

        user_msg = (
            f"Question: {question}\n\n"
            f"Expected answer: {expected_answer}\n\n"
            f"Source excerpt: {source_excerpt}\n\n"
            f"Candidate chunks:{chunks_text if chunks_text else ' (no candidates)'}"
        )

        model = self._get_model()
        if model is None:
            return {
                "qid": qid(row),
                "llm_verdict": "reject",
                "llm_confidence": 0.0,
                "llm_reasoning": "LLM unavailable (no API key).",
                "support_level": "no_support",
                "claim_coverage": 0.0,
                "new_chunk_id": None,
            }

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            resp = model.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            content = str(resp.content)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
            else:
                parsed = {"llm_verdict": "reject", "llm_confidence": 0.0, "llm_reasoning": "Parse failed."}

            best_idx = int(parsed.get("best_chunk_index", 0) or 0)
            new_chunk_id = None
            if parsed.get("llm_verdict") == "accept" and 0 <= best_idx < len(candidates):
                new_chunk_id = str(candidates[best_idx].get("chunk_id") or candidates[best_idx].get("canonical_chunk_id") or "")
            elif parsed.get("llm_verdict") == "remap" and 0 <= best_idx < len(candidates):
                new_chunk_id = str(candidates[best_idx].get("chunk_id") or candidates[best_idx].get("canonical_chunk_id") or "")

            return {
                "qid": qid(row),
                "llm_verdict": str(parsed.get("llm_verdict", "reject")),
                "llm_confidence": float(parsed.get("llm_confidence", 0.0)),
                "llm_reasoning": str(parsed.get("llm_reasoning", "")),
                "support_level": str(parsed.get("support_level", "no_support")),
                "claim_coverage": float(parsed.get("claim_coverage", 0.0)),
                "new_chunk_id": new_chunk_id or None,
                "new_root_id": None,
            }
        except Exception as exc:
            return {
                "qid": qid(row),
                "llm_verdict": "reject",
                "llm_confidence": 0.0,
                "llm_reasoning": f"LLM error: {exc}",
                "support_level": "no_support",
                "claim_coverage": 0.0,
                "new_chunk_id": None,
            }


def qid(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("qid") or row.get("sample_id") or row.get("query_id") or "")


def build_pool_index(pool_rows: list[dict[str, Any]]) -> PoolIndex:
    by_chunk_id: dict[str, dict[str, Any]] = {}
    by_root_id: dict[str, list[dict[str, Any]]] = {}
    by_canonical_chunk_id: dict[str, dict[str, Any]] = {}
    by_canonical_root_id: dict[str, list[dict[str, Any]]] = {}
    by_filename: dict[str, list[dict[str, Any]]] = {}
    for row in pool_rows:
        chunk = attach_canonical_ids(row)
        if chunk.get("chunk_id"):
            by_chunk_id[str(chunk["chunk_id"])] = chunk
        if chunk.get("root_chunk_id"):
            by_root_id.setdefault(str(chunk["root_chunk_id"]), []).append(chunk)
        if chunk.get("canonical_chunk_id"):
            by_canonical_chunk_id[str(chunk["canonical_chunk_id"])] = chunk
        if chunk.get("canonical_root_id"):
            by_canonical_root_id.setdefault(str(chunk["canonical_root_id"]), []).append(chunk)
        fname = normalize_filename_for_match(chunk.get("filename") or chunk.get("file_name") or "")
        if fname:
            by_filename.setdefault(fname, []).append(chunk)
    return PoolIndex(by_chunk_id, by_root_id, by_canonical_chunk_id, by_canonical_root_id, by_filename)


def _first(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    return values


def _row_chunk_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("gold_chunk_ids", "expected_chunk_ids", "canonical_chunk_ids", "expected_canonical_chunk_ids"):
        for item in row.get(field) or []:
            if item:
                values.append(str(item))
    return values


def _row_root_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("expected_root_ids", "canonical_root_ids", "expected_canonical_root_ids"):
        for item in row.get(field) or []:
            if item:
                values.append(str(item))
    return values


def _lookup_chunk(index: PoolIndex, chunk_id: str) -> dict[str, Any] | None:
    return index.by_chunk_id.get(chunk_id) or index.by_canonical_chunk_id.get(chunk_id)


def _lookup_root(index: PoolIndex, root_id: str) -> list[dict[str, Any]]:
    return index.by_root_id.get(root_id) or index.by_canonical_root_id.get(root_id) or []


def _support_text_hash(row: dict[str, Any]) -> str:
    support = _first(row.get("supporting_chunks") or [])
    if not isinstance(support, dict):
        return ""
    return normalized_text_hash(support.get("anchor_text") or support.get("text") or support.get("retrieval_text") or "")


def _pool_text_hash(chunk: dict[str, Any] | None) -> str:
    if not chunk:
        return ""
    return normalized_text_hash(chunk.get("text") or chunk.get("retrieval_text") or "")


def _page_distance(row: dict[str, Any], chunk: dict[str, Any] | None) -> int | None:
    if not chunk:
        return None
    expected = _first(row.get("expected_pages") or row.get("gold_pages") or [])
    if expected is None:
        return None
    try:
        return abs(int(expected) - int(chunk.get("page_number") or chunk.get("page_start") or 0))
    except (TypeError, ValueError):
        return None


def precheck_row(row: dict[str, Any], index: PoolIndex, *, max_page_distance: int = 1) -> dict[str, Any]:
    chunk_ids = _row_chunk_ids(row)
    root_ids = _row_root_ids(row)
    chunk = next((_lookup_chunk(index, item) for item in chunk_ids if _lookup_chunk(index, item)), None)
    root_hit = any(_lookup_root(index, item) for item in root_ids)
    support_hash = _support_text_hash(row)
    pool_hash = _pool_text_hash(chunk)
    page_distance = _page_distance(row, chunk)

    reasons: list[str] = []
    if not chunk:
        reasons.append("chunk_id_missing")
    if root_ids and not root_hit:
        reasons.append("root_id_missing")
    if support_hash and pool_hash and support_hash != pool_hash:
        reasons.append("text_hash_mismatch")
    if page_distance is not None and page_distance > max_page_distance:
        reasons.append("page_distance_mismatch")

    return {
        "qid": qid(row),
        "pre_check_status": "pre_check_pass" if not reasons else "pre_check_fail",
        "pre_check_reasons": reasons,
        "chunk_found": bool(chunk),
        "root_found": root_hit,
        "support_text_hash": support_hash or None,
        "pool_text_hash": pool_hash or None,
        "page_distance": page_distance,
    }


def _row_source_filename(row: dict[str, Any]) -> str:
    gold_files = row.get("gold_files") or row.get("expected_files") or []
    if isinstance(gold_files, list) and gold_files:
        return str(gold_files[0])
    meta = row.get("metadata") or {}
    return str(meta.get("source_file") or row.get("source_file") or "")


def candidate_chunks(row: dict[str, Any], index: PoolIndex, *, limit: int = 8) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk_id in _row_chunk_ids(row):
        chunk = _lookup_chunk(index, chunk_id)
        if chunk and str(chunk.get("chunk_id")) not in seen:
            candidates.append(chunk)
            seen.add(str(chunk.get("chunk_id")))
    for root_id in _row_root_ids(row):
        for chunk in _lookup_root(index, root_id):
            if str(chunk.get("chunk_id")) not in seen:
                candidates.append(chunk)
                seen.add(str(chunk.get("chunk_id")))
            if len(candidates) >= limit:
                return candidates

    if len(candidates) < 3:
        fname = normalize_filename_for_match(_row_source_filename(row))
        pool_chunks = index.by_filename.get(fname, [])
        scored = sorted(
            ((alignment_score(row, chunk), chunk) for chunk in pool_chunks if str(chunk.get("chunk_id")) not in seen),
            key=lambda item: item[0].score,
            reverse=True,
        )
        for score, chunk in scored:
            if str(chunk.get("chunk_id")) not in seen:
                candidates.append(chunk)
                seen.add(str(chunk.get("chunk_id")))
            if len(candidates) >= limit:
                break

    return candidates[:limit]


def apply_llm_policy(suggestion: dict[str, Any]) -> str:
    verdict = str(suggestion.get("llm_verdict") or "")
    support = str(suggestion.get("support_level") or "")
    confidence = float(suggestion.get("llm_confidence") or 0.0)
    coverage = float(suggestion.get("claim_coverage") or 0.0)
    if verdict == "remap":
        return "llm_remapped"
    if verdict == "accept" and support == "full_support" and coverage >= 0.80 and confidence >= 0.85:
        return "llm_approved"
    return "needs_human_review"


def agreement_policy(agreement_rate: float) -> str:
    if agreement_rate >= 0.90:
        return "batch_promote"
    if agreement_rate >= 0.70:
        return "advisory_only"
    return "human_only"


def apply_human_decisions(rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("qid") or item.get("id")): item for item in decisions}
    output: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        quality = dict(out.get("quality") or {})
        decision = by_id.get(qid(row))
        if decision:
            if decision.get("new_chunk_id"):
                out["gold_chunk_ids"] = [decision["new_chunk_id"]]
            if decision.get("new_root_id"):
                out["expected_root_ids"] = [decision["new_root_id"]]
            quality["review_status"] = decision.get("review_status", "corrected")
            quality["review_source"] = decision.get("review_source", "human")
            quality["reviewer_notes"] = decision.get("reviewer_notes", "")
        quality["qrel_version"] = QREL_VERSION
        out["quality"] = quality
        out["qrel_version"] = QREL_VERSION
        output.append(out)
    return output


def diff_rows(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_by_id = {qid(row): row for row in old_rows}
    changes: list[dict[str, Any]] = []
    for row in new_rows:
        old = old_by_id.get(qid(row), {})
        changed_fields = [
            field
            for field in ("gold_chunk_ids", "expected_root_ids", "canonical_chunk_ids", "canonical_root_ids", "quality")
            if old.get(field) != row.get(field)
        ]
        if changed_fields:
            changes.append({"qid": qid(row), "changed_fields": changed_fields})
    return {"qrel_version": QREL_VERSION, "changed_row_count": len(changes), "changes": changes}


def write_human_queue(path: Path, rows: list[dict[str, Any]], prechecks: list[dict[str, Any]]) -> None:
    failed = {item["qid"]: item for item in prechecks if item["pre_check_status"] == "pre_check_fail"}
    lines = ["# Human Review Queue", "", "| qid | reasons | question |", "| --- | --- | --- |"]
    for row in rows:
        item = failed.get(qid(row))
        if not item:
            continue
        question = str(row.get("question") or row.get("query") or "").replace("|", "\\|")
        lines.append(f"| {qid(row)} | {', '.join(item['pre_check_reasons'])} | {question} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_precheck(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    pool = build_pool_index(load_jsonl(args.chunk_pool))
    prechecks = [precheck_row(row, pool) for row in rows]
    args.review_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.review_dir / "precheck.jsonl", prechecks)
    write_human_queue(args.review_dir / "human_review_queue.md", rows, prechecks)
    summary = {
        "qrel_version": QREL_VERSION,
        "rows": len(rows),
        "pre_check_pass": sum(1 for item in prechecks if item["pre_check_status"] == "pre_check_pass"),
        "pre_check_fail": sum(1 for item in prechecks if item["pre_check_status"] == "pre_check_fail"),
    }
    write_json(args.review_dir / "precheck_summary.json", summary)
    return 0 if summary["pre_check_fail"] == 0 else 2


def run_llm_review(args: argparse.Namespace, judge: QrelJudge | None = None) -> int:
    rows = load_jsonl(args.input)
    pool = build_pool_index(load_jsonl(args.chunk_pool))
    rng = random.Random(args.seed)
    scope = {item.strip() for item in args.scope.split(",") if item.strip()}
    selected = [
        row
        for row in rows
        if (row.get("quality") or {}).get("alignment_status") in scope
        or ("aligned-sample" in scope and (row.get("quality") or {}).get("alignment_status") == "aligned")
    ]
    if "aligned-sample" in scope:
        aligned = [row for row in rows if (row.get("quality") or {}).get("alignment_status") == "aligned"]
        selected = [row for row in selected if (row.get("quality") or {}).get("alignment_status") != "aligned"]
        selected.extend(rng.sample(aligned, min(args.sample_size, len(aligned))))

    if judge is None:
        judge = NoopJudge() if args.no_llm else LLMJudge()

    suggestions: list[dict[str, Any]] = []
    for row in selected:
        suggestion = judge.judge(row, candidate_chunks(row, pool))
        suggestion["proposed_status"] = apply_llm_policy(suggestion)
        suggestions.append(suggestion)

    args.review_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.review_dir / "llm_review_suggestions.jsonl", suggestions)

    status_counts: dict[str, int] = {}
    for s in suggestions:
        key = s.get("proposed_status", "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1
    summary = {
        "qrel_version": QREL_VERSION,
        "rows_reviewed": len(suggestions),
        "proposed_status_counts": status_counts,
    }
    write_json(args.review_dir / "llm_review_summary.json", summary)
    print(f"LLM review: {len(suggestions)} rows -> {status_counts}")
    return 0


def run_apply_review(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    decisions = load_jsonl(args.review_decisions)
    output = apply_human_decisions(rows, decisions)
    write_jsonl(args.output, output)
    write_json(args.review_dir / "v2_to_v2.1_diff.json", diff_rows(rows, output))
    return 0


def _auto_decisions_from_suggestions(suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for s in suggestions:
        proposed = str(s.get("proposed_status") or "")
        if proposed == "llm_approved":
            decisions.append({
                "qid": s.get("qid", ""),
                "review_status": "approved",
                "review_source": "llm",
                "reviewer_notes": s.get("llm_reasoning", ""),
                "new_chunk_id": s.get("new_chunk_id"),
                "new_root_id": s.get("new_root_id"),
            })
        elif proposed == "llm_remapped":
            decisions.append({
                "qid": s.get("qid", ""),
                "review_status": "corrected",
                "review_source": "llm",
                "reviewer_notes": s.get("llm_reasoning", ""),
                "new_chunk_id": s.get("new_chunk_id"),
                "new_root_id": s.get("new_root_id"),
            })
        elif proposed == "needs_human_review":
            decisions.append({
                "qid": s.get("qid", ""),
                "review_status": "needs_review",
                "review_source": "llm_suggestion",
                "reviewer_notes": s.get("llm_reasoning", ""),
                "new_chunk_id": None,
                "new_root_id": None,
            })
    return decisions


def run_auto_apply(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    suggestions = load_jsonl(args.review_dir / "llm_review_suggestions.jsonl")
    decisions = _auto_decisions_from_suggestions(suggestions)
    output = apply_human_decisions(rows, decisions)
    write_jsonl(args.output, output)
    diff = diff_rows(rows, output)
    write_json(args.review_dir / "v2_to_v2.1_diff.json", diff)

    quality_counts: dict[str, int] = {}
    for row in output:
        status = (row.get("quality") or {}).get("review_status", "unknown")
        quality_counts[status] = quality_counts.get(status, 0) + 1
    coverage = sum(
        1 for row in output
        if (row.get("gold_chunk_ids") or row.get("canonical_chunk_ids"))
    ) / max(1, len(output))
    print(f"Auto-apply: {len(output)} rows, review statuses: {quality_counts}, chunk_qrel_coverage={coverage:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review RAG chunk qrels without mutating v2 input.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunk-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--llm-review", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--apply-review", action="store_true")
    parser.add_argument("--review-decisions", type=Path, default=DEFAULT_REVIEW_DIR / "human_decisions.jsonl")
    parser.add_argument("--auto-apply", action="store_true")
    parser.add_argument("--scope", default="failed,ambiguous,aligned-sample")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=31)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.precheck_only:
        return run_precheck(args)
    if args.apply_review:
        return run_apply_review(args)
    if args.auto_apply:
        return run_auto_apply(args)
    if args.llm_review:
        return run_llm_review(args)
    return run_precheck(args)


if __name__ == "__main__":
    raise SystemExit(main())
