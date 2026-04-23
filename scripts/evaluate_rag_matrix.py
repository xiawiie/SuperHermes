from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "A0": {
        "description": "raw text baseline",
        "reindex_mode": "raw",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "raw",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "A1": {
        "description": "title-context retrieval text",
        "reindex_mode": "title_context",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "B1": {
        "description": "title-context retrieval text with structure rerank",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "G0": {
        "description": "structure rerank without confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "G1": {
        "description": "recommended confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.05",
            "LOW_CONF_ROOT_SHARE": "0.45",
            "LOW_CONF_TOP_SCORE": "0.20",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "G2": {
        "description": "looser confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.03",
            "LOW_CONF_ROOT_SHARE": "0.35",
            "LOW_CONF_TOP_SCORE": "0.15",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "G3": {
        "description": "stricter confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.08",
            "LOW_CONF_ROOT_SHARE": "0.55",
            "LOW_CONF_TOP_SCORE": "0.25",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
}

PAIR_DEFINITIONS = (
    ("A1_vs_A0", "A0", "A1"),
    ("B1_vs_A1", "A1", "B1"),
    ("G1_vs_G0", "G0", "G1"),
    ("G1_vs_G2", "G2", "G1"),
    ("G1_vs_G3", "G3", "G1"),
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _doc_text(doc: dict) -> str:
    parts = [
        doc.get("anchor_id"),
        doc.get("section_title"),
        doc.get("section_path"),
        doc.get("retrieval_text"),
        doc.get("text"),
        doc.get("text_preview"),
    ]
    return " ".join(str(part) for part in parts if part)


def _doc_match_flags(
    doc: dict,
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
) -> dict[str, bool]:
    chunk_ids = set(_as_list(expected_chunk_ids))
    root_ids = set(_as_list(expected_root_ids))
    anchors = _as_list(expected_anchors)
    keywords = _as_list(expected_keywords)
    text = _doc_text(doc)

    chunk_hit = bool(str(doc.get("chunk_id") or "") in chunk_ids)
    root_hit = bool(str(doc.get("root_chunk_id") or "") in root_ids)
    anchor_hit = any(anchor and anchor in text for anchor in anchors)
    keyword_hit = any(keyword and keyword in text for keyword in keywords)

    return {
        "chunk": chunk_hit,
        "root": root_hit,
        "anchor": anchor_hit,
        "keyword": keyword_hit,
        "any": chunk_hit or root_hit or anchor_hit or keyword_hit,
    }


def first_relevant_rank(
    docs: list[dict],
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    top_k: int = 5,
) -> int | None:
    for idx, doc in enumerate((docs or [])[:top_k], 1):
        flags = _doc_match_flags(
            doc,
            expected_chunk_ids=expected_chunk_ids,
            expected_root_ids=expected_root_ids,
            expected_anchors=expected_anchors,
            expected_keywords=expected_keywords,
        )
        if flags["any"]:
            return idx
    return None


def compute_retrieval_metrics(
    docs: list[dict],
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    top_docs = (docs or [])[:top_k]
    has_expected = bool(
        _as_list(expected_chunk_ids)
        or _as_list(expected_root_ids)
        or _as_list(expected_anchors)
        or _as_list(expected_keywords)
    )
    flags = [
        _doc_match_flags(
            doc,
            expected_chunk_ids=expected_chunk_ids,
            expected_root_ids=expected_root_ids,
            expected_anchors=expected_anchors,
            expected_keywords=expected_keywords,
        )
        for doc in top_docs
    ]
    relevant_count = sum(1 for item in flags if item["any"])
    rank = first_relevant_rank(
        top_docs,
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
        expected_anchors=expected_anchors,
        expected_keywords=expected_keywords,
        top_k=top_k,
    )
    returned_count = len(top_docs)
    precision = (relevant_count / returned_count) if returned_count else 0.0

    return {
        "top_k": top_k,
        "returned_count": returned_count,
        "scorable": has_expected,
        "hit_at_5": bool(rank),
        "root_hit_at_5": any(item["root"] for item in flags),
        "anchor_hit_at_5": any(item["anchor"] for item in flags),
        "keyword_hit_at_5": any(item["keyword"] for item in flags),
        "legacy_chunk_hit_at_5": any(item["chunk"] for item in flags),
        "first_relevant_rank": rank,
        "mrr": (1.0 / rank) if rank else 0.0,
        "context_precision_id_at_5": precision if has_expected else None,
        "irrelevant_context_ratio_at_5": (1.0 - precision) if has_expected else None,
        "relevant_count": relevant_count,
        "error_rate": 0.0,
    }


def compare_sample_rank(old_metrics: dict, new_metrics: dict) -> str:
    old_hit = bool(old_metrics.get("hit_at_5"))
    new_hit = bool(new_metrics.get("hit_at_5"))
    old_rank = old_metrics.get("first_relevant_rank")
    new_rank = new_metrics.get("first_relevant_rank")

    if not old_hit and new_hit:
        return "win"
    if old_hit and not new_hit:
        return "loss"
    if old_hit and new_hit:
        if new_rank < old_rank:
            return "win"
        if new_rank > old_rank:
            return "loss"
    return "tie"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _average(rows: list[dict], metric: str) -> float | None:
    values = [
        row.get("metrics", {}).get(metric)
        for row in rows
        if isinstance(row.get("metrics", {}).get(metric), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _average_field(rows: list[dict], field: str) -> float | None:
    values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float, bool))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _build_pairwise(rows: list[dict], old_variant: str, new_variant: str) -> dict[str, int]:
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row.get("sample_id"))][str(row.get("variant"))] = row

    counts = Counter({"wins": 0, "losses": 0, "ties": 0, "missing": 0})
    for variants in by_sample.values():
        old_row = variants.get(old_variant)
        new_row = variants.get(new_variant)
        if not old_row or not new_row:
            counts["missing"] += 1
            continue
        result = compare_sample_rank(old_row.get("metrics", {}), new_row.get("metrics", {}))
        counts[{"win": "wins", "loss": "losses", "tie": "ties"}[result]] += 1
    return dict(counts)


def summarize_results(rows: list[dict], variants: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rows": len(rows),
        "variants": {},
        "paired_comparisons": {},
        "diagnostics": {},
    }

    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        diagnostic_counts = Counter(
            (row.get("diagnostic_result") or {}).get("category", "unknown")
            for row in variant_rows
        )
        summary["variants"][variant] = {
            "rows": len(variant_rows),
            "hit_at_5": _average(variant_rows, "hit_at_5"),
            "root_hit_at_5": _average(variant_rows, "root_hit_at_5"),
            "anchor_hit_at_5": _average(variant_rows, "anchor_hit_at_5"),
            "keyword_hit_at_5": _average(variant_rows, "keyword_hit_at_5"),
            "legacy_chunk_hit_at_5": _average(variant_rows, "legacy_chunk_hit_at_5"),
            "mrr": _average(variant_rows, "mrr"),
            "context_precision_id_at_5": _average(variant_rows, "context_precision_id_at_5"),
            "irrelevant_context_ratio_at_5": _average(variant_rows, "irrelevant_context_ratio_at_5"),
            "avg_latency_ms": _average_field(variant_rows, "latency_ms"),
            "error_rate": _average_field(variant_rows, "error_rate"),
            "fallback_trigger_rate": _average_field(variant_rows, "fallback_required"),
        }
        summary["diagnostics"][variant] = dict(diagnostic_counts)

    for pair_name, old_variant, new_variant in PAIR_DEFINITIONS:
        if old_variant in variants and new_variant in variants:
            summary["paired_comparisons"][pair_name] = _build_pairwise(rows, old_variant, new_variant)

    return summary


def render_summary_markdown(summary: dict) -> str:
    lines = [
        "# RAG Matrix Evaluation Summary",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        f"Rows: `{summary.get('sample_rows')}`",
        "",
        "## Variant Metrics",
        "",
        "| Variant | Rows | Hit@5 | Root@5 | Anchor@5 | Keyword@5 | MRR | CtxPrecision@5 | Irrelevant@5 | Avg ms | Error | Fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, metrics in summary.get("variants", {}).items():
        lines.append(
            "| {variant} | {rows} | {hit} | {root} | {anchor} | {keyword} | {mrr} | {precision} | {irrelevant} | {latency} | {error} | {fallback} |".format(
                variant=variant,
                rows=metrics.get("rows", 0),
                hit=_fmt_metric(metrics.get("hit_at_5")),
                root=_fmt_metric(metrics.get("root_hit_at_5")),
                anchor=_fmt_metric(metrics.get("anchor_hit_at_5")),
                keyword=_fmt_metric(metrics.get("keyword_hit_at_5")),
                mrr=_fmt_metric(metrics.get("mrr")),
                precision=_fmt_metric(metrics.get("context_precision_id_at_5")),
                irrelevant=_fmt_metric(metrics.get("irrelevant_context_ratio_at_5")),
                latency=_fmt_metric(metrics.get("avg_latency_ms")),
                error=_fmt_metric(metrics.get("error_rate")),
                fallback=_fmt_metric(metrics.get("fallback_trigger_rate")),
            )
        )

    lines.extend(["", "## Paired Comparisons", ""])
    for pair_name, counts in summary.get("paired_comparisons", {}).items():
        lines.append(
            f"- `{pair_name}`: wins={counts.get('wins', 0)}, losses={counts.get('losses', 0)}, ties={counts.get('ties', 0)}, missing={counts.get('missing', 0)}"
        )

    lines.extend(["", "## Diagnostics", ""])
    for variant, counts in summary.get("diagnostics", {}).items():
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"- `{variant}`: {rendered or 'none'}")

    lines.extend([
        "",
        "## Notes",
        "",
        "- `keyword_hit_at_5` uses any expected keyword in this implementation.",
        "- DeepEval/Ragas/TruLens are not runtime dependencies in this report.",
    ])
    return "\n".join(lines) + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1.000" if value else "0.000"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _expected_fields(record: dict) -> dict[str, list[str]]:
    expected_chunk_ids = (
        _as_list(record.get("expected_chunk_ids"))
        or _as_list(record.get("legacy_gold_chunk_ids"))
        or _as_list(record.get("gold_chunk_ids"))
    )
    return {
        "expected_chunk_ids": expected_chunk_ids,
        "expected_root_ids": _as_list(record.get("expected_root_ids")),
        "expected_anchors": _as_list(record.get("expected_anchors")),
        "expected_keywords": _as_list(record.get("expected_keywords")),
    }


def _summarize_doc(doc: dict) -> dict[str, Any]:
    text = str(doc.get("text") or doc.get("retrieval_text") or doc.get("text_preview") or "")
    return {
        "chunk_id": doc.get("chunk_id"),
        "parent_chunk_id": doc.get("parent_chunk_id"),
        "root_chunk_id": doc.get("root_chunk_id"),
        "anchor_id": doc.get("anchor_id"),
        "section_title": doc.get("section_title"),
        "section_path": doc.get("section_path"),
        "filename": doc.get("filename"),
        "page_number": doc.get("page_number"),
        "score": doc.get("score"),
        "rerank_score": doc.get("rerank_score"),
        "final_score": doc.get("final_score"),
        "text_preview": text[:240],
    }


def _ensure_backend_imports() -> tuple[Any, Any]:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from rag_diagnostics import classify_failure
    from rag_utils import retrieve_documents

    return classify_failure, retrieve_documents


def evaluate_sample(record: dict, variant: str, top_k: int) -> dict[str, Any]:
    classify_failure, retrieve_documents = _ensure_backend_imports()
    query = str(record.get("question") or record.get("query") or record.get("input") or "")
    sample_id = str(record.get("sample_id") or record.get("id") or query[:60])
    expected = _expected_fields(record)
    started = time.perf_counter()
    try:
        retrieved = retrieve_documents(query, top_k=top_k, context_files=None)
        docs = retrieved.get("docs", [])
        meta = retrieved.get("meta", {})
        latency_ms = (time.perf_counter() - started) * 1000
        metrics = compute_retrieval_metrics(docs, top_k=top_k, **expected)
        metrics["error_rate"] = 0.0
        rag_trace = {"retrieved_chunks": docs, **meta}
        diagnostic_result = classify_failure(query=query, rag_trace=rag_trace, **expected)
        return {
            "sample_id": sample_id,
            "variant": variant,
            "query": query,
            "expected": expected,
            "retrieved_chunks": [_summarize_doc(doc) for doc in docs[:top_k]],
            "trace": _summarize_trace(meta),
            "metrics": metrics,
            "diagnostic_result": diagnostic_result,
            "latency_ms": latency_ms,
            "error_rate": 0.0,
            "fallback_required": bool(meta.get("fallback_required")),
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "sample_id": sample_id,
            "variant": variant,
            "query": query,
            "expected": expected,
            "retrieved_chunks": [],
            "trace": {},
            "metrics": {
                "top_k": top_k,
                "returned_count": 0,
                "scorable": bool(any(expected.values())),
                "hit_at_5": False,
                "root_hit_at_5": False,
                "anchor_hit_at_5": False,
                "keyword_hit_at_5": False,
                "legacy_chunk_hit_at_5": False,
                "first_relevant_rank": None,
                "mrr": 0.0,
                "context_precision_id_at_5": 0.0 if any(expected.values()) else None,
                "irrelevant_context_ratio_at_5": 1.0 if any(expected.values()) else None,
                "relevant_count": 0,
                "error_rate": 1.0,
            },
            "diagnostic_result": {
                "category": "error",
                "failed_stage": "unknown",
                "evidence": {"error": str(exc)},
                "suggestions": ["检查检索运行环境、Milvus 连接和 embedding/rerank 服务配置。"],
            },
            "latency_ms": latency_ms,
            "error_rate": 1.0,
            "fallback_required": False,
            "error": str(exc),
        }


def _summarize_trace(meta: dict) -> dict[str, Any]:
    keys = [
        "retrieval_mode",
        "candidate_k",
        "leaf_retrieve_level",
        "rerank_enabled",
        "rerank_applied",
        "rerank_error",
        "structure_rerank_enabled",
        "structure_rerank_applied",
        "structure_rerank_root_weight",
        "same_root_cap",
        "confidence_gate_enabled",
        "fallback_required",
        "confidence_reasons",
        "top_margin",
        "top_score",
        "dominant_root_id",
        "dominant_root_share",
        "anchor_match",
        "query_anchors",
        "hybrid_error",
        "dense_error",
    ]
    summary = {key: meta.get(key) for key in keys if key in meta}
    for trace_key in (
        "candidates_before_rerank",
        "candidates_after_rerank",
        "candidates_after_structure_rerank",
    ):
        summary[trace_key] = (meta.get(trace_key) or [])[:10]
    return summary


def evaluate_variant(dataset: Path, variant: str, output: Path, limit: int | None, top_k: int) -> int:
    records = load_jsonl(dataset, limit=limit)
    rows = []
    for idx, record in enumerate(records, 1):
        row = evaluate_sample(record, variant=variant, top_k=top_k)
        rows.append(row)
        print(
            f"done variant={variant} sample={idx}/{len(records)} id={row['sample_id']} "
            f"hit={row['metrics'].get('hit_at_5')} err={bool(row.get('error'))}",
            flush=True,
        )
    write_jsonl(output, rows)
    return 0


def _merged_env(variant: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in VARIANT_CONFIGS[variant]["env"].items()})
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run_reindex(variant: str) -> None:
    print(f"reindex variant={variant} mode={VARIANT_CONFIGS[variant]['reindex_mode']}", flush=True)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reindex_knowledge_base.py")],
        cwd=PROJECT_ROOT,
        env=_merged_env(variant),
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"reindex failed for {variant} with exit code {result.returncode}")


def _run_worker(dataset: Path, report_dir: Path, variant: str, limit: int | None, top_k: int) -> Path:
    output = report_dir / f"variant-{variant}.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        str(dataset),
        "--top-k",
        str(top_k),
        "--worker-variant",
        variant,
        "--worker-output",
        str(output),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_merged_env(variant), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"worker failed for {variant} with exit code {result.returncode}")
    return output


def _git_status_summary() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return [f"git status failed: {exc}"]


def _write_config(path: Path, args: argparse.Namespace, variants: list[str], destructive_reindex_run: bool) -> None:
    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "limit": args.limit,
        "top_k": args.top_k,
        "variants": {variant: VARIANT_CONFIGS[variant] for variant in variants},
        "destructive_reindex_run": destructive_reindex_run,
        "skip_reindex": args.skip_reindex,
        "git_status": _git_status_summary(),
        "env_keys": [
            "EVAL_RETRIEVAL_TEXT_MODE",
            "STRUCTURE_RERANK_ENABLED",
            "CONFIDENCE_GATE_ENABLED",
            "LOW_CONF_TOP_MARGIN",
            "LOW_CONF_ROOT_SHARE",
            "LOW_CONF_TOP_SCORE",
            "ENABLE_ANCHOR_GATE",
        ],
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_matrix(args: argparse.Namespace) -> int:
    variants = parse_variants(args.variants)
    report_dir = args.output_root / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    destructive_reindex_run = False
    variant_outputs: list[Path] = []
    last_reindex_mode = None
    for variant in variants:
        config = VARIANT_CONFIGS[variant]
        if config["requires_reindex"] and not args.skip_reindex:
            if not args.allow_destructive_reindex:
                raise RuntimeError(
                    f"{variant} requires destructive reindex. Pass --allow-destructive-reindex or --skip-reindex."
                )
            _run_reindex(variant)
            destructive_reindex_run = True
            last_reindex_mode = config["reindex_mode"]
        elif config["requires_reindex"] and args.skip_reindex:
            print(f"skip reindex variant={variant}", flush=True)
        elif last_reindex_mode and config["reindex_mode"] != last_reindex_mode:
            print(
                f"warning variant={variant} expects mode={config['reindex_mode']} but last reindex={last_reindex_mode}",
                flush=True,
            )

        variant_outputs.append(_run_worker(args.dataset, report_dir, variant, args.limit, args.top_k))

    rows: list[dict] = []
    for output in variant_outputs:
        rows.extend(load_jsonl(output))

    write_jsonl(report_dir / "results.jsonl", rows)
    summary = summarize_results(rows, variants=variants)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (report_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    _write_config(report_dir / "config.json", args, variants, destructive_reindex_run)
    print(f"wrote report {report_dir}", flush=True)
    return 0


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return variants


def validate_variant_order(variants: list[str], skip_reindex: bool) -> None:
    if skip_reindex:
        return

    current_index_mode = None
    for variant in variants:
        config = VARIANT_CONFIGS[variant]
        expected_mode = config["reindex_mode"]
        if config["requires_reindex"]:
            current_index_mode = expected_mode
            continue
        if current_index_mode != expected_mode:
            raise RuntimeError(
                f"{variant} expects {expected_mode} index. Include a preceding reindex variant such as A1 "
                "or pass --skip-reindex when intentionally reusing an existing index."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SuperHermes RAG matrix retrieval evaluation.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / ".jbeval" / "datasets" / "rag_tuning_derived.jsonl")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / ".jbeval" / "reports")
    parser.add_argument("--run-id", default=f"rag-matrix-{datetime.now().strftime('%Y%m%d-%H%M')}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variants", default="A0,A1,B1,G0,G1,G2,G3")
    parser.add_argument("--skip-reindex", action="store_true")
    parser.add_argument("--allow-destructive-reindex", action="store_true")
    parser.add_argument("--worker-variant", choices=sorted(VARIANT_CONFIGS), default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.output_root = args.output_root.resolve()

    if args.worker_variant:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-variant")
        return evaluate_variant(
            dataset=args.dataset,
            variant=args.worker_variant,
            output=args.worker_output.resolve(),
            limit=args.limit,
            top_k=args.top_k,
        )

    validate_variant_order(parse_variants(args.variants), skip_reindex=args.skip_reindex)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
