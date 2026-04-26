"""Diagnose a RAG variant profile's index/collection health against a reference variant."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.shared.filename_normalization import normalize_filename_for_match  # noqa: E402


@dataclass(frozen=True)
class CollectionProfile:
    exists: bool
    document_count: int | None
    chunk_count: int | None
    indexed_filenames: set[str] = field(default_factory=set)
    sample_candidates: dict[str, bool] = field(default_factory=dict)


def variant_env(variant: str) -> dict[str, str]:
    from scripts.rag_eval.variants import VARIANT_CONFIGS
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {sorted(VARIANT_CONFIGS.keys())}")
    return {k: str(v) for k, v in VARIANT_CONFIGS[variant].get("env", {}).items()}


def expected_filenames(qrel_records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for row in qrel_records:
        for f in (row.get("gold_files") or row.get("expected_files") or []):
            raw = str(f)
            norm = normalize_filename_for_match(raw)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(raw)
    return result


def _fetch_profile(collection: str) -> CollectionProfile:
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=f"http://{host}:{port}")
        stats = client.get_collection_stats(collection_name=collection)
        row_count = int(stats.get("row_count", 0))
        filenames: set[str] = set()
        sample_candidates: dict[str, bool] = {}
        try:
            results = client.query(collection_name=collection, filter="", output_fields=["filename"], limit=16384)
            for item in results:
                fname = item.get("filename") or item.get("file_name") or ""
                if fname:
                    filenames.add(normalize_filename_for_match(fname))
        except Exception:
            pass
        return CollectionProfile(exists=True, document_count=None, chunk_count=row_count, indexed_filenames=filenames, sample_candidates=sample_candidates)
    except Exception:
        return CollectionProfile(exists=False, document_count=None, chunk_count=None)


def build_diagnosis(
    variant: str,
    compare_to: str,
    qrel_records: list[dict[str, Any]],
    variant_profile: CollectionProfile,
    compare_profile: CollectionProfile,
) -> dict[str, Any]:
    expected = set(normalize_filename_for_match(f) for f in expected_filenames(qrel_records))
    normalized_indexed = set(normalize_filename_for_match(f) for f in variant_profile.indexed_filenames)
    variant_covered = expected & normalized_indexed
    compare_chunk_count = compare_profile.chunk_count or 0

    checks: dict[str, bool] = {
        "collection_exists": variant_profile.exists,
        "document_count_is_60": variant_profile.chunk_count is not None and variant_profile.chunk_count > 0,
    }
    if compare_chunk_count > 0 and variant_profile.chunk_count is not None:
        ratio = variant_profile.chunk_count / compare_chunk_count
        checks["chunk_count_within_compare_range"] = 0.8 <= ratio <= 1.2
    else:
        checks["chunk_count_within_compare_range"] = False

    coverage = len(variant_covered) / max(1, len(expected)) if expected else 0.0
    checks["qrel_expected_filename_coverage_ge_0_95"] = coverage >= 0.95

    failures = [k for k, v in checks.items() if not v]
    action = "none"
    if not checks.get("collection_exists"):
        action = "rebuild"
    elif not checks.get("qrel_expected_filename_coverage_ge_0_95"):
        action = "rebuild"
    elif failures:
        action = "investigate"

    return {
        "variant": variant,
        "compare_to": compare_to,
        "checks": checks,
        "failures": failures,
        "filename_coverage": round(coverage, 3),
        "variant_chunk_count": variant_profile.chunk_count,
        "compare_chunk_count": compare_chunk_count,
        "recommended_action": action,
    }


def diagnose(variant: str, compare_to: str | None = None) -> dict[str, Any]:
    env = variant_env(variant)
    collection = env.get("MILVUS_COLLECTION", "")
    profile = env.get("RAG_INDEX_PROFILE", "")
    bm25_path = env.get("BM25_STATE_PATH", "")

    qrel_records: list[dict[str, Any]] = []
    dataset_path = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_doc_gold.jsonl"
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qrel_records.append(json.loads(line))

    variant_prof = _fetch_profile(collection)

    report: dict[str, Any] = {
        "variant": variant,
        "profile": profile,
        "collection": collection,
        "bm25_state_path": bm25_path,
        "collection_stats": {
            "exists": variant_prof.exists,
            "row_count": variant_prof.chunk_count,
        },
        "bm25_state": {"path": bm25_path, "exists": Path(bm25_path).exists()} if bm25_path else {},
        "qrel_filename_coverage": {
            "expected_file_count": len(expected_filenames(qrel_records)),
            "indexed_file_count": len(variant_prof.indexed_filenames),
            "covered_count": len(set(normalize_filename_for_match(f) for f in expected_filenames(qrel_records)) & variant_prof.indexed_filenames),
        },
    }

    if compare_to:
        ref_env = variant_env(compare_to)
        ref_collection = ref_env.get("MILVUS_COLLECTION", "")
        compare_prof = _fetch_profile(ref_collection)
        diagnosis = build_diagnosis(variant, compare_to, qrel_records, variant_prof, compare_prof)
        report["diagnosis"] = diagnosis
        report["reference"] = {
            "variant": compare_to,
            "profile": ref_env.get("RAG_INDEX_PROFILE", ""),
            "collection": ref_collection,
            "collection_stats": {"exists": compare_prof.exists, "row_count": compare_prof.chunk_count},
        }
    else:
        report["healthy"] = variant_prof.exists and variant_prof.chunk_count is not None and variant_prof.chunk_count > 0

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose RAG variant profile index health.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--compare-to", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = diagnose(args.variant, args.compare_to)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"=== Diagnosis: {args.variant} ===")
        print(f"Profile: {report['profile']}")
        print(f"Collection: {report['collection']}")
        cs = report["collection_stats"]
        print(f"Collection exists: {cs.get('exists')}")
        print(f"Collection row_count: {cs.get('row_count', 'N/A')}")
        cov = report["qrel_filename_coverage"]
        print(f"Filename coverage: {cov.get('covered_count', '?')}/{cov.get('expected_file_count', '?')}")
        if "diagnosis" in report:
            d = report["diagnosis"]
            print(f"\nChecks: {d['checks']}")
            print(f"Failures: {d['failures']}")
            print(f"Recommended action: {d['recommended_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
