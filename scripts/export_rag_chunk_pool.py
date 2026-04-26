from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.rag_dataset_utils import write_jsonl
from scripts.rag_qrels import attach_canonical_ids


DEFAULT_OUTPUT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_chunk_pool_v1.jsonl"
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}
CHUNK_OUTPUT_FIELDS = [
    "text",
    "retrieval_text",
    "filename",
    "file_type",
    "file_path",
    "page_number",
    "page_start",
    "page_end",
    "chunk_id",
    "parent_chunk_id",
    "root_chunk_id",
    "chunk_level",
    "chunk_role",
    "chunk_idx",
    "section_title",
    "section_type",
    "section_path",
    "anchor_id",
]


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    page_number = _to_int(chunk.get("page_number"))
    normalized = {
        "text": str(chunk.get("text") or ""),
        "retrieval_text": str(chunk.get("retrieval_text") or chunk.get("text") or ""),
        "filename": str(chunk.get("filename") or chunk.get("file_name") or ""),
        "file_type": str(chunk.get("file_type") or ""),
        "file_path": str(chunk.get("file_path") or ""),
        "page_number": page_number,
        "page_start": _to_int(chunk.get("page_start"), page_number),
        "page_end": _to_int(chunk.get("page_end"), page_number),
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "parent_chunk_id": str(chunk.get("parent_chunk_id") or ""),
        "root_chunk_id": str(chunk.get("root_chunk_id") or ""),
        "chunk_level": _to_int(chunk.get("chunk_level")),
        "chunk_role": str(chunk.get("chunk_role") or ""),
        "chunk_idx": _to_int(chunk.get("chunk_idx")),
        "section_title": str(chunk.get("section_title") or ""),
        "section_type": str(chunk.get("section_type") or ""),
        "section_path": str(chunk.get("section_path") or ""),
        "anchor_id": str(chunk.get("anchor_id") or ""),
    }
    return attach_canonical_ids(normalized)


def export_from_milvus(manager: object | None = None, filter_expr: str = "") -> list[dict[str, Any]]:
    if manager is None:
        from milvus_client import MilvusManager

        manager = MilvusManager()
    rows = manager.query_all(filter_expr=filter_expr, output_fields=CHUNK_OUTPUT_FIELDS)
    return [normalize_chunk(row) for row in rows if row.get("chunk_id")]


def export_from_documents(
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
    loader: object | None = None,
    include_parents: bool = False,
) -> list[dict[str, Any]]:
    if loader is None:
        from document_loader import DocumentLoader

        loader = DocumentLoader()
    if not documents_dir.is_dir():
        raise FileNotFoundError(documents_dir)

    rows: list[dict[str, Any]] = []
    for path in sorted(documents_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue
        for chunk in loader.load_document(str(path), path.name):
            if include_parents or _to_int(chunk.get("chunk_level")) == 3:
                rows.append(normalize_chunk(chunk))
    return rows


def write_chunk_pool(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (row.get("filename", ""), row.get("chunk_idx", 0), row.get("chunk_id", "")))
    write_jsonl(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RAG chunk pool for dataset alignment and negative mining.")
    parser.add_argument("--source", choices=["milvus", "documents"], default="milvus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--filter", default="", help="Milvus filter expression when --source=milvus.")
    parser.add_argument("--include-parents", action="store_true", help="Include root/parent chunks for document-loader export.")
    args = parser.parse_args()

    if args.source == "milvus":
        rows = export_from_milvus(filter_expr=args.filter)
    else:
        rows = export_from_documents(args.documents_dir, include_parents=args.include_parents)

    write_chunk_pool(args.output, rows)
    print(f"Wrote {len(rows)} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
