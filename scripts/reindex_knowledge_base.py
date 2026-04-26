from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_SUFFIXES = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}
RETRIEVAL_TEXT_MODES = {"raw", "title_context", "title_context_filename"}
V3_PROFILE_DEFAULTS = {
    "gold_tc": {
        "collection": "embeddings_collection_gold_tc",
        "text_mode": "title_context",
        "state_path": PROJECT_ROOT / "data" / "bm25_state_gold_tc.json",
    },
    "gold_tcf": {
        "collection": "embeddings_collection_gold_tcf",
        "text_mode": "title_context_filename",
        "state_path": PROJECT_ROOT / "data" / "bm25_state_gold_tcf.json",
    },
    "v3_quality": {
        "collection": "embeddings_collection_v3_quality",
        "text_mode": "title_context_filename",
        "state_path": PROJECT_ROOT / "data" / "bm25_state_v3_quality.json",
    },
    "v3_fast": {
        "collection": "embeddings_collection_v3_fast",
        "text_mode": "title_context_filename",
        "state_path": PROJECT_ROOT / "data" / "bm25_state_v3_fast.json",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a SuperHermes Milvus collection from data/documents.",
    )
    parser.add_argument("--documents-dir", type=Path, default=PROJECT_ROOT / "data" / "documents")
    parser.add_argument(
        "--index-profile",
        default=os.getenv("RAG_INDEX_PROFILE", "legacy"),
        help="Logical index profile used to namespace BM25/parent chunks/cache.",
    )
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "embeddings_collection"))
    parser.add_argument(
        "--text-mode",
        choices=sorted(RETRIEVAL_TEXT_MODES),
        default=os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N supported documents.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Load and validate chunks without writing Milvus.")
    parser.add_argument(
        "--skip-drop",
        action="store_true",
        help="Do not drop the target collection before writing. Use only for append diagnostics.",
    )
    parser.add_argument(
        "--preserve-parent-store",
        action="store_true",
        help="Do not clear the shared parent chunk table/cache before writing.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Override BM25 state path. By default it is derived from collection and text mode.",
    )
    parser.add_argument(
        "--no-state-backup",
        action="store_true",
        help="Delete an existing BM25 state without writing a timestamped backup first.",
    )
    args = parser.parse_args()
    defaults = V3_PROFILE_DEFAULTS.get(str(args.index_profile))
    if defaults:
        if parser.get_default("collection") == args.collection:
            args.collection = str(defaults["collection"])
        if parser.get_default("text_mode") == args.text_mode:
            args.text_mode = str(defaults["text_mode"])
        if args.state_path is None:
            args.state_path = Path(defaults["state_path"])
    return args


def _configure_env(args: argparse.Namespace) -> None:
    os.environ["MILVUS_COLLECTION"] = args.collection
    os.environ["EVAL_RETRIEVAL_TEXT_MODE"] = args.text_mode
    os.environ["RAG_INDEX_PROFILE"] = args.index_profile
    if args.state_path is not None:
        os.environ["BM25_STATE_PATH"] = str(args.state_path)


def _import_backend() -> dict[str, Any]:
    from backend.infra.cache import cache
    from backend.infra.db.database import init_db
    from backend.documents.loader import DocumentLoader
    from backend.infra.embedding import EmbeddingService
    from backend.infra.vector_store.milvus_client import MilvusManager
    from backend.infra.vector_store.milvus_writer import MilvusWriter
    from backend.infra.vector_store.parent_chunk_store import ParentChunkStore

    return {
        "cache": cache,
        "init_db": init_db,
        "DocumentLoader": DocumentLoader,
        "EmbeddingService": EmbeddingService,
        "MilvusManager": MilvusManager,
        "MilvusWriter": MilvusWriter,
        "ParentChunkStore": ParentChunkStore,
    }


def _supported_files(documents_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        path
        for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if limit is not None:
        files = files[: max(0, limit)]
    return files


def _reset_state_file(state_path: Path, backup: bool) -> None:
    if not state_path.exists():
        return
    if backup:
        backup_path = state_path.with_name(
            f"{state_path.stem}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}{state_path.suffix}"
        )
        shutil.copy2(state_path, backup_path)
        print(f"Backed up BM25 state to {backup_path}", flush=True)
    state_path.unlink()
    print(f"Removed BM25 state {state_path}", flush=True)


def _clear_parent_store(backend: dict[str, Any], index_profile: str) -> None:
    cache = backend["cache"]
    ParentChunkStore = backend["ParentChunkStore"]

    deleted = ParentChunkStore(index_profile=index_profile).delete_by_profile()
    print(f"Cleared parent chunk table profile={index_profile} rows={deleted}", flush=True)

    try:
        cache.delete_pattern(f"parent_chunk:{index_profile}:*")
        if index_profile == "legacy":
            cache.delete_pattern("parent_chunk:*")
        print(f"Cleared parent chunk cache profile={index_profile}", flush=True)
    except Exception:
        pass


def _validate_docs(docs: list[dict], source: Path) -> tuple[int, int]:
    parent_count = 0
    leaf_count = 0
    for doc in docs:
        retrieval_text = str(doc.get("retrieval_text") or "")
        if len(retrieval_text) > 4000:
            raise ValueError(f"{source.name} produced retrieval_text longer than 4000 chars")
        level = int(doc.get("chunk_level", 0) or 0)
        if level in (1, 2):
            parent_count += 1
        elif level == 3:
            leaf_count += 1
    return parent_count, leaf_count


def main() -> int:
    args = _parse_args()
    _configure_env(args)

    documents_dir = args.documents_dir.resolve()
    if not documents_dir.is_dir():
        print(f"Documents folder not found: {documents_dir}", flush=True)
        return 1

    files = _supported_files(documents_dir, args.limit)
    if not files:
        print(f"No supported documents found in {documents_dir}", flush=True)
        return 1

    backend = _import_backend()
    backend["init_db"]()
    DocumentLoader = backend["DocumentLoader"]
    EmbeddingService = backend["EmbeddingService"]
    MilvusManager = backend["MilvusManager"]
    MilvusWriter = backend["MilvusWriter"]
    ParentChunkStore = backend["ParentChunkStore"]

    loader = DocumentLoader()
    embedding_service = EmbeddingService(state_path=args.state_path)
    state_path = Path(getattr(embedding_service, "_state_path"))

    print(
        "Reindex config: "
        f"profile={args.index_profile} collection={args.collection} text_mode={loader.retrieval_text_mode} "
        f"documents={len(files)} dry_run={args.dry_run} state_path={state_path}",
        flush=True,
    )

    if args.dry_run:
        total_parent = 0
        total_leaf = 0
        for path in files:
            docs = loader.load_document(str(path), path.name)
            parent_count, leaf_count = _validate_docs(docs, path)
            total_parent += parent_count
            total_leaf += leaf_count
            print(f"Validated {path.name}: parent={parent_count} leaf={leaf_count}", flush=True)
        print(f"Dry run complete. documents={len(files)} parent={total_parent} leaf={total_leaf}", flush=True)
        return 0

    _reset_state_file(state_path, backup=not args.no_state_backup)

    milvus_manager = MilvusManager()
    if not args.skip_drop:
        try:
            milvus_manager.drop_collection()
            print(f"Dropped Milvus collection {args.collection}", flush=True)
        except Exception as exc:
            print(f"Skip dropping Milvus collection {args.collection}: {exc}", flush=True)

    if not args.preserve_parent_store:
        _clear_parent_store(backend, args.index_profile)

    embedding_service = EmbeddingService(state_path=state_path)
    parent_store = ParentChunkStore(index_profile=args.index_profile)
    writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

    total_leaf = 0
    total_parent = 0
    for path in files:
        print(f"Indexing {path.name} ...", flush=True)
        docs = loader.load_document(str(path), path.name)
        _validate_docs(docs, path)
        parent_docs = [doc for doc in docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in docs if int(doc.get("chunk_level", 0) or 0) == 3]
        parent_store.upsert_documents(parent_docs)
        writer.write_documents(leaf_docs, batch_size=args.batch_size)
        total_parent += len(parent_docs)
        total_leaf += len(leaf_docs)
        print(f"  parent={len(parent_docs)} leaf={len(leaf_docs)}", flush=True)

    print(f"Done. documents={len(files)} parent={total_parent} leaf={total_leaf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
