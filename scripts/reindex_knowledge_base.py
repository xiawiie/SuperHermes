from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cache import cache  # noqa: E402
from database import SessionLocal  # noqa: E402
from document_loader import DocumentLoader  # noqa: E402
from embedding import EmbeddingService  # noqa: E402
from milvus_client import MilvusManager  # noqa: E402
from milvus_writer import MilvusWriter  # noqa: E402
from models import ParentChunk  # noqa: E402
from parent_chunk_store import ParentChunkStore  # noqa: E402


def main() -> int:
    documents_dir = PROJECT_ROOT / "data" / "documents"
    if not documents_dir.is_dir():
        print(f"Documents folder not found: {documents_dir}")
        return 1

    state_path = PROJECT_ROOT / "data" / "bm25_state.json"
    if state_path.exists():
        backup_path = state_path.with_name(
            f"{state_path.stem}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}{state_path.suffix}"
        )
        shutil.copy2(state_path, backup_path)
        state_path.unlink()
        print(f"Backed up BM25 state to {backup_path}")

    milvus_manager = MilvusManager()
    try:
        milvus_manager.drop_collection()
        print("Dropped existing Milvus collection")
    except Exception as exc:
        print(f"Skip dropping Milvus collection: {exc}")

    db = SessionLocal()
    try:
        db.query(ParentChunk).delete(synchronize_session=False)
        db.commit()
        print("Cleared parent chunk table")
    finally:
        db.close()

    try:
        cache.delete_pattern("parent_chunk:*")
    except Exception:
        pass

    embedding_service = EmbeddingService(state_path=state_path)
    loader = DocumentLoader()
    parent_store = ParentChunkStore()
    writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)
    print(f"Retrieval text mode: {loader.retrieval_text_mode}")

    supported_files = sorted(
        path for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".doc", ".docx", ".xls", ".xlsx"}
    )
    if not supported_files:
        print(f"No supported documents found in {documents_dir}")
        return 1

    total_leaf = 0
    total_parent = 0
    for path in supported_files:
        print(f"Indexing {path.name} ...")
        docs = loader.load_document(str(path), path.name)
        parent_docs = [doc for doc in docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in docs if int(doc.get("chunk_level", 0) or 0) == 3]
        parent_store.upsert_documents(parent_docs)
        writer.write_documents(leaf_docs)
        total_parent += len(parent_docs)
        total_leaf += len(leaf_docs)
        print(f"  parent={len(parent_docs)} leaf={len(leaf_docs)}")

    print(f"Done. documents={len(supported_files)} parent={total_parent} leaf={total_leaf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
