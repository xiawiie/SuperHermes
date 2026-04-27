from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

CANONICAL_MODULES = [
    "backend.api",
    "backend.app",
    "backend.application.main",
    "backend.chat.agent",
    "backend.chat.tools",
    "backend.contracts.schemas",
    "backend.documents.loader",
    "backend.evaluation.answer_eval",
    "backend.infra.cache",
    "backend.infra.embedding",
    "backend.infra.db.database",
    "backend.infra.db.models",
    "backend.infra.db.conversation_storage",
    "backend.infra.vector_store.milvus_client",
    "backend.infra.vector_store.milvus_writer",
    "backend.infra.vector_store.parent_chunk_store",
    "backend.rag.utils",
    "backend.rag.pipeline",
    "backend.rag.query_plan",
    "backend.rag.diagnostics",
    "backend.rag.trace",
    "backend.security.auth",
]

LEGACY_MODULES = [
    "agent",
    "answer_eval",
    "auth",
    "cache",
    "conversation_storage",
    "database",
    "document_loader",
    "embedding",
    "filename_normalization",
    "json_utils",
    "milvus_client",
    "milvus_writer",
    "models",
    "parent_chunk_store",
    "query_plan",
    "rag_confidence",
    "rag_context",
    "rag_diagnostics",
    "rag_pipeline",
    "rag_profiles",
    "rag_rerank",
    "rag_retrieval",
    "rag_trace",
    "rag_types",
    "rag_utils",
    "tools",
]

FORBIDDEN_ROOT_FILES = {f"{name}.py" for name in LEGACY_MODULES}
ALLOWED_ROOT_FILES = {"__init__.py", "api.py", "app.py"}


@pytest.mark.parametrize("module_name", CANONICAL_MODULES)
def test_canonical_backend_imports_are_available(module_name: str):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", LEGACY_MODULES)
def test_legacy_bare_imports_fail_with_backend_on_pythonpath(module_name: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module_name!r})"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0, result.stdout
    assert "ModuleNotFoundError" in result.stderr


def test_backend_root_has_no_legacy_alias_files():
    root_files = {path.name for path in BACKEND_ROOT.iterdir() if path.is_file()}

    assert not (root_files & FORBIDDEN_ROOT_FILES)
    assert ALLOWED_ROOT_FILES <= root_files


def test_backend_app_import_smoke():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", "from backend.app import app; assert app"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
