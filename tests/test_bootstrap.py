import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"


class BootstrapTests(unittest.TestCase):
    def run_python(self, code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )

    def test_database_defaults_to_docker_postgres(self):
        env = os.environ.copy()
        env.pop("DATABASE_URL", None)
        env.pop("FALLBACK_DATABASE_URL", None)
        env["PYTHONPATH"] = str(BACKEND_DIR)

        result = self.run_python(
            "import database; print(database.DATABASE_URL)",
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(
            result.stdout.strip().startswith("postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/langchain_app"),
            msg=result.stdout.strip() or result.stderr,
        )

    def test_init_db_falls_back_to_sqlite_when_primary_database_is_unavailable(self):
        env = os.environ.copy()
        fallback_path = PROJECT_ROOT / "data" / "test-bootstrap-fallback.db"
        env["DATABASE_URL"] = "postgresql+psycopg2://postgres:wrong-password@127.0.0.1:5432/langchain_app"
        env["FALLBACK_DATABASE_URL"] = f"sqlite:///{fallback_path.as_posix()}"
        env["PYTHONPATH"] = str(BACKEND_DIR)
        fallback_path.unlink(missing_ok=True)

        try:
            result = self.run_python(
                "import database; database.init_db(); print(database.DATABASE_URL)",
                env=env,
            )
        finally:
            fallback_path.unlink(missing_ok=True)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(
            result.stdout.strip().startswith(f"sqlite:///{fallback_path.as_posix()}"),
            msg=result.stdout.strip() or result.stderr,
        )

    def test_app_import_does_not_eagerly_initialize_remote_dependencies(self):
        env = os.environ.copy()
        env["DATABASE_URL"] = "sqlite:///:memory:"

        stub_root = PROJECT_ROOT / ".test-stubs"
        shutil.rmtree(stub_root, ignore_errors=True)
        (stub_root / "langchain").mkdir(parents=True, exist_ok=True)
        try:
            (stub_root / "langchain_huggingface.py").write_text(
                textwrap.dedent(
                    """
                    raise RuntimeError("EMBEDDING_MODULE_IMPORTED_AT_APP_IMPORT")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (stub_root / "langchain" / "__init__.py").write_text("", encoding="utf-8")
            (stub_root / "langchain" / "chat_models.py").write_text(
                textwrap.dedent(
                    """
                    raise RuntimeError("CHAT_MODEL_MODULE_IMPORTED_AT_APP_IMPORT")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (stub_root / "langchain" / "agents.py").write_text(
                textwrap.dedent(
                    """
                    raise RuntimeError("AGENT_MODULE_IMPORTED_AT_APP_IMPORT")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            env["PYTHONPATH"] = os.pathsep.join([str(stub_root), str(BACKEND_DIR)])

            result = self.run_python(
                "import app; print(app.app.title)",
                env=env,
            )
        finally:
            shutil.rmtree(stub_root, ignore_errors=True)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SuperHermes", result.stdout)


if __name__ == "__main__":
    unittest.main()
