import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = BASE_DIR / "data" / "superhermes.db"
DEFAULT_DOCKER_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/langchain_app"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    DEFAULT_DOCKER_DATABASE_URL,
)
FALLBACK_DATABASE_URL = os.getenv(
    "FALLBACK_DATABASE_URL",
    f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}",
)
ALLOW_DATABASE_FALLBACK = os.getenv("ALLOW_DATABASE_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}

Base = declarative_base()


def _build_engine(database_url: str):
    engine_options = {
        "pool_pre_ping": True,
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT_SECONDS", "30")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800")),
    }

    if database_url.startswith("sqlite"):
        for key in ("pool_size", "max_overflow", "pool_timeout", "pool_recycle"):
            engine_options.pop(key, None)

    return create_engine(database_url, **engine_options)


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    parsed = make_url(database_url)
    if parsed.drivername != "sqlite" or not parsed.database or parsed.database == ":memory:":
        return
    Path(parsed.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _configure_engine(database_url: str) -> None:
    global DATABASE_URL, engine
    DATABASE_URL = database_url
    _ensure_sqlite_parent_dir(database_url)
    engine = _build_engine(database_url)
    SessionLocal.configure(bind=engine)


engine = _build_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def _ensure_runtime_indexes() -> None:
    if engine.dialect.name != "postgresql":
        return

    statements = [
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_updated ON chat_sessions (user_id, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id_order ON chat_messages (session_ref_id, id)",
        "CREATE INDEX IF NOT EXISTS ix_parent_chunks_filename_chunk ON parent_chunks (filename, chunk_id)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    import models  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine)
        _ensure_runtime_indexes()
    except OperationalError:
        should_fallback = (
            ALLOW_DATABASE_FALLBACK
            and not DATABASE_URL.startswith("sqlite")
            and FALLBACK_DATABASE_URL
            and FALLBACK_DATABASE_URL != DATABASE_URL
        )
        if not should_fallback:
            raise
        _configure_engine(FALLBACK_DATABASE_URL)
        Base.metadata.create_all(bind=engine)
        _ensure_runtime_indexes()
