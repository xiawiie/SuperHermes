from __future__ import annotations

from pathlib import Path


TEXT_EXTENSIONS = {
    ".css",
    ".env",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".txt",
}

SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}

SKIP_PREFIXES = {
    ("backend", "data"),
    ("eval", "datasets"),
    ("eval", "qrels"),
    ("eval", "reports"),
    ("eval", "reviews"),
}

MOJIBAKE_MARKERS = tuple(
    "".join(chr(codepoint) for codepoint in codepoints)
    for codepoints in (
        (0x9225,),
        (0x9242,),
        (0x9241,),
        (0x951B,),
        (0x93C2,),
        (0x9428,),
        (0x7487,),
        (0x69D1,),
        (0x6D30,),
        (0x7ECB, 0x20AC),
        (0xFFFD,),
    )
)


def _is_skipped(path: Path) -> bool:
    parts = path.parts
    if any(part in SKIP_PARTS for part in parts):
        return True
    return any(parts[: len(prefix)] == prefix for prefix in SKIP_PREFIXES)


def test_user_visible_text_files_do_not_contain_mojibake() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in repo_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS or _is_skipped(relative):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in MOJIBAKE_MARKERS):
                failures.append(f"{relative}:{line_number}: {line[:120]}")

    assert not failures, "Mojibake markers found:\n" + "\n".join(failures[:20])
