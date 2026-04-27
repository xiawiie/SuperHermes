from __future__ import annotations


def dedupe_filenames(filenames: list[str] | None, max_count: int | None = None) -> list[str]:
    seen = set()
    result = []
    for name in (filenames or []):
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result[:max_count] if max_count else result
