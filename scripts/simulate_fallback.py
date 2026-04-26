from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def _trigger(strategy: str, gs3_row: dict[str, Any]) -> bool:
    metrics = gs3_row.get("metrics") or {}
    if strategy == "strict":
        return bool(gs3_row.get("fallback_required") or metrics.get("fallback_triggered"))
    if strategy == "medium":
        if bool(gs3_row.get("fallback_required") or metrics.get("fallback_triggered")):
            return True
        top_score = (gs3_row.get("trace") or {}).get("confidence", {}).get("top_score")
        return isinstance(top_score, (int, float)) and float(top_score) < 0.20
    if strategy == "broad":
        reasons = (gs3_row.get("trace") or {}).get("confidence", {}).get("reasons") or []
        return len(reasons) > 0
    raise ValueError(f"Unknown strategy: {strategy}")


def _metric(row: dict[str, Any], key: str) -> float:
    v = (row.get("metrics") or {}).get(key)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def simulate(rows: list[dict[str, Any]], gs3_variant: str, v3q_variant: str) -> dict[str, Any]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_sample[str(r.get("sample_id"))][str(r.get("variant"))] = r

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for variants in by_sample.values():
        g = variants.get(gs3_variant)
        q = variants.get(v3q_variant)
        if g and q:
            pairs.append((g, q))

    out: dict[str, Any] = {"rows": len(pairs), "strategies": {}}
    for strategy in ("strict", "medium", "broad"):
        blended_file: list[float] = []
        blended_file_page: list[float] = []
        blended_chunk: list[float] = []
        optimistic_lat: list[float] = []
        conservative_lat: list[float] = []
        fallback_count = 0

        for gs3, v3q in pairs:
            use_fallback = _trigger(strategy, gs3)
            chosen = v3q if use_fallback else gs3
            if use_fallback:
                fallback_count += 1

            blended_file.append(_metric(chosen, "file_hit_at_5"))
            blended_file_page.append(_metric(chosen, "file_page_hit_at_5"))
            blended_chunk.append(_metric(chosen, "chunk_hit_at_5"))

            gs3_lat = float(gs3.get("latency_ms") or 0.0)
            v3q_lat = float(v3q.get("latency_ms") or 0.0)
            optimistic_lat.append(v3q_lat if use_fallback else gs3_lat)
            conservative_lat.append((gs3_lat + v3q_lat) if use_fallback else gs3_lat)

        n = max(1, len(pairs))
        fallback_rate = fallback_count / n
        result = {
            "fallback_rate": fallback_rate,
            "file_at_5": _avg(blended_file),
            "file_page_at_5": _avg(blended_file_page),
            "chunk_at_5": _avg(blended_chunk),
            "optimistic_p50_ms": _p(optimistic_lat, 0.50),
            "optimistic_p95_ms": _p(optimistic_lat, 0.95),
            "conservative_p50_ms": _p(conservative_lat, 0.50),
            "conservative_p95_ms": _p(conservative_lat, 0.95),
        }
        result["quality_gain_per_10pct_fallback"] = (
            (result["file_page_at_5"] or 0.0) / fallback_rate * 0.10 if fallback_rate > 0 else 0.0
        )
        out["strategies"][strategy] = result
    return out


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Fallback Simulation",
        "",
        f"- Rows: `{report.get('rows', 0)}`",
        "",
        "| Strategy | FallbackRate | File@5 | File+Page@5 | Chunk@5 | Opt P50 | Opt P95 | Cons P50 | Cons P95 | Gain/10% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, m in (report.get("strategies") or {}).items():
        lines.append(
            "| {s} | {fr:.3f} | {f:.3f} | {fp:.3f} | {c:.3f} | {op50:.1f} | {op95:.1f} | {cp50:.1f} | {cp95:.1f} | {gain:.3f} |".format(
                s=strategy,
                fr=float(m.get("fallback_rate") or 0.0),
                f=float(m.get("file_at_5") or 0.0),
                fp=float(m.get("file_page_at_5") or 0.0),
                c=float(m.get("chunk_at_5") or 0.0),
                op50=float(m.get("optimistic_p50_ms") or 0.0),
                op95=float(m.get("optimistic_p95_ms") or 0.0),
                cp50=float(m.get("conservative_p50_ms") or 0.0),
                cp95=float(m.get("conservative_p95_ms") or 0.0),
                gain=float(m.get("quality_gain_per_10pct_fallback") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate GS3 + V3Q fallback strategies.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gs3-variant", default="GS3")
    parser.add_argument("--v3q-variant", default="V3Q")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    report = simulate(rows, args.gs3_variant, args.v3q_variant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fallback_simulation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "fallback_simulation.md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
