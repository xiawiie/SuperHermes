"""Derive an open-retrieval-natural dataset from the gold dataset.

Removes 《...》 markers from queries to simulate natural user questions
without explicit document hints. This creates a harder evaluation set
where the system must identify the correct document without the
book-title signal.

Usage:
    python scripts/derive_natural_query_subset.py
    python scripts/derive_natural_query_subset.py --input .jbeval/datasets/rag_doc_gold.jsonl
    python scripts/derive_natural_query_subset.py --output .jbeval/datasets/rag_doc_gold_natural_v1.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_doc_gold.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_doc_gold_natural_v1.jsonl"

_BOOK_TITLE_RE = re.compile(r"《[^》]+》中[，,]?\s*")
_HEADING_PREFIX_RE = re.compile(r"^(如何|怎么|怎样|方法|步骤|操作|配置|设置|查看|修改|删除|添加|开启|关闭|恢复|登录|退出|重启|升级|备份|恢复|安装|卸载|连接|断开|启用|禁用|检查|测试|验证|调试|排查|解决|处理|清理|更新|调整|切换|绑定|解绑|注册|注销|下载|上传|导入|导出|保存|打印|分享|搜索|过滤|排序|分组|统计|分析|监控|告警|通知|授权|认证|加密|解密|签名|验签)\s*")


def _strip_book_title(query: str) -> str:
    """Remove 《...》中， prefix from a query, preserving the semantic core."""
    cleaned = _BOOK_TITLE_RE.sub("", query).strip()
    if not cleaned:
        return query
    return cleaned


def _naturalize_record(record: dict) -> dict:
    """Create a natural-query variant of a gold record."""
    natural = copy.deepcopy(record)

    original_query = str(record.get("query") or record.get("question") or "")
    natural_query = _strip_book_title(original_query)

    natural["query"] = natural_query
    if "question" in natural:
        natural["question"] = natural_query

    natural["original_query"] = original_query
    natural["generation_method"] = "naturalized_from_gold_v1"
    natural["dataset_type"] = "open-retrieval-natural"
    natural["benchmark_schema_version"] = record.get("benchmark_schema_version", "rag-doc-gold-v2")

    return natural


def derive_natural_dataset(input_path: Path, output_path: Path) -> int:
    """Read gold dataset and write natural-query variant."""
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    natural_records = [_naturalize_record(r) for r in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in natural_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Derived {len(natural_records)} natural-query records", flush=True)
    print(f"  from: {input_path}", flush=True)
    print(f"  to:   {output_path}", flush=True)

    sample_count = min(5, len(natural_records))
    for i in range(sample_count):
        orig = records[i].get("query") or records[i].get("question") or ""
        nat = natural_records[i].get("query") or ""
        print(f"  sample {i+1}: '{orig}' -> '{nat}'", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive open-retrieval-natural dataset from gold dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to gold dataset JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output natural dataset JSONL",
    )
    args = parser.parse_args()
    return derive_natural_dataset(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
