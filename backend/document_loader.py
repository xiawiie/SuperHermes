"""Document loading and chunking service."""
from __future__ import annotations

import os
import re
from typing import Dict, List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Load source documents and emit root / leaf chunks."""

    _TITLE_BLACKLIST = {
        "目录",
        "前言",
        "附录",
        "附录a",
        "附录b",
        "附件",
        "封面",
        "修订记录",
        "目录页",
    }
    _ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]\s*")
    _DECIMAL_PATTERN = re.compile(r"^\d+(?:\.\d+){0,4}\s+\S+")
    _LIST_PATTERN = re.compile(r"^[一二三四五六七八九十]+、\s*\S+")
    _PAREN_LIST_PATTERN = re.compile(r"^[（(][一二三四五六七八九十0-9A-Za-z]+[)）]\s*\S+")
    _ANCHOR_PATTERN = re.compile(
        r"(第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]|"
        r"\d+(?:\.\d+){1,4}|"
        r"[一二三四五六七八九十]+、|"
        r"[（(][一二三四五六七八九十0-9A-Za-z]+[)）]|"
        r"附录[A-Za-z0-9一二三四五六七八九十]+|"
        r"附件[0-9一二三四五六七八九十]+)"
    )

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.retrieval_text_mode = os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context").strip().lower()
        self._root_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(1200, chunk_size * 2),
            chunk_overlap=max(240, chunk_overlap * 2),
            add_start_index=True,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self._leaf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(320, chunk_size),
            chunk_overlap=max(60, chunk_overlap),
            add_start_index=True,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

    @classmethod
    def _normalize_title(cls, value: str | None) -> str:
        text = re.sub(r"\s+", " ", (value or "").strip())
        return text.strip(" -:：|")

    @classmethod
    def _extract_anchor_id(cls, title: str | None) -> str:
        normalized = cls._normalize_title(title)
        if not normalized:
            return ""
        match = cls._ANCHOR_PATTERN.search(normalized)
        return match.group(0) if match else ""

    @classmethod
    def _is_informative_title(cls, title: str | None) -> bool:
        normalized = cls._normalize_title(title)
        if not normalized:
            return False
        if normalized.lower() in cls._TITLE_BLACKLIST:
            return False
        if normalized.isdigit():
            return False
        if cls._extract_anchor_id(normalized):
            return True
        if len(normalized) < 2:
            return False
        stripped = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
        return len(stripped) >= 3

    def _compose_retrieval_text(
        self,
        body: str,
        current_title: str | None,
        parent_title: str | None,
    ) -> str:
        content = (body or "").strip()
        current = self._normalize_title(current_title)
        parent = self._normalize_title(parent_title)
        if not self._is_informative_title(current):
            current = ""
        if not self._is_informative_title(parent):
            parent = ""

        if current and parent and current.lower() in {"概述", "说明", "注意事项", "简介"}:
            heading = f"{parent} > {current}"
        elif current:
            heading = current
        elif parent:
            heading = parent
        else:
            heading = ""

        if heading and content:
            return f"{heading}\n{content}"
        return heading or content

    def _make_retrieval_text(
        self,
        raw_text: str,
        body: str,
        current_title: str | None,
        parent_title: str | None,
    ) -> str:
        if self.retrieval_text_mode == "raw":
            return (raw_text or body or "").strip()
        return self._compose_retrieval_text(
            body=body,
            current_title=current_title,
            parent_title=parent_title,
        )

    @classmethod
    def _heading_depth(cls, line: str) -> int:
        normalized = cls._normalize_title(line)
        if not normalized:
            return 0
        if cls._ARTICLE_PATTERN.match(normalized):
            if "编" in normalized or "部分" in normalized or "章" in normalized:
                return 1
            if "节" in normalized:
                return 2
            return 3
        match = re.match(r"^(\d+(?:\.\d+){0,4})\s+\S+", normalized)
        if match:
            return match.group(1).count(".") + 1
        if cls._LIST_PATTERN.match(normalized):
            return 1
        if cls._PAREN_LIST_PATTERN.match(normalized):
            return 2
        return 0

    @classmethod
    def _is_heading_candidate(cls, line: str) -> bool:
        normalized = cls._normalize_title(line)
        if not normalized or len(normalized) > 80:
            return False
        return cls._heading_depth(normalized) > 0

    def _detect_profile(self, raw_docs: list) -> str:
        sample_lines: list[str] = []
        for doc in raw_docs[:3]:
            for line in (doc.page_content or "").splitlines():
                normalized = self._normalize_title(line)
                if normalized:
                    sample_lines.append(normalized)
                if len(sample_lines) >= 120:
                    break
            if len(sample_lines) >= 120:
                break

        if not sample_lines:
            return "generic"
        heading_hits = sum(1 for line in sample_lines if self._is_heading_candidate(line))
        return "structured" if heading_hits >= 3 or (heading_hits / len(sample_lines)) >= 0.08 else "generic"

    @staticmethod
    def _make_base_doc(filename: str, file_path: str, doc_type: str, page_number: int) -> dict:
        return {
            "filename": filename,
            "file_path": file_path,
            "file_type": doc_type,
            "page_number": page_number,
        }

    @staticmethod
    def _make_chunk_id(filename: str, prefix: str, level: int, index: int) -> str:
        return f"{filename}::{prefix}::l{level}::{index}"

    def _split_generic_page(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        if not text:
            return []

        chunks: List[dict] = []
        filename = base_doc["filename"]
        page_number = int(base_doc.get("page_number", 0))
        root_docs = self._root_splitter.create_documents([text], [base_doc])
        root_counter = 0
        leaf_counter = 0

        for root_doc in root_docs:
            root_text = (root_doc.page_content or "").strip()
            if not root_text:
                continue
            root_id = self._make_chunk_id(filename, f"p{page_number}", 1, root_counter)
            root_counter += 1
            chunks.append(
                {
                    **base_doc,
                    "text": root_text,
                    "retrieval_text": root_text,
                    "chunk_id": root_id,
                    "parent_chunk_id": "",
                    "root_chunk_id": root_id,
                    "chunk_level": 1,
                    "chunk_role": "root",
                    "chunk_idx": page_global_chunk_idx,
                    "section_title": "",
                    "section_type": "generic",
                    "section_path": "",
                    "anchor_id": "",
                    "page_start": page_number,
                    "page_end": page_number,
                }
            )
            page_global_chunk_idx += 1

            leaf_docs = self._leaf_splitter.create_documents([root_text], [base_doc])
            for leaf_doc in leaf_docs:
                leaf_text = (leaf_doc.page_content or "").strip()
                if not leaf_text:
                    continue
                leaf_id = self._make_chunk_id(filename, f"p{page_number}", 3, leaf_counter)
                leaf_counter += 1
                chunks.append(
                    {
                        **base_doc,
                        "text": leaf_text,
                        "retrieval_text": leaf_text,
                        "chunk_id": leaf_id,
                        "parent_chunk_id": root_id,
                        "root_chunk_id": root_id,
                        "chunk_level": 3,
                        "chunk_role": "leaf",
                        "chunk_idx": page_global_chunk_idx,
                        "section_title": "",
                        "section_type": "generic",
                        "section_path": "",
                        "anchor_id": "",
                        "page_start": page_number,
                        "page_end": page_number,
                    }
                )
                page_global_chunk_idx += 1

        return chunks

    def _flush_section(
        self,
        sections: list[dict],
        filename: str,
        section_counter: int,
        page_global_chunk_idx: int,
    ) -> list[dict]:
        docs: list[dict] = []
        for section in sections:
            body = "\n".join(section["lines"]).strip()
            if not body:
                continue
            root_id = self._make_chunk_id(filename, f"s{section['section_index']}", 1, 0)
            root_text = "\n".join(item for item in [section["title"], body] if item).strip()
            root_doc = {
                "filename": filename,
                "file_path": section["file_path"],
                "file_type": section["file_type"],
                "page_number": section["page_start"],
                "text": root_text,
                "retrieval_text": self._make_retrieval_text(
                    raw_text=root_text,
                    body=body,
                    current_title=section["title"],
                    parent_title=section["parent_title"],
                ),
                "chunk_id": root_id,
                "parent_chunk_id": "",
                "root_chunk_id": root_id,
                "chunk_level": 1,
                "chunk_role": "root",
                "chunk_idx": page_global_chunk_idx,
                "section_title": section["title"],
                "section_type": section["section_type"],
                "section_path": section["section_path"],
                "anchor_id": section["anchor_id"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            }
            docs.append(root_doc)
            page_global_chunk_idx += 1

            leaf_docs = self._leaf_splitter.create_documents([body], [root_doc])
            for leaf_index, leaf_doc in enumerate(leaf_docs):
                leaf_text = (leaf_doc.page_content or "").strip()
                if not leaf_text:
                    continue
                docs.append(
                    {
                        **root_doc,
                        "text": leaf_text,
                        "retrieval_text": self._make_retrieval_text(
                            raw_text=leaf_text,
                            body=leaf_text,
                            current_title=section["title"],
                            parent_title=section["parent_title"],
                        ),
                        "chunk_id": self._make_chunk_id(filename, f"s{section['section_index']}", 3, leaf_index),
                        "parent_chunk_id": root_id,
                        "chunk_level": 3,
                        "chunk_role": "leaf",
                        "chunk_idx": page_global_chunk_idx,
                    }
                )
                page_global_chunk_idx += 1
        return docs

    def _build_structured_chunks(
        self,
        raw_docs: list,
        filename: str,
        file_path: str,
        doc_type: str,
    ) -> list[dict]:
        sections: list[dict] = []
        current: dict | None = None
        heading_stack: dict[int, str] = {}

        for doc in raw_docs:
            page_number = int(doc.metadata.get("page", 0) or 0)
            for raw_line in (doc.page_content or "").splitlines():
                line = self._normalize_title(raw_line)
                if not line:
                    continue
                depth = self._heading_depth(line)
                if depth:
                    if current and current["lines"]:
                        sections.append(current)
                    heading_stack = {k: v for k, v in heading_stack.items() if k < depth}
                    parent_title = heading_stack.get(depth - 1, "")
                    heading_stack[depth] = line
                    section_path = " > ".join(heading_stack[i] for i in sorted(heading_stack))
                    current = {
                        "title": line,
                        "parent_title": parent_title,
                        "section_path": section_path,
                        "section_type": "structured",
                        "anchor_id": self._extract_anchor_id(line),
                        "page_start": page_number,
                        "page_end": page_number,
                        "file_path": file_path,
                        "file_type": doc_type,
                        "section_index": len(sections),
                        "lines": [],
                    }
                    continue

                if current is None:
                    current = {
                        "title": "",
                        "parent_title": "",
                        "section_path": "",
                        "section_type": "structured",
                        "anchor_id": "",
                        "page_start": page_number,
                        "page_end": page_number,
                        "file_path": file_path,
                        "file_type": doc_type,
                        "section_index": len(sections),
                        "lines": [],
                    }
                current["page_end"] = page_number
                current["lines"].append(line)

        if current and current["lines"]:
            sections.append(current)

        docs: list[dict] = []
        page_global_chunk_idx = 0
        for idx, section in enumerate(sections):
            section["section_index"] = idx
            emitted = self._flush_section([section], filename, idx, page_global_chunk_idx)
            page_global_chunk_idx += len(emitted)
            docs.extend(emitted)
        return docs

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """Load a single document and split it into chunks."""
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        try:
            raw_docs = loader.load()
            if self._detect_profile(raw_docs) == "structured":
                structured_docs = self._build_structured_chunks(raw_docs, filename, file_path, doc_type)
                if structured_docs:
                    return structured_docs

            documents = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                base_doc = self._make_base_doc(
                    filename=filename,
                    file_path=file_path,
                    doc_type=doc_type,
                    page_number=int(doc.metadata.get("page", 0) or 0),
                )
                page_chunks = self._split_generic_page(
                    text=(doc.page_content or "").strip(),
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
            return documents
        except Exception as exc:
            raise Exception(f"Failed to process document: {str(exc)}") from exc

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """Load all supported documents from a folder."""
        all_documents = []
        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (
                file_lower.endswith(".pdf")
                or file_lower.endswith((".docx", ".doc"))
                or file_lower.endswith((".xlsx", ".xls"))
            ):
                continue
            file_path = os.path.join(folder_path, filename)
            try:
                all_documents.extend(self.load_document(file_path, filename))
            except Exception:
                continue
        return all_documents
