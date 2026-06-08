"""Shared local retrieval helpers for the Day 8 RAG tasks.

The course brief recommends production services such as Weaviate, PageIndex and
cross-encoder rerankers. For the individual automated tests we keep a local,
deterministic implementation so the pipeline can run without Docker or API
setup, while preserving the same public task interfaces.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
STANDARDIZED_DIR = DATA_DIR / "standardized"
LANDING_DIR = DATA_DIR / "landing"

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
DEFAULT_DIM = 384


def tokenize(text: str) -> list[str]:
    """Simple Unicode tokenization that works acceptably for Vietnamese text."""
    return TOKEN_RE.findall(text.lower())


def infer_doc_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return path.parent.name or "unknown"


def read_json_article(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = data.get("title") or path.stem
    url = data.get("url", "")
    crawled = data.get("date_crawled") or data.get("crawled_at") or ""
    content = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or data.get("text")
        or ""
    )
    header = f"# {title}\n\nSource: {url}\nCrawled: {crawled}\n\n"
    return header + str(content)


def load_raw_documents() -> list[dict]:
    """Load standardized markdown first, with a JSON-news fallback."""
    documents: list[dict] = []

    if STANDARDIZED_DIR.exists():
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            if md_file.name.startswith("."):
                continue
            content = md_file.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "source": md_file.name,
                        "path": str(md_file.relative_to(PROJECT_DIR)),
                        "type": infer_doc_type(md_file),
                    },
                }
            )

    news_dir = LANDING_DIR / "news"
    if news_dir.exists():
        existing_sources = {doc["metadata"]["source"] for doc in documents}
        for json_file in sorted(news_dir.glob("*.json")):
            md_name = f"{json_file.stem}.md"
            if md_name in existing_sources:
                continue
            try:
                content = read_json_article(json_file).strip()
            except (OSError, json.JSONDecodeError):
                continue
            if content:
                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": json_file.name,
                            "path": str(json_file.relative_to(PROJECT_DIR)),
                            "type": "news",
                        },
                    }
                )

    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk by paragraphs while enforcing a hard character budget."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
        if split_at > int(chunk_size * 0.45) and end < len(text):
            end = start + split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def make_chunks(documents: Iterable[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for doc in documents:
        for i, content in enumerate(chunk_text(doc.get("content", ""), chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "content": content,
                    "metadata": {**doc.get("metadata", {}), "chunk_index": i},
                }
            )
    return chunks


def hashed_embedding(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Lightweight signed hashing vector for dependency-free semantic scoring."""
    vector = [0.0] * dim
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def all_chunks(chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    return make_chunks(load_raw_documents(), chunk_size, chunk_overlap)


def normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return []
    max_score = max(float(r.get("score", 0.0)) for r in results) or 1.0
    normalized = []
    for result in results:
        item = result.copy()
        item["score"] = float(item.get("score", 0.0)) / max_score
        normalized.append(item)
    return normalized
