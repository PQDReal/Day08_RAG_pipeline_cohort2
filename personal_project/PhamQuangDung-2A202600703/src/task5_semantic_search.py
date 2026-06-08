"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
import math
from pathlib import Path

try:
    from src.task4_chunking_indexing import (
        INDEX_FILE,
        embed_texts,
        run_pipeline,
    )
except ModuleNotFoundError:
    from task4_chunking_indexing import (
        INDEX_FILE,
        embed_texts,
        run_pipeline,
    )


def _load_index() -> list[dict]:
    """Load the local JSONL vector index created in Task 4."""
    if not INDEX_FILE.exists():
        run_pipeline()

    chunks = []
    with Path(INDEX_FILE).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = embed_texts([query])[0]
    results = []
    for chunk in _load_index():
        score = _cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append({
            "content": chunk["content"],
            "score": round(score, 6),
            "metadata": chunk.get("metadata", {}),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
