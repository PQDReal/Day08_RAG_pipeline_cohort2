"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

try:
    from .retrieval_utils import cosine, hashed_embedding, tokenize
except ImportError:
    from retrieval_utils import cosine, hashed_embedding, tokenize

JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
USE_JINA_RERANKER = os.getenv("USE_JINA_RERANKER", "true").lower() == "true"


def rerank_local(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Dependency-free reranker used when the Jina API is unavailable."""
    query_tokens = set(tokenize(query))
    query_embedding = hashed_embedding(query)
    scored = []
    for candidate in candidates:
        content = candidate.get("content", "")
        content_tokens = set(tokenize(content))
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        semantic = cosine(query_embedding, hashed_embedding(content))
        original = float(candidate.get("score", 0.0))
        score = 0.45 * original + 0.35 * max(semantic, 0.0) + 0.20 * overlap
        item = candidate.copy()
        item["original_score"] = original
        item["score"] = float(score)
        item["combined_relevance"] = float(score)
        item["reranker"] = "local"
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_jina_api(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank bằng Jina AI API.

    API nhận query + list documents, trả về index và relevance_score. Ta map lại
    về candidate gốc để giữ metadata/source cho pipeline citation.
    """
    if not JINA_API_KEY:
        raise RuntimeError("Thiếu JINA_API_KEY trong .env")
    if not candidates:
        return []

    response = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c.get("content", "") for c in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    results = []
    for result in payload.get("results", []):
        idx = int(result["index"])
        item = candidates[idx].copy()
        item["original_score"] = float(item.get("score", 0.0))
        item["score"] = float(result.get("relevance_score", 0.0))
        item["combined_relevance"] = item["score"]
        item["reranker"] = "jina"
        results.append(item)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if USE_JINA_RERANKER:
        try:
            return rerank_jina_api(query, candidates, top_k)
        except Exception as exc:
            print(f"⚠ Jina reranker lỗi, fallback local: {exc}")
    return rerank_local(query, candidates, top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []
    embeddings = [
        c.get("embedding") or hashed_embedding(c.get("content", ""))
        for c in candidates
    ]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            relevance = cosine(query_embedding, embeddings[idx])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(cosine(embeddings[idx], embeddings[sel]) for sel in selected)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(cosine(query_embedding, embeddings[idx]))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


def _metadata_value(metadata: dict, *keys: str, default: str = "N/A") -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _preview(text: str, max_chars: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _demo_candidates(query: str, top_k: int = 8) -> list[dict]:
    """Build realistic candidates from Task 5 + 6 when running this file."""
    try:
        from .task5_semantic_search import semantic_search
        from .task6_lexical_search import lexical_search
    except ImportError:
        from task5_semantic_search import semantic_search
        from task6_lexical_search import lexical_search

    dense = semantic_search(query, top_k=top_k)
    lexical = lexical_search(query, top_k=top_k)
    return rerank_rrf([dense, lexical], top_k=top_k)


def print_reranked_results(query: str, results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        source_file = _metadata_value(metadata, "source", "path")
        title = _metadata_value(metadata, "title", default=source_file)
        source_type = _metadata_value(metadata, "type", "doc_type", default="unknown")
        chunk_id = _metadata_value(
            metadata,
            "chunk_id",
            default=f"{source_file}::chunk_{metadata.get('chunk_index', 0)}",
        )

        print(f"\n--- Reranked Result {i} ---")
        print(f"Final score: {float(result.get('score', 0.0)):.4f}")
        print(f"Original score: {float(result.get('original_score', result.get('score', 0.0))):.4f}")
        print(f"Combined relevance: {float(result.get('combined_relevance', result.get('score', 0.0))):.4f}")
        print(f"Reranker: {result.get('reranker', 'rrf/local')}")
        print(f"Title: {title}")
        print(f"Source type: {source_type}")
        print(f"Source file: {source_file}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Content preview: {_preview(result.get('content', ''))}")


if __name__ == "__main__":
    query = os.getenv(
        "TASK7_DEMO_QUERY",
        "Chi Dân và An Tây liên quan đến vụ án ma túy như thế nào?",
    )
    candidates = _demo_candidates(query, top_k=8)
    results = rerank(query, candidates, top_k=3)
    print_reranked_results(query, results)
