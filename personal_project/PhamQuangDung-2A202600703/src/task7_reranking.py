"""
Task 7 — Reranking Module.

Implemented methods:
    - cross_encoder: local cross-encoder-style heuristic reranker
    - MMR: Maximal Marginal Relevance for relevance + diversity
    - RRF: Reciprocal Rank Fusion for merging ranked lists

The default cross_encoder method does not need an API key. It combines query
token recall, phrase/bigram overlap, and the original retrieval score.
"""

import math

try:
    from src.task4_chunking_indexing import embed_texts
    from src.task6_lexical_search import tokenize
except ModuleNotFoundError:
    from task4_chunking_indexing import embed_texts
    from task6_lexical_search import tokenize


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in values]
    return [(value - min_score) / (max_score - min_score) for value in values]


def _heuristic_relevance(query: str, content: str) -> float:
    """Cross-encoder-style relevance signal for Vietnamese text."""
    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    query_set = set(query_tokens)
    content_set = set(content_tokens)
    token_recall = len(query_set & content_set) / len(query_set)

    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    content_bigrams = set(zip(content_tokens, content_tokens[1:]))
    bigram_recall = (
        len(query_bigrams & content_bigrams) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )

    phrase_bonus = 1.0 if query.lower() in content.lower() else 0.0
    return 0.65 * token_recall + 0.25 * bigram_recall + 0.10 * phrase_bonus


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates with a local heuristic that mimics cross-encoder behavior.

    A real cross-encoder scores a (query, document) pair jointly. This local
    version approximates that by checking query/document token and phrase
    interaction, then blends it with the original retriever score.
    """
    if top_k <= 0:
        return []

    original_scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    normalized_scores = _normalize_scores(original_scores)

    reranked = []
    for candidate, normalized_score in zip(candidates, normalized_scores):
        relevance_score = _heuristic_relevance(query, candidate.get("content", ""))
        final_score = 0.75 * relevance_score + 0.25 * normalized_score

        item = candidate.copy()
        item["score"] = round(final_score, 6)
        metadata = item.get("metadata", {}).copy()
        metadata["rerank_method"] = "heuristic_cross_encoder"
        metadata["original_score"] = candidate.get("score", 0.0)
        item["metadata"] = metadata
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    MMR = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected_docs))
    """
    if top_k <= 0:
        return []

    missing_texts = [
        candidate.get("content", "")
        for candidate in candidates
        if "embedding" not in candidate
    ]
    generated_embeddings = iter(embed_texts(missing_texts)) if missing_texts else iter([])

    enriched = []
    for candidate in candidates:
        item = candidate.copy()
        if "embedding" not in item:
            item["embedding"] = next(generated_embeddings)
        enriched.append(item)

    selected: list[int] = []
    remaining = list(range(len(enriched)))

    for _ in range(min(top_k, len(enriched))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_similarity(query_embedding, enriched[idx]["embedding"])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine_similarity(enriched[idx]["embedding"], enriched[sel_idx]["embedding"])
                    for sel_idx in selected
                )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        enriched[best_idx]["score"] = round(best_score, 6)
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = enriched[idx].copy()
        item.pop("embedding", None)
        metadata = item.get("metadata", {}).copy()
        metadata["rerank_method"] = "mmr"
        item["metadata"] = metadata
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d)))
    """
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        metadata = item.get("metadata", {}).copy()
        metadata["rerank_method"] = "rrf"
        item["metadata"] = metadata
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" | "mmr" | "rrf"
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        query_embedding = embed_texts([query])[0]
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
