"""
Task 7 — Reranking Module

Chosen method:
    MMR — Maximal Marginal Relevance

Goal:
    Re-score and re-order retrieved candidates based on:
    1. Relevance to the query
    2. Diversity among selected chunks

Why MMR:
    - Runs locally
    - No API key required
    - Reduces duplicated chunks from the same article/document
    - Useful after semantic search and BM25 return overlapping results

Required function:
    def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]
"""

import copy
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = Path("data/vector_store")
MANIFEST_PATH = VECTOR_STORE_DIR / "index_manifest.json"

DEFAULT_TOP_K = 5
MMR_LAMBDA = 0.75
# MMR_LAMBDA gần 1.0 = ưu tiên relevance.
# MMR_LAMBDA thấp hơn = ưu tiên diversity nhiều hơn.
# 0.75 là cân bằng hợp lý: vẫn ưu tiên đúng query, nhưng giảm trùng lặp.


_MODEL_CACHE: SentenceTransformer | None = None
_MODEL_NAME_CACHE: str | None = None


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest: {MANIFEST_PATH}. Hãy chạy Task 4 trước."
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model_name() -> str:
    """
    Dùng đúng embedding model đã chọn ở Task 4.
    """
    manifest = load_manifest()
    return manifest["embedding"]["model"]


def load_embedding_model() -> SentenceTransformer:
    """
    Load SentenceTransformer model used for reranking.
    """
    global _MODEL_CACHE, _MODEL_NAME_CACHE

    model_name = get_embedding_model_name()

    if _MODEL_CACHE is not None and _MODEL_NAME_CACHE == model_name:
        return _MODEL_CACHE

    _MODEL_CACHE = SentenceTransformer(model_name)
    _MODEL_NAME_CACHE = model_name

    return _MODEL_CACHE


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to range [0, 1].
    If all scores are equal, return 0.5 for all items.
    """
    scores = scores.astype("float32")

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score == min_score:
        return np.ones_like(scores, dtype="float32") * 0.5

    return (scores - min_score) / (max_score - min_score)


def format_candidate_text(candidate: dict) -> str:
    """
    Combine metadata and content for better reranking.

    Adding title/source_file helps with entity queries such as:
    - Nguyễn Công Trí
    - Chi Dân
    - Miu Lê
    - Điều 249
    """
    metadata = candidate.get("metadata", {})

    title = metadata.get("title", "")
    source_file = metadata.get("source_file", "")
    content = candidate.get("content", "")

    return f"{title}\n{source_file}\n{content}"

def is_boilerplate_chunk(candidate: dict) -> bool:
    """
    Detect low-value web boilerplate chunks such as navigation,
    sharing buttons, javascript links, footer, or menu text.
    These chunks can match article title but do not contain useful evidence.
    """
    content = candidate.get("content", "").strip().lower()

    if len(content) < 80:
        return True

    boilerplate_signals = [
        "javascript:;",
        "chia sẻ bài viết",
        "trở lại pháp luật",
        "xem thêm",
        "in\")",
        "facebook",
        "twitter",
        "email",
        "đăng nhập",
        "menu",
    ]

    signal_count = sum(1 for signal in boilerplate_signals if signal in content)

    # If too many boilerplate signals appear and the chunk is short,
    # it is likely not useful for answering.
    if signal_count >= 2 and len(content) < 500:
        return True

    # Too many Markdown links/images often means navigation/header/footer.
    link_count = content.count("](")
    image_count = content.count("![")

    if link_count >= 8 and len(content) < 800:
        return True

    if image_count >= 3 and len(content) < 800:
        return True

    return False


def filter_boilerplate_candidates(candidates: list[dict]) -> list[dict]:
    """
    Remove low-value boilerplate chunks while keeping fallback behavior:
    if filtering removes everything, return original candidates.
    """
    filtered = [c for c in candidates if not is_boilerplate_chunk(c)]
    return filtered if filtered else candidates

def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """
    Remove duplicated chunks based on chunk_id.
    If chunk_id does not exist, use content hash fallback.
    """
    seen = set()
    unique_candidates = []

    for candidate in candidates:
        metadata = candidate.get("metadata", {})
        chunk_id = metadata.get("chunk_id")

        if not chunk_id:
            chunk_id = hash(candidate.get("content", ""))

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        unique_candidates.append(candidate)

    return unique_candidates


def mmr_select(
    candidate_embeddings: np.ndarray,
    relevance_scores: np.ndarray,
    top_k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> list[tuple[int, float]]:
    """
    Select candidate indices using Maximal Marginal Relevance.

    MMR formula:
        score = lambda * relevance_to_query
                - (1 - lambda) * max_similarity_to_selected

    Returns:
        List of tuples:
        [
            (candidate_index, mmr_selection_score)
        ]
    """
    num_candidates = len(relevance_scores)

    if num_candidates == 0:
        return []

    top_k = min(top_k, num_candidates)

    selected: list[int] = []
    remaining = set(range(num_candidates))
    selected_with_scores: list[tuple[int, float]] = []

    # First item: most relevant to query.
    first_index = int(np.argmax(relevance_scores))
    selected.append(first_index)
    remaining.remove(first_index)
    selected_with_scores.append((first_index, float(relevance_scores[first_index])))

    while len(selected) < top_k and remaining:
        best_index = None
        best_mmr_score = -float("inf")

        selected_embeddings = candidate_embeddings[selected]

        for idx in remaining:
            relevance = relevance_scores[idx]

            # Similarity between this candidate and already selected candidates.
            similarity_to_selected = candidate_embeddings[idx] @ selected_embeddings.T
            max_similarity = float(np.max(similarity_to_selected))

            mmr_score = (
                lambda_mult * float(relevance)
                - (1.0 - lambda_mult) * max_similarity
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_index = idx

        selected.append(best_index)
        remaining.remove(best_index)
        selected_with_scores.append((best_index, float(best_mmr_score)))

    return selected_with_scores


def tokenize_for_fallback(text: str) -> set[str]:
    """
    Lightweight Unicode-aware tokenizer for offline fallback reranking.
    """
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def fallback_rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
    error: Exception,
) -> list[dict]:
    """
    Rerank without loading an embedding model.

    This keeps tests and demos usable on machines that cannot access the model
    registry. The normal MMR path still runs whenever SentenceTransformer works.
    """
    query_tokens = tokenize_for_fallback(query)
    original_scores = np.array(
        [float(c.get("score", 0.0)) for c in candidates],
        dtype="float32",
    )
    original_scores_norm = normalize_scores(original_scores)

    scored_items = []

    for index, candidate in enumerate(candidates):
        candidate_text = format_candidate_text(candidate)
        candidate_tokens = tokenize_for_fallback(candidate_text)

        if query_tokens:
            token_overlap = len(query_tokens & candidate_tokens) / len(query_tokens)
        else:
            token_overlap = 0.0

        combined_score = float(0.70 * token_overlap + 0.30 * original_scores_norm[index])
        scored_items.append((index, combined_score, token_overlap))

    scored_items.sort(key=lambda item: item[1], reverse=True)

    results = []

    for rank, (candidate_index, combined_score, token_overlap) in enumerate(
        scored_items[:top_k],
        start=1,
    ):
        original_candidate = candidates[candidate_index]
        metadata = dict(original_candidate.get("metadata", {}))
        metadata.update(
            {
                "rerank_method": "fallback_token_overlap",
                "rerank_rank": rank,
                "original_score": float(original_candidate.get("score", 0.0)),
                "combined_relevance_score": combined_score,
                "token_overlap_score": float(token_overlap),
                "rerank_fallback_used": True,
                "rerank_fallback_reason": str(error),
            }
        )

        results.append(
            {
                "content": original_candidate.get("content", ""),
                "score": combined_score,
                "metadata": metadata,
            }
        )

    return results


def rerank(query: str, candidates: list[dict], top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Re-score and re-order candidates based on relevance to query.

    Args:
        query:
            User query string.
        candidates:
            List of candidate chunks from retrieval modules.
            Each candidate should have:
                - content: str
                - score: float
                - metadata: dict
        top_k:
            Number of reranked results to return.

    Returns:
        List of dictionaries:
        [
            {
                "content": str,
                "score": float,
                "metadata": dict
            }
        ]
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng.")

    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0.")

    if not candidates:
        return []

    filtered_candidates = filter_boilerplate_candidates(candidates)
    unique_candidates = deduplicate_candidates(filtered_candidates)

    for candidate in unique_candidates:
        if "content" not in candidate:
            raise ValueError("Mỗi candidate phải có trường 'content'.")
        if "metadata" not in candidate:
            candidate["metadata"] = {}
        if "score" not in candidate:
            candidate["score"] = 0.0

    top_k = min(top_k, len(unique_candidates))

    try:
        model = load_embedding_model()

        candidate_texts = [format_candidate_text(c) for c in unique_candidates]

        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        candidate_embeddings = model.encode(
            candidate_texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
    except Exception as error:
        return fallback_rerank(
            query=query,
            candidates=unique_candidates,
            top_k=top_k,
            error=error,
        )

    # Semantic relevance between query and candidates.
    semantic_relevance = candidate_embeddings @ query_embedding
    semantic_relevance_norm = normalize_scores(semantic_relevance)

    # Original retrieval scores can come from semantic search or BM25.
    # Because their scales are different, we normalize them before combining.
    original_scores = np.array(
        [float(c.get("score", 0.0)) for c in unique_candidates],
        dtype="float32",
    )
    original_scores_norm = normalize_scores(original_scores)

    # Combined relevance:
    # 80% semantic relevance from query-candidate embedding similarity
    # 20% normalized original retrieval score
    combined_relevance = (
        0.80 * semantic_relevance_norm
        + 0.20 * original_scores_norm
    ).astype("float32")

    selected = mmr_select(
        candidate_embeddings=candidate_embeddings,
        relevance_scores=combined_relevance,
        top_k=top_k,
        lambda_mult=MMR_LAMBDA,
    )

    reranked_results = []

    for rank, (candidate_index, mmr_selection_score) in enumerate(selected, start=1):
        original_candidate = unique_candidates[candidate_index]
        item = copy.deepcopy(original_candidate)

        original_score = float(original_candidate.get("score", 0.0))
        combined_score = float(combined_relevance[candidate_index])
        semantic_score = float(semantic_relevance[candidate_index])

        # Final score is rank-aware to keep output sorted by reranked order.
        # This is not a probability. It is a rerank score for ordering.
        final_score = float((1.0 / rank) + 0.01 * combined_score)

        metadata = dict(item.get("metadata", {}))
        metadata.update(
            {
                "rerank_method": "MMR",
                "rerank_rank": rank,
                "original_score": original_score,
                "semantic_relevance_score": semantic_score,
                "combined_relevance_score": combined_score,
                "mmr_selection_score": float(mmr_selection_score),
                "mmr_lambda": MMR_LAMBDA,
            }
        )

        reranked_results.append(
            {
                "content": item["content"],
                "score": final_score,
                "metadata": metadata,
            }
        )

    # Ensure output is sorted descending by score.
    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["score"],
        reverse=True,
    )

    return reranked_results[:top_k]


def print_rerank_results(results: list[dict]) -> None:
    for rank, item in enumerate(results, start=1):
        metadata = item["metadata"]
        content_preview = item["content"].replace("\n", " ")[:300]

        print(f"\n--- Reranked Result {rank} ---")
        print(f"Final score: {item['score']:.4f}")
        print(f"Original score: {metadata.get('original_score')}")
        print(f"Combined relevance: {metadata.get('combined_relevance_score')}")
        print(f"MMR score: {metadata.get('mmr_selection_score')}")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Source type: {metadata.get('source_type', '')}")
        print(f"Source file: {metadata.get('source_file', '')}")
        print(f"Chunk ID: {metadata.get('chunk_id', '')}")
        print(f"Content preview: {content_preview}...")


def demo() -> None:
    """
    Demo reranking by combining candidates from:
    - Task 5 semantic_search
    - Task 6 lexical_search
    """
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search

    test_queries = [
        "Nguyễn Công Trí bị bắt vì liên quan đến ma túy như thế nào?",
        "Tổ chức sử dụng trái phép chất ma túy bị xử lý ra sao?",
        "Các tội phạm về ma túy trong Bộ luật Hình sự gồm những tội nào?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        semantic_candidates = semantic_search(query, top_k=10)
        lexical_candidates = lexical_search(query, top_k=10)

        candidates = semantic_candidates + lexical_candidates

        reranked = rerank(query=query, candidates=candidates, top_k=5)

        print_rerank_results(reranked)


if __name__ == "__main__":
    demo()
