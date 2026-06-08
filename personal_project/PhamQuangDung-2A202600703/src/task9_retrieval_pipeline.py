"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import re

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _query_intent(query: str) -> str:
    legal_terms = {
        "luật", "nghị", "định", "pháp", "hình", "phạt", "điều",
        "cai", "nghiện", "danh", "mục", "tiền", "chất", "trách", "nhiệm",
    }
    news_terms = {"nghệ", "sĩ", "diễn", "viên", "người", "mẫu", "bị", "bắt", "g-dragon", "hữu", "tín"}
    tokens = set(query.lower().replace("-", " ").split())
    if tokens & legal_terms:
        return "legal"
    if tokens & news_terms:
        return "news"
    return "mixed"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _result_type(result: dict) -> str:
    metadata = result.get("metadata", {})
    source = str(metadata.get("source", "")).lower()
    doc_type = str(metadata.get("type", "")).lower()
    if doc_type == "legal" or source.startswith(("luat-", "nghi-dinh")):
        return "legal"
    if doc_type == "news" or source.startswith("article_"):
        return "news"
    return "unknown"


def _dedupe_results(results: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for result in results:
        metadata = result.get("metadata", {})
        key = (
            metadata.get("source"),
            metadata.get("chunk_index"),
            result.get("content", "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _apply_intent_filter(query: str, results: list[dict], top_k: int) -> list[dict]:
    intent = _query_intent(query)
    if intent not in {"legal", "news"}:
        return results

    preferred = [result for result in results if _result_type(result) == intent]
    query_lower = query.lower()
    topic_terms: set[str] = set()
    if "cai" in query_lower or "nghiện" in query_lower:
        topic_terms = {"cai", "nghiện"}
    elif "hình phạt" in query_lower or "tàng trữ" in query_lower:
        topic_terms = {"phạt", "tù", "tội", "tàng", "trữ"}
    elif "danh mục" in query_lower or "tiền chất" in query_lower:
        topic_terms = {"danh", "mục", "tiền", "chất"}

    if topic_terms:
        topic_matched = [
            result for result in preferred
            if topic_terms & _tokens(result.get("content", ""))
        ]
        if len(topic_matched) >= min(top_k, 3):
            return topic_matched

    if len(preferred) >= max(1, min(top_k, 3)):
        remainder = [result for result in results if result not in preferred]
        return preferred + remainder
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    search_k = max(top_k * 3, top_k)

    dense_results = semantic_search(query, top_k=search_k)
    sparse_results = lexical_search(query, top_k=search_k)

    merged = rerank_rrf([dense_results, sparse_results], top_k=search_k)
    for item in merged:
        metadata = item.get("metadata", {}).copy()
        metadata["retrieval_stage"] = "rrf_hybrid"
        item["metadata"] = metadata
        item["source"] = "hybrid"

    merged = _dedupe_results(_apply_intent_filter(query, merged, top_k))

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    final_results = _dedupe_results(_apply_intent_filter(query, final_results, top_k))[:top_k]

    for item in final_results:
        item["source"] = "hybrid"

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) < threshold "
            f"({score_threshold:.3f}). Fallback → PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        for item in fallback:
            item["source"] = "pageindex"
        return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
