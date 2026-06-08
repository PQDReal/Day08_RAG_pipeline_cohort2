"""
Task 9 — Hybrid Retrieval Pipeline

Goal:
    Combine the retrieval components built in previous tasks:

    Task 5: semantic_search(query, top_k)
    Task 6: lexical_search(query, top_k)
    Task 7: rerank(query, candidates, top_k)
    Task 8: pageindex_search(query, top_k) as fallback

Required behavior:
    1. Run semantic search.
    2. Run lexical/BM25 search.
    3. Merge and deduplicate candidates.
    4. Rerank candidates.
    5. If retrieval quality is low or no result is found, fallback to PageIndex.
    6. Return list[dict] with:
        - content
        - score
        - metadata
"""

import hashlib
from typing import Any


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL_SIZE = 10
DEFAULT_SCORE_THRESHOLD = 0.35


# ============================================================
# ROBUST IMPORTS
# ============================================================
# These try/except imports allow the file to work in both cases:
# 1. Running directly:
#       python src/task9_hybrid_retrieval.py
# 2. Importing as module:
#       from src.task9_hybrid_retrieval import retrieve

try:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank
except ModuleNotFoundError:
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank


try:
    from task8_pageindex_vectorless import pageindex_search
except ModuleNotFoundError:
    try:
        from src.task8_pageindex_vectorless import pageindex_search
    except ModuleNotFoundError:
        pageindex_search = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_candidate_id(candidate: dict) -> str:
    """
    Create stable candidate ID for deduplication.

    Prefer chunk_id from metadata.
    If missing, fallback to hash of content.
    """
    metadata = candidate.get("metadata", {})
    chunk_id = metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    content = candidate.get("content", "")
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def attach_retrieval_source(candidate: dict, source: str) -> dict:
    """
    Add retrieval source metadata to a candidate.

    source can be:
        - semantic
        - lexical
        - pageindex
    """
    item = {
        "content": candidate.get("content", ""),
        "score": float(candidate.get("score", 0.0)),
        "metadata": dict(candidate.get("metadata", {})),
    }

    metadata = item["metadata"]
    metadata["retrieval_source"] = source
    metadata[f"{source}_score"] = float(candidate.get("score", 0.0))

    return item


def merge_candidates(
    semantic_results: list[dict],
    lexical_results: list[dict],
) -> list[dict]:
    """
    Merge semantic and lexical candidates.

    If the same chunk appears in both sources:
        - keep one candidate
        - store both semantic_score and lexical_score in metadata
        - combine source labels into retrieval_sources
    """
    merged: dict[str, dict] = {}

    for source, results in [
        ("semantic", semantic_results),
        ("lexical", lexical_results),
    ]:
        for candidate in results:
            candidate_id = get_candidate_id(candidate)
            item = attach_retrieval_source(candidate, source)

            if candidate_id not in merged:
                item["metadata"]["retrieval_sources"] = [source]
                item["metadata"]["hybrid_candidate_id"] = candidate_id
                merged[candidate_id] = item
            else:
                existing = merged[candidate_id]
                existing_metadata = existing["metadata"]

                retrieval_sources = set(existing_metadata.get("retrieval_sources", []))
                retrieval_sources.add(source)
                existing_metadata["retrieval_sources"] = sorted(retrieval_sources)

                existing_metadata[f"{source}_score"] = float(candidate.get("score", 0.0))

                # Keep the higher raw score only as a generic candidate score.
                # Rerank will normalize and re-score later.
                existing["score"] = max(
                    float(existing.get("score", 0.0)),
                    float(candidate.get("score", 0.0)),
                )

    return list(merged.values())


def get_best_combined_relevance(results: list[dict]) -> float:
    """
    Get best combined relevance score from reranked results.

    Task 7 adds combined_relevance_score in metadata.
    If it is missing, fallback to result score.
    """
    if not results:
        return 0.0

    scores = []

    for item in results:
        metadata = item.get("metadata", {})
        value = metadata.get("combined_relevance_score", item.get("score", 0.0))

        try:
            scores.append(float(value))
        except Exception:
            scores.append(0.0)

    return max(scores) if scores else 0.0


def should_use_pageindex_fallback(
    reranked_results: list[dict],
    score_threshold: float,
    force_pageindex: bool = False,
) -> bool:
    """
    Decide whether to fallback to PageIndex.

    PageIndex fallback is used when:
    - force_pageindex=True
    - no reranked results exist
    - best combined relevance is lower than threshold
    """
    if force_pageindex:
        return True

    if not reranked_results:
        return True

    best_score = get_best_combined_relevance(reranked_results)

    return best_score < score_threshold


def annotate_hybrid_results(
    results: list[dict],
    query: str,
    semantic_count: int,
    lexical_count: int,
    merged_count: int,
    pageindex_used: bool = False,
) -> list[dict]:
    """
    Add pipeline-level metadata to final results.
    """
    annotated = []
    result_source = "pageindex" if pageindex_used else "hybrid"

    for rank, item in enumerate(results, start=1):
        metadata = dict(item.get("metadata", {}))

        metadata.update(
            {
                "final_rank": rank,
                "query": query,
                "retrieval_pipeline": (
                    "pageindex_fallback"
                    if pageindex_used
                    else "hybrid_semantic_bm25_mmr"
                ),
                "semantic_candidates_count": semantic_count,
                "lexical_candidates_count": lexical_count,
                "merged_candidates_count": merged_count,
                "pageindex_used": pageindex_used,
            }
        )

        annotated.append(
            {
                "content": item.get("content", ""),
                "score": float(item.get("score", 0.0)),
                "source": result_source,
                "metadata": metadata,
            }
        )

    return annotated


# ============================================================
# TASK 9 REQUIRED FUNCTION
# ============================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    use_pageindex_fallback: bool = True,
    force_pageindex: bool = False,
) -> list[dict]:
    """
    Hybrid retrieval pipeline.

    Args:
        query:
            User query string.
        top_k:
            Number of final results to return.
        score_threshold:
            Minimum quality threshold before using PageIndex fallback.
        candidate_pool_size:
            Number of candidates retrieved from semantic and lexical search.
        use_pageindex_fallback:
            Whether to use PageIndex if local hybrid retrieval is weak.
        force_pageindex:
            Force PageIndex fallback for testing/demo.

    Returns:
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

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên lớn hơn 0.")

    if not isinstance(candidate_pool_size, int) or candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size phải là số nguyên lớn hơn 0.")

    # 1. Semantic search
    semantic_results = semantic_search(query=query, top_k=candidate_pool_size)

    # 2. Lexical / BM25 search
    lexical_results = lexical_search(query=query, top_k=candidate_pool_size)

    # 3. Merge candidates
    merged_candidates = merge_candidates(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
    )

    # 4. Rerank
    reranked_results = rerank(
        query=query,
        candidates=merged_candidates,
        top_k=top_k,
    )

    # 5. Optional PageIndex fallback
    fallback_needed = should_use_pageindex_fallback(
        reranked_results=reranked_results,
        score_threshold=score_threshold,
        force_pageindex=force_pageindex,
    )

    if fallback_needed and use_pageindex_fallback:
        if pageindex_search is not None:
            try:
                pageindex_results = pageindex_search(query=query, top_k=top_k)

                if pageindex_results:
                    return annotate_hybrid_results(
                        results=pageindex_results,
                        query=query,
                        semantic_count=len(semantic_results),
                        lexical_count=len(lexical_results),
                        merged_count=len(merged_candidates),
                        pageindex_used=True,
                    )

            except Exception as error:
                print(f"PageIndex fallback failed, using local reranked results. Error: {error}")

    # 6. Return local hybrid results
    return annotate_hybrid_results(
        results=reranked_results,
        query=query,
        semantic_count=len(semantic_results),
        lexical_count=len(lexical_results),
        merged_count=len(merged_candidates),
        pageindex_used=False,
    )


# ============================================================
# DEMO / CLI
# ============================================================

def print_retrieve_results(results: list[dict]) -> None:
    for rank, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        preview = item.get("content", "").replace("\n", " ")[:500]

        print(f"\n--- Hybrid Retrieval Result {rank} ---")
        print(f"Score: {item.get('score')}")
        print(f"Pipeline: {metadata.get('retrieval_pipeline')}")
        print(f"PageIndex used: {metadata.get('pageindex_used')}")
        print(f"Title: {metadata.get('title', metadata.get('source', ''))}")
        print(f"Source type: {metadata.get('source_type', '')}")
        print(f"Source file: {metadata.get('source_file', '')}")
        print(f"Retrieval sources: {metadata.get('retrieval_sources')}")
        print(f"Rerank method: {metadata.get('rerank_method')}")
        print(f"Chunk ID: {metadata.get('chunk_id', '')}")
        print(f"Preview: {preview}...")


def demo() -> None:
    test_queries = [
        "Nguyễn Công Trí bị bắt vì liên quan đến ma túy như thế nào?",
        "Tổ chức sử dụng trái phép chất ma túy bị xử lý ra sao?",
        "Các tội phạm về ma túy trong Bộ luật Hình sự gồm những tội nào?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = retrieve(query=query, top_k=5)
        print_retrieve_results(results)


def interactive_search() -> None:
    print("\n=== Task 9: Hybrid Retrieval Pipeline ===")
    print("Pipeline: semantic_search + lexical_search + rerank + PageIndex fallback")
    print("Gõ 'q', 'quit' hoặc 'exit' để thoát.\n")

    while True:
        query = input("Nhập query: ").strip()

        if query.lower() in {"q", "quit", "exit"}:
            print("Thoát hybrid retrieval.")
            break

        if not query:
            print("Query không được rỗng.")
            continue

        top_k_input = input("Nhập top_k, mặc định 5: ").strip()

        try:
            top_k = int(top_k_input) if top_k_input else DEFAULT_TOP_K
        except ValueError:
            print("top_k không hợp lệ. Dùng mặc định top_k=5.")
            top_k = DEFAULT_TOP_K

        force_input = input("Force PageIndex fallback? y/N: ").strip().lower()
        force_pageindex = force_input in {"y", "yes"}

        results = retrieve(
            query=query,
            top_k=top_k,
            force_pageindex=force_pageindex,
        )

        print_retrieve_results(results)

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    # Đổi demo() thành interactive_search() nếu muốn tự nhập.
    demo()
