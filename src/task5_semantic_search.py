"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

try:
    from .retrieval_utils import cosine, hashed_embedding
    from .task4_chunking_indexing import (
        EMBEDDING_DIM,
        VECTOR_STORE,
        WEAVIATE_API_KEY,
        WEAVIATE_COLLECTION,
        WEAVIATE_URL,
        chunk_documents,
        load_documents,
    )
except ImportError:
    from retrieval_utils import cosine, hashed_embedding
    from task4_chunking_indexing import (
        EMBEDDING_DIM,
        VECTOR_STORE,
        WEAVIATE_API_KEY,
        WEAVIATE_COLLECTION,
        WEAVIATE_URL,
        chunk_documents,
        load_documents,
    )


def _semantic_search_local(query: str, top_k: int) -> list[dict]:
    query_embedding = hashed_embedding(query, EMBEDDING_DIM)
    chunks = chunk_documents(load_documents())
    results = []
    for chunk in chunks:
        score = cosine(query_embedding, hashed_embedding(chunk["content"], EMBEDDING_DIM))
        if score > 0:
            results.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _semantic_search_weaviate(query: str, top_k: int) -> list[dict]:
    import weaviate
    from weaviate.auth import Auth
    from weaviate.classes.query import MetadataQuery

    if not WEAVIATE_URL:
        return []

    client = None
    try:
        auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
        if auth:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=auth,
            )
        else:
            client = weaviate.connect_to_local()

        if not client.collections.exists(WEAVIATE_COLLECTION):
            return []

        query_embedding = hashed_embedding(query, EMBEDDING_DIM)
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            props = obj.properties
            distance = getattr(obj.metadata, "distance", None)
            score = 1.0 - float(distance if distance is not None else 1.0)
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": score,
                    "metadata": {
                        "source": props.get("source", ""),
                        "type": props.get("doc_type", ""),
                        "path": props.get("path", ""),
                        "chunk_index": props.get("chunk_index", 0),
                    },
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results
    finally:
        if client is not None:
            client.close()


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

    if VECTOR_STORE == "weaviate":
        try:
            results = _semantic_search_weaviate(query, top_k)
            if results:
                return results
        except Exception as exc:
            print(f"⚠ Weaviate semantic search lỗi, fallback local: {exc}")

    return _semantic_search_local(query, top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
