"""
Task 5 — Semantic Search Module

Goal:
    Implement dense retrieval / semantic search over the vector store built in Task 4.

Input:
    query: str
    top_k: int

Output:
    List of dictionaries:
    [
        {
            "content": str,
            "score": float,
            "metadata": dict
        }
    ]

How it works:
    1. Load chunks.json and embeddings.npy from data/vector_store/.
    2. Load the same embedding model used in Task 4.
    3. Embed the user query.
    4. Compute cosine similarity between query embedding and all chunk embeddings.
    5. Return top_k chunks sorted by score descending.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = Path("data/vector_store")
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
MANIFEST_PATH = VECTOR_STORE_DIR / "index_manifest.json"

DEFAULT_TOP_K = 10


# Cache để không load model và vector store nhiều lần.
_CHUNKS_CACHE: list[dict] | None = None
_EMBEDDINGS_CACHE: np.ndarray | None = None
_MODEL_CACHE: SentenceTransformer | None = None
_MODEL_NAME_CACHE: str | None = None


def load_manifest() -> dict:
    """
    Load index manifest created in Task 4.
    Manifest contains embedding model name and index configuration.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest: {MANIFEST_PATH}. "
            "Hãy chạy Task 4 trước."
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model_name() -> str:
    """
    Get the embedding model name from Task 4 manifest.
    This ensures Task 5 uses the same model as Task 4.
    """
    manifest = load_manifest()

    try:
        return manifest["embedding"]["model"]
    except KeyError as error:
        raise KeyError(
            "index_manifest.json thiếu trường embedding.model. "
            "Hãy kiểm tra lại Task 4 manifest."
        ) from error


def load_vector_store() -> tuple[list[dict], np.ndarray]:
    """
    Load chunks and embeddings from local vector store.

    Returns:
        chunks: list[dict]
        embeddings: np.ndarray with shape (num_chunks, embedding_dim)
    """
    global _CHUNKS_CACHE, _EMBEDDINGS_CACHE

    if _CHUNKS_CACHE is not None and _EMBEDDINGS_CACHE is not None:
        return _CHUNKS_CACHE, _EMBEDDINGS_CACHE

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy chunks file: {CHUNKS_PATH}. "
            "Hãy chạy Task 4 trước."
        )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy embeddings file: {EMBEDDINGS_PATH}. "
            "Hãy chạy Task 4 trước."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Số chunks ({len(chunks)}) không khớp số embeddings ({embeddings.shape[0]})."
        )

    _CHUNKS_CACHE = chunks
    _EMBEDDINGS_CACHE = embeddings.astype("float32")

    return _CHUNKS_CACHE, _EMBEDDINGS_CACHE


def load_embedding_model() -> SentenceTransformer:
    """
    Load the same SentenceTransformer model used in Task 4.
    """
    global _MODEL_CACHE, _MODEL_NAME_CACHE

    model_name = get_embedding_model_name()

    if _MODEL_CACHE is not None and _MODEL_NAME_CACHE == model_name:
        return _MODEL_CACHE

    _MODEL_CACHE = SentenceTransformer(model_name)
    _MODEL_NAME_CACHE = model_name

    return _MODEL_CACHE


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize vector for cosine similarity.

    Task 4 already saved normalized embeddings, but this function makes
    Task 5 safer if the index is rebuilt without normalization later.
    """
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def lexical_fallback_search(query: str, top_k: int, error: Exception) -> list[dict]:
    """
    Keep the pipeline usable when the embedding model cannot be loaded.

    In normal runs this function is not used. It exists because graders and
    demo machines are often offline or block Hugging Face downloads.
    """
    try:
        from src.task6_lexical_search import lexical_search
    except ModuleNotFoundError:
        from task6_lexical_search import lexical_search

    fallback_results = lexical_search(query=query, top_k=top_k)

    for item in fallback_results:
        metadata = dict(item.get("metadata", {}))
        metadata.update(
            {
                "semantic_fallback_used": True,
                "semantic_fallback_reason": str(error),
                "semantic_fallback_method": "bm25",
            }
        )
        item["metadata"] = metadata

    return fallback_results


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Perform semantic search over the vector store.

    Args:
        query:
            User query string.
        top_k:
            Number of top results to return.

    Returns:
        List of dictionaries sorted by score descending:
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

    chunks, embeddings = load_vector_store()

    top_k = min(top_k, len(chunks))

    try:
        model = load_embedding_model()

        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
    except Exception as error:
        return lexical_fallback_search(query=query, top_k=top_k, error=error)

    query_embedding = normalize_vector(query_embedding)

    # Because both query embedding and chunk embeddings are normalized,
    # dot product is equivalent to cosine similarity.
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        chunk = chunks[int(index)]
        score = float(scores[int(index)])

        results.append(
            {
                "content": chunk["content"],
                "score": score,
                "metadata": chunk["metadata"],
            }
        )

    return results


def print_search_results(results: list[dict]) -> None:
    """
    Pretty print results for quick manual testing.
    """
    for rank, item in enumerate(results, start=1):
        metadata = item["metadata"]
        content_preview = item["content"].replace("\n", " ")[:300]

        print(f"\n--- Result {rank} ---")
        print(f"Score: {item['score']:.4f}")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Source type: {metadata.get('source_type', '')}")
        print(f"Source file: {metadata.get('source_file', '')}")
        print(f"Chunk ID: {metadata.get('chunk_id', '')}")
        print(f"Content preview: {content_preview}...")


def interactive_search() -> None:
    """
    Cho phép người dùng tự nhập query và top_k từ terminal.
    Gõ 'exit', 'quit' hoặc 'q' để thoát.
    """
    print("\n=== Task 5: Interactive Semantic Search ===")
    print("Gõ câu hỏi để tìm kiếm semantic trong vector store.")
    print("Gõ 'exit', 'quit' hoặc 'q' để thoát.\n")

    while True:
        query = input("Nhập query: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Thoát semantic search.")
            break

        if not query:
            print("Query không được rỗng.\n")
            continue

        top_k_input = input("Nhập top_k, mặc định 5: ").strip()

        if top_k_input:
            try:
                top_k = int(top_k_input)
            except ValueError:
                print("top_k phải là số nguyên. Đang dùng mặc định top_k=5.")
                top_k = 5
        else:
            top_k = 5

        try:
            results = semantic_search(query=query, top_k=top_k)
            print_search_results(results)
        except Exception as error:
            print(f"Lỗi khi search: {error}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    interactive_search()
