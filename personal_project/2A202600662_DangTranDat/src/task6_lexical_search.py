"""
Task 6 — Lexical Search Module

Goal:
    Implement lexical search using BM25 over chunks created in Task 4.

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

Why BM25:
    Semantic search is good for meaning-based retrieval, but it can miss
    exact entities such as names, law article numbers, and phrases.
    BM25 is strong for exact keyword matching, so it complements Task 5.

BM25 mechanism:
    BM25 scores documents based on:
    - term frequency: how often query terms appear in a chunk
    - inverse document frequency: rare terms are more important
    - document length normalization: avoids over-favoring long chunks
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


VECTOR_STORE_DIR = Path("data/vector_store")
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"

DEFAULT_TOP_K = 10


_CHUNKS_CACHE: list[dict] | None = None
_BM25_CACHE: BM25Okapi | None = None
_TOKENIZED_CORPUS_CACHE: list[list[str]] | None = None


def load_chunks() -> list[dict]:
    """
    Load chunks created in Task 4.

    Each chunk must have:
        - content
        - metadata
    """
    global _CHUNKS_CACHE

    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy chunks file: {CHUNKS_PATH}. "
            "Hãy chạy Task 4 trước."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("chunks.json rỗng. Hãy kiểm tra lại Task 4.")

    _CHUNKS_CACHE = chunks
    return _CHUNKS_CACHE


def tokenize(text: str) -> list[str]:
    """
    Tokenize Vietnamese/English text for BM25.

    This simple tokenizer:
    - lowercases text
    - keeps Vietnamese characters
    - keeps numbers for legal article references like 249, 250, 251
    - removes most punctuation

    Note:
        This is not as advanced as underthesea/pyvi, but it is lightweight
        and sufficient for the assignment.
    """
    text = text.lower()

    tokens = re.findall(
        r"[a-zA-ZÀ-ỹ0-9]+",
        text,
        flags=re.UNICODE,
    )

    return tokens


def build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Build BM25 index from chunks.

    We index both metadata title and chunk content because article title
    often contains important entities such as Nguyễn Công Trí, Chi Dân,
    Miu Lê, Điều 249, etc.
    """
    global _BM25_CACHE, _TOKENIZED_CORPUS_CACHE

    chunks = load_chunks()

    if _BM25_CACHE is not None:
        return _BM25_CACHE, chunks

    tokenized_corpus = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "")
        source_file = metadata.get("source_file", "")
        content = chunk.get("content", "")

        searchable_text = f"{title}\n{source_file}\n{content}"
        tokenized_corpus.append(tokenize(searchable_text))

    bm25 = BM25Okapi(tokenized_corpus)

    _BM25_CACHE = bm25
    _TOKENIZED_CORPUS_CACHE = tokenized_corpus

    return _BM25_CACHE, chunks


def lexical_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Perform BM25 lexical search.

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

    bm25, chunks = build_bm25_index()

    top_k = min(top_k, len(chunks))

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        chunk = chunks[index]
        score = float(scores[index])

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
    Pretty print BM25 results for manual testing.
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
    Allow user to search interactively from terminal.
    Type 'exit', 'quit', or 'q' to quit.
    """
    print("\n=== Task 6: Interactive Lexical Search / BM25 ===")
    print("Gõ câu hỏi hoặc từ khóa để tìm kiếm BM25 trong vector store.")
    print("BM25 phù hợp với tên riêng, số điều luật, cụm từ chính xác.")
    print("Gõ 'exit', 'quit' hoặc 'q' để thoát.\n")

    while True:
        query = input("Nhập query: ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Thoát lexical search.")
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
            results = lexical_search(query=query, top_k=top_k)
            print_search_results(results)
        except Exception as error:
            print(f"Lỗi khi search: {error}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    interactive_search()