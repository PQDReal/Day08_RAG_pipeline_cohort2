"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import math
from collections import Counter

try:
    from .retrieval_utils import normalize_scores, tokenize
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:
    from retrieval_utils import normalize_scores, tokenize
    from task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized = [tokenize(doc.get("content", "")) for doc in corpus]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avgdl": avgdl, "n_docs": len(tokenized)}


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], index: dict) -> float:
    if not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    score = 0.0
    k1 = 1.5
    b = 0.75
    n_docs = index["n_docs"]
    avgdl = index["avgdl"] or 1.0
    dl = len(doc_tokens)
    for token in query_tokens:
        if token not in tf:
            continue
        df = index["doc_freq"].get(token, 0)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf[token] + k1 * (1 - b + b * dl / avgdl)
        score += idf * (tf[token] * (k1 + 1)) / denom
    return float(score)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)
    scored = []
    for doc, tokens in zip(corpus, index["tokenized"]):
        score = _bm25_score(query_tokens, tokens, index)
        if score > 0:
            scored.append(
                {
                    "content": doc["content"],
                    "score": score,
                    "metadata": doc.get("metadata", {}),
                }
            )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return normalize_scores(scored)[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
