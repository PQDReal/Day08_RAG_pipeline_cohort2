"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from .retrieval_utils import tokenize
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:
    from retrieval_utils import tokenize
    from task4_chunking_indexing import chunk_documents, load_documents

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
USE_PAGEINDEX_API = os.getenv("USE_PAGEINDEX_API", "true").lower() == "true"
PAGEINDEX_FOLDER_NAME = os.getenv("PAGEINDEX_FOLDER_NAME", "Day08_RAG_DrugLaw")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if USE_PAGEINDEX_API and PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            folder = client.create_folder(
                name=PAGEINDEX_FOLDER_NAME,
                description="Day08 RAG markdown documents",
            )
            folder_id = folder.get("id") or folder.get("folder_id")
            uploaded = []
            for md_file in STANDARDIZED_DIR.rglob("*.md"):
                result = client.submit_document(
                    file_path=str(md_file),
                    folder_id=folder_id,
                )
                uploaded.append(
                    {
                        "filename": md_file.name,
                        "type": md_file.parent.name,
                        "doc_id": result.get("id") or result.get("doc_id") or result.get("document_id"),
                        "provider": "pageindex",
                    }
                )
            return uploaded
        except Exception as exc:
            print(f"⚠ PageIndex upload lỗi, fallback local manifest: {exc}")

    uploaded = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        uploaded.append({"filename": md_file.name, "type": md_file.parent.name, "provider": "local"})
    return uploaded


def _pageindex_search_api(query: str, top_k: int) -> list[dict]:
    """Best-effort PageIndex SDK retrieval, with generic response parsing."""
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    documents = client.list_documents(limit=50)
    items = documents.get("documents") or documents.get("items") or documents.get("data") or []
    results = []

    for doc in items:
        doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
        if not doc_id:
            continue
        if hasattr(client, "is_retrieval_ready"):
            try:
                if not client.is_retrieval_ready(doc_id):
                    continue
            except Exception:
                pass

        query_result = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = query_result.get("retrieval_id") or query_result.get("id")
        payload = client.get_retrieval(retrieval_id) if retrieval_id else query_result
        chunks = (
            payload.get("chunks")
            or payload.get("results")
            or payload.get("retrieval")
            or payload.get("data")
            or []
        )
        if isinstance(chunks, dict):
            chunks = chunks.get("chunks") or chunks.get("results") or []

        for chunk in chunks:
            content = (
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("answer")
                or chunk.get("markdown")
                or ""
            )
            if not content:
                continue
            metadata = chunk.get("metadata") or {}
            metadata.setdefault("source", doc.get("filename") or doc.get("name") or str(doc_id))
            metadata.setdefault("type", "pageindex")
            results.append(
                {
                    "content": content,
                    "score": float(chunk.get("score", chunk.get("relevance", 1.0))),
                    "metadata": metadata,
                    "source": "pageindex",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _pageindex_search_local(query: str, top_k: int) -> list[dict]:
    """Local vectorless fallback when PageIndex is not configured or unavailable."""
    query_tokens = set(tokenize(query))
    chunks = chunk_documents(load_documents())
    scored = []
    for chunk in chunks:
        tokens = set(tokenize(chunk["content"]))
        overlap = len(query_tokens & tokens)
        score = overlap / max(len(query_tokens), 1)
        if score > 0 or not query_tokens:
            scored.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                    "source": "pageindex",
                }
            )

    if not scored:
        scored = [
            {
                "content": chunk["content"],
                "score": 0.0,
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
            for chunk in chunks[:top_k]
        ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0:
        return []

    if USE_PAGEINDEX_API and PAGEINDEX_API_KEY:
        try:
            results = _pageindex_search_api(query, top_k)
            if results:
                return results
        except Exception as exc:
            print(f"⚠ PageIndex query lỗi, fallback local: {exc}")

    return _pageindex_search_local(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
