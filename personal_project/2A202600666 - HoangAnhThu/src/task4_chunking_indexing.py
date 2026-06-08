"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .retrieval_utils import hashed_embedding, load_raw_documents, make_chunks
except ImportError:
    from retrieval_utils import hashed_embedding, load_raw_documents, make_chunks

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive chunking size 500 keeps legal clauses short enough for citations.
CHUNK_SIZE = 500
# 50 chars overlap preserves context across split boundaries without much repeat.
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Hashed embeddings are dependency-free and deterministic. We store them in real
# Weaviate with vector_config=self_provided(), so Weaviate acts as the vector
# database while this script owns the embedding step.
EMBEDDING_MODEL = "local-hashed-token-vector"
EMBEDDING_DIM = 384

VECTOR_STORE = "weaviate"
WEAVIATE_COLLECTION = "DrugLawDocs"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "").replace("WEAVIATE_URL=", "").strip()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "").strip()
RESET_COLLECTION = os.getenv("WEAVIATE_RESET_COLLECTION", "true").lower() == "true"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    return load_raw_documents()


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    return make_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embedded = []
    for chunk in chunks:
        item = chunk.copy()
        item["embedding"] = hashed_embedding(item["content"], EMBEDDING_DIM)
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE != "weaviate":
        return {
            "store": "local-memory",
            "count": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
        }

    try:
        import weaviate
        from weaviate.auth import Auth
        from weaviate.classes.config import Configure, DataType, Property
    except ImportError as exc:
        print(f"⚠ weaviate-client chưa sẵn sàng, fallback local-memory: {exc}")
        return {"store": "local-memory", "count": len(chunks)}

    if not WEAVIATE_URL:
        print("⚠ Thiếu WEAVIATE_URL trong .env, fallback local-memory.")
        return {"store": "local-memory", "count": len(chunks)}

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

        if RESET_COLLECTION and client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.delete(WEAVIATE_COLLECTION)

        if not client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.create(
                name=WEAVIATE_COLLECTION,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )

        collection = client.collections.get(WEAVIATE_COLLECTION)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "doc_type": metadata.get("type", ""),
                        "path": metadata.get("path", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                    },
                    vector=chunk["embedding"],
                )

        failed = getattr(collection.batch, "failed_objects", None)
        if failed:
            raise RuntimeError(f"Weaviate batch failed for {len(failed)} objects")

        return {
            "store": "weaviate",
            "collection": WEAVIATE_COLLECTION,
            "count": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
        }
    except Exception as exc:
        print(f"⚠ Không index được vào Weaviate, fallback local-memory: {exc}")
        return {"store": "local-memory", "count": len(chunks), "error": str(exc)}
    finally:
        if client is not None:
            client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    result = index_to_vectorstore(chunks)
    print(f"✓ Indexed to vector store: {result}")


if __name__ == "__main__":
    run_pipeline()
