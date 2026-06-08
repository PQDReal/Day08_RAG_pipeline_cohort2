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

import hashlib
import json
import math
import re
import warnings
from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_FILE = INDEX_DIR / "task4_chunks.jsonl"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# RecursiveCharacterTextSplitter style is a safe default for mixed legal/news
# markdown: it preserves paragraphs when possible and still handles noisy HTML.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# FastEmbed downloads and runs this multilingual sentence-transformer through
# ONNX. It is much lighter than bge-m3, supports Vietnamese reasonably well,
# and keeps the lab local after the first model download.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Local JSONL index is simple and inspectable in class; Weaviate is the better
# production choice when a server is available for hybrid retrieval.
VECTOR_STORE = "jsonl"  # "weaviate" | "chromadb" | "faiss" | "jsonl"
USE_FASTEMBED = True


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
    return documents


def _split_text_recursive(text: str) -> list[str]:
    """Dependency-free recursive splitter compatible with the lab tests."""
    chunks: list[str] = []

    def trim_to_boundary(value: str, limit: int) -> str:
        value = value.strip()
        if len(value) <= limit:
            return value
        truncated = value[:limit].rstrip()
        boundary = max(
            truncated.rfind(". "),
            truncated.rfind("\n"),
            truncated.rfind("; "),
            truncated.rfind(" "),
        )
        if boundary > int(limit * 0.75):
            return truncated[:boundary].strip()
        return truncated.strip()

    def split_piece(piece: str, separators: list[str]) -> None:
        piece = piece.strip()
        if not piece:
            return
        if len(piece) <= CHUNK_SIZE:
            chunks.append(piece)
            return
        if not separators:
            for start in range(0, len(piece), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = piece[start:start + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(chunk)
            return

        separator = separators[0]
        parts = piece.split(separator) if separator else list(piece)
        buffer = ""
        for part in parts:
            candidate = part if not buffer else buffer + separator + part
            if len(candidate) <= CHUNK_SIZE:
                buffer = candidate
                continue
            split_piece(buffer, separators[1:])
            buffer = part
        split_piece(buffer, separators[1:])

    split_piece(text, ["\n\n", "\n", ". ", " ", ""])
    return [trim_to_boundary(chunk, int(CHUNK_SIZE * 1.1)) for chunk in chunks]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc_index, doc in enumerate(documents):
        splits = _split_text_recursive(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                },
            })
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _hash_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in _tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


@lru_cache(maxsize=1)
def _get_fastembed_model():
    from fastembed import TextEmbedding

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mean pooling.*")
        return TextEmbedding(model_name=EMBEDDING_MODEL)


def _fastembed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_fastembed_model()
    return [embedding.tolist() for embedding in model.embed(texts)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text with FastEmbed, falling back to hashing if unavailable."""
    if USE_FASTEMBED:
        try:
            return _fastembed_texts(texts)
        except Exception as exc:
            print(f"⚠ FastEmbed unavailable ({exc}); falling back to local hashing.")
    return [_hash_embedding(text) for text in texts]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return INDEX_FILE


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

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
