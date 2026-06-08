"""
Task 4 — Chunking & Indexing

Goal:
    Convert all standardized Markdown documents into retrievable chunks,
    generate embeddings, and save a local vector index for later semantic search.

Assignment requirements:
    1. Clearly state chunking strategy, chunk_size, overlap, and reason.
    2. Clearly state embedding model, embedding dimension, and reason.
    3. Successfully index all documents from data/standardized/.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# TASK 4 — CONFIGURATION & DESIGN DECISIONS
# ============================================================

STANDARDIZED_DIR = Path("data/standardized")
INDEX_DIR = Path("data/vector_store")

CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
MANIFEST_PATH = INDEX_DIR / "index_manifest.json"
DOCUMENTS_INDEX_PATH = INDEX_DIR / "documents_indexed.json"


# ------------------------------------------------------------
# 1. CHUNKING STRATEGY
# ------------------------------------------------------------
# Strategy used:
#     RecursiveCharacterTextSplitter
#
# Why this strategy:
#     The dataset contains mixed document types:
#     - Long legal documents converted from DOCX/PDF
#     - Short news articles crawled from online newspapers
#
#     After conversion to Markdown, legal documents may not always have
#     clean and consistent headings. RecursiveCharacterTextSplitter is
#     a safe default because it tries to split by larger boundaries first
#     such as paragraphs and lines, then falls back to smaller separators
#     when necessary.
#
# Chunk size:
#     800 characters
#
# Why chunk_size = 800:
#     Legal documents often contain article numbers, clauses, penalties,
#     and conditions in the same paragraph. A chunk size of 800 characters
#     is large enough to preserve legal context but still small enough for
#     precise retrieval.
#
# Chunk overlap:
#     120 characters
#
# Why overlap = 120:
#     Legal meaning can be lost if a clause is cut between two chunks.
#     Overlap helps preserve continuity across chunk boundaries.
# ------------------------------------------------------------

CHUNKING_STRATEGY = "RecursiveCharacterTextSplitter"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ------------------------------------------------------------
# 2. EMBEDDING MODEL
# ------------------------------------------------------------
# Model used:
#     sentence-transformers/all-MiniLM-L6-v2
#
# Dimension:
#     384
#
# Why this model:
#     - Runs locally on CPU
#     - Does not require API keys
#     - Does not cost tokens
#     - Lightweight enough for weak laptops
#     - Suitable for a small RAG assignment/demo pipeline
#
# Trade-off:
#     This is not the strongest model for Vietnamese semantic retrieval.
#     For production or higher-quality Vietnamese retrieval, a multilingual
#     model such as BAAI/bge-m3 or paraphrase-multilingual-MiniLM-L12-v2
#     may perform better.
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_EMBEDDING_DIMENSION = 384


# ------------------------------------------------------------
# 3. VECTOR STORE
# ------------------------------------------------------------
# Vector store used:
#     Local NumPy-based vector store
#
# Files saved:
#     - data/vector_store/chunks.json
#     - data/vector_store/embeddings.npy
#     - data/vector_store/index_manifest.json
#     - data/vector_store/documents_indexed.json
#
# Why local vector store:
#     Weaviate is recommended in the assignment, but local NumPy indexing
#     is simpler and more reliable for an individual assignment. It avoids
#     Docker/cloud setup issues and is enough for Task 5 semantic search
#     using cosine similarity.
# ------------------------------------------------------------

VECTOR_STORE_TYPE = "local_numpy"


def parse_frontmatter(markdown_text: str) -> tuple[dict, str]:
    """
    Extract YAML-like frontmatter from a Markdown file.

    Example:
        ---
        title: "abc"
        source_type: "news"
        url: "https://..."
        ---

    Returns:
        metadata: dict
        body_text: str
    """
    metadata = {}

    if not markdown_text.startswith("---"):
        return metadata, markdown_text

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", markdown_text, re.DOTALL)

    if not match:
        return metadata, markdown_text

    frontmatter = match.group(1)
    body_text = markdown_text[match.end():]

    for line in frontmatter.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        metadata[key] = value

    return metadata, body_text


def load_markdown_documents() -> list[dict]:
    """
    Load all Markdown files from data/standardized/.

    Returns:
        List of documents:
        [
            {
                "content": "...",
                "metadata": {...}
            }
        ]
    """
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    documents = []

    for doc_index, file_path in enumerate(md_files):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        frontmatter, body = parse_frontmatter(text)

        relative_path = file_path.relative_to(STANDARDIZED_DIR)
        parts = relative_path.parts

        inferred_source_type = parts[0] if len(parts) > 1 else "unknown"
        source_type = frontmatter.get("source_type", inferred_source_type)

        metadata = {
            **frontmatter,
            "doc_index": doc_index,
            "source_path": str(file_path),
            "source_file": file_path.name,
            "source_type": source_type,
            "relative_path": str(relative_path),
        }

        documents.append(
            {
                "content": body.strip(),
                "metadata": metadata,
            }
        )

    return documents


def load_documents() -> list[dict]:
    """
    Public wrapper required by the assignment test suite.

    The implementation name is kept as load_markdown_documents() because it is
    more explicit, while this wrapper preserves the expected interface.
    """
    return load_markdown_documents()


def build_chunks(documents: list[dict]) -> list[dict]:
    """
    Split all documents into chunks.

    Each chunk contains:
        - id
        - content
        - metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        content = doc["content"]
        doc_metadata = doc["metadata"]
        doc_index = doc_metadata["doc_index"]

        if not content.strip():
            continue

        split_texts = splitter.split_text(content)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_text = chunk_text.strip()

            # Very short chunks often carry little retrieval value.
            if len(chunk_text) < 50:
                continue

            chunk_id = f"doc_{doc_index:03d}_chunk_{chunk_index:04d}"

            chunk_metadata = {
                **doc_metadata,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "chunking_strategy": CHUNKING_STRATEGY,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }

            all_chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return all_chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Public wrapper required by the assignment test suite.
    """
    return build_chunks(documents)


def build_embeddings(chunks: list[dict]) -> tuple[np.ndarray, int]:
    """
    Generate embeddings for all chunks.

    normalize_embeddings=True:
        The vectors are normalized, so cosine similarity can be computed
        by dot product in Task 5.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    actual_dimension = model.get_sentence_embedding_dimension()

    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    return embeddings, actual_dimension


def summarize_indexed_documents(documents: list[dict], chunks: list[dict]) -> list[dict]:
    """
    Create a document-level summary to prove that all Markdown documents
    were included in the index.
    """
    chunk_count_by_doc = defaultdict(int)

    for chunk in chunks:
        doc_index = chunk["metadata"]["doc_index"]
        chunk_count_by_doc[doc_index] += 1

    indexed_documents = []

    for doc in documents:
        metadata = doc["metadata"]
        doc_index = metadata["doc_index"]
        content_length = len(doc["content"])

        indexed_documents.append(
            {
                "doc_index": doc_index,
                "title": metadata.get("title", metadata.get("source_file", "")),
                "source_type": metadata.get("source_type", "unknown"),
                "source_file": metadata.get("source_file", ""),
                "source_path": metadata.get("source_path", ""),
                "relative_path": metadata.get("relative_path", ""),
                "content_length": content_length,
                "num_chunks": chunk_count_by_doc.get(doc_index, 0),
                "indexed": chunk_count_by_doc.get(doc_index, 0) > 0,
            }
        )

    return indexed_documents


def save_vector_store(
    documents: list[dict],
    chunks: list[dict],
    embeddings: np.ndarray,
    actual_embedding_dimension: int,
) -> None:
    """
    Save the local vector store and manifest.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(EMBEDDINGS_PATH, embeddings)

    indexed_documents = summarize_indexed_documents(documents, chunks)

    with open(DOCUMENTS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(indexed_documents, f, ensure_ascii=False, indent=2)

    documents_indexed_count = sum(1 for doc in indexed_documents if doc["indexed"])

    manifest = {
        "task": "Task 4 - Chunking & Indexing",
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "source_dir": str(STANDARDIZED_DIR),
        "index_dir": str(INDEX_DIR),

        "documents_total": len(documents),
        "documents_indexed": documents_indexed_count,
        "num_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),

        "chunking": {
            "strategy": CHUNKING_STRATEGY,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "reason": (
                "RecursiveCharacterTextSplitter is used because the dataset contains "
                "mixed legal documents and news articles. It preserves larger text "
                "boundaries such as paragraphs and lines when possible, and falls back "
                "to smaller separators when needed. The chunk_size of 800 keeps enough "
                "legal context such as article numbers, clauses, and penalties. The "
                "overlap of 120 reduces the risk of cutting important legal meaning "
                "between chunks."
            ),
        },

        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "expected_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "actual_dimension": actual_embedding_dimension,
            "dimension_from_matrix": int(embeddings.shape[1]),
            "reason": (
                "The selected SentenceTransformer model runs locally, does not require "
                "API keys, does not consume paid tokens, and is lightweight enough for "
                "weak laptops. It produces 384-dimensional embeddings, suitable for a "
                "small RAG demo pipeline."
            ),
            "trade_off": (
                "This model is not the strongest embedding model for Vietnamese. "
                "For production or better Vietnamese retrieval quality, multilingual "
                "models such as BAAI/bge-m3 or paraphrase-multilingual-MiniLM-L12-v2 "
                "should be considered."
            ),
        },

        "vector_store": {
            "type": VECTOR_STORE_TYPE,
            "chunks_file": str(CHUNKS_PATH),
            "embeddings_file": str(EMBEDDINGS_PATH),
            "documents_index_file": str(DOCUMENTS_INDEX_PATH),
            "manifest_file": str(MANIFEST_PATH),
            "reason": (
                "A local NumPy vector store is used to avoid Docker/cloud complexity. "
                "It is sufficient for this assignment and can be queried with cosine "
                "similarity in Task 5."
            ),
        },

        "index_validation": {
            "embeddings_match_chunks": embeddings.shape[0] == len(chunks),
            "embedding_dimension_valid": int(embeddings.shape[1]) == actual_embedding_dimension,
            "all_non_empty_documents_indexed": all(
                doc["indexed"] for doc in indexed_documents if doc["content_length"] > 0
            ),
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("=== Task 4: Chunking & Indexing ===\n")

    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục {STANDARDIZED_DIR}")

    documents = load_markdown_documents()
    print(f"Loaded Markdown documents: {len(documents)}")

    if not documents:
        raise ValueError("Không có file Markdown nào trong data/standardized/")

    print("\nChunking configuration:")
    print(f"- Strategy: {CHUNKING_STRATEGY}")
    print(f"- Chunk size: {CHUNK_SIZE}")
    print(f"- Chunk overlap: {CHUNK_OVERLAP}")
    print(
        "- Reason: robust for mixed legal documents and news articles; "
        "preserves legal context while keeping chunks retrievable."
    )

    chunks = build_chunks(documents)
    print(f"\nCreated chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("Không tạo được chunk nào. Kiểm tra nội dung file Markdown.")

    print("\nEmbedding configuration:")
    print(f"- Model: {EMBEDDING_MODEL_NAME}")
    print(f"- Expected dimension: {EXPECTED_EMBEDDING_DIMENSION}")
    print("- Reason: local, lightweight, no API key, no token cost.")

    embeddings, actual_embedding_dimension = build_embeddings(chunks)

    print(f"\nActual embedding dimension: {actual_embedding_dimension}")
    print(f"Embedding matrix shape: {embeddings.shape}")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Số lượng embeddings không khớp số lượng chunks.")

    if embeddings.shape[1] != actual_embedding_dimension:
        raise ValueError("Embedding dimension không khớp với model.")

    save_vector_store(
        documents=documents,
        chunks=chunks,
        embeddings=embeddings,
        actual_embedding_dimension=actual_embedding_dimension,
    )

    indexed_documents = summarize_indexed_documents(documents, chunks)
    documents_indexed_count = sum(1 for doc in indexed_documents if doc["indexed"])

    print("\n✅ TASK 4 COMPLETED: Chunking & Indexing successful")
    print(f"Documents loaded: {len(documents)}")
    print(f"Documents indexed: {documents_indexed_count}/{len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding dimension: {actual_embedding_dimension}")
    print(f"Vector store type: {VECTOR_STORE_TYPE}")

    print("\nSaved files:")
    print(f"- Chunks: {CHUNKS_PATH}")
    print(f"- Embeddings: {EMBEDDINGS_PATH}")
    print(f"- Documents index: {DOCUMENTS_INDEX_PATH}")
    print(f"- Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
