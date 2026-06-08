"""
Task 10 — Generation Có Citation.

Yêu cầu:
    1. Chọn top_k, top_p phù hợp và giải thích lý do
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import json
import copy
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# ROBUST IMPORT
# =============================================================================
# Chạy trực tiếp:
#     python src/task10_generation.py
# Import từ test:
#     from src.task10_generation import generate_with_citation

try:
    from task9_retrieval_pipeline import retrieve
except ModuleNotFoundError:
    from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: số chunks đưa vào context.
# Chọn 5 vì đủ evidence để trả lời, nhưng không quá dài gây nhiễu context.
TOP_K = 5

# top_p: nucleus sampling.
# Chọn 0.9 để model có độ linh hoạt ngôn ngữ vừa đủ, nhưng vẫn không quá random.
TOP_P = 0.9

# temperature: độ ngẫu nhiên.
# Chọn 0.3 vì RAG cần factual, ưu tiên bám tài liệu hơn sáng tạo.
TEMPERATURE = 0

# Model generation.
# Nếu dùng OpenRouter, nên set trong .env:
#     OPENROUTER_API_KEY=...
#     OPENAI_BASE_URL=https://openrouter.ai/api/v1
#     GENERATION_MODEL=openai/gpt-4o-mini
#
# Nếu dùng OpenAI trực tiếp:
#     OPENAI_API_KEY=...
#     GENERATION_MODEL=gpt-4o-mini
GENERATION_MODEL = os.getenv("GENERATION_MODEL")

if not GENERATION_MODEL:
    if os.getenv("OPENROUTER_API_KEY"):
        GENERATION_MODEL = "openai/gpt-4o-mini"
    else:
        GENERATION_MODEL = "gpt-4o-mini"


OUTPUT_DIR = Path("data/generated_answers")
OUTPUT_PATH = OUTPUT_DIR / "last_answer.json"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
Answer the following question comprehensively in Vietnamese.

Citation rules:
- Every factual claim MUST have a citation.
- Use citation IDs exactly as provided in the context, for example [S1], [S2].
- Only cite a source if that source directly supports the claim.
- Do not invent citations.

Evidence rules:
- Only use information from the provided context.
- If the context does not contain enough evidence, say exactly:
  "I cannot verify this information"
- Do not guess.
- Do not use outside knowledge.

Style:
- Answer in Vietnamese.
- Be clear, concise, and structured.
"""


# =============================================================================
# DOCUMENT REORDERING — tránh lost in the middle
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM thường chú ý tốt hơn ở đầu và cuối prompt, yếu hơn ở giữa.
    Vì vậy ta đặt:
        - chunk tốt nhất ở đầu
        - chunk tốt thứ hai ở cuối
        - các chunk còn lại ở giữa

    Input order theo score:
        [1, 2, 3, 4, 5]

    Output:
        [1, 3, 5, 4, 2]

    Args:
        chunks: List sorted by score descending.

    Returns:
        Reordered chunks.
    """
    if len(chunks) <= 2:
        return chunks

    # Lấy các vị trí lẻ theo human ranking: 1, 3, 5...
    # Index Python tương ứng: 0, 2, 4...
    front_part = chunks[0::2]

    # Lấy các vị trí chẵn theo human ranking: 2, 4...
    # Index Python tương ứng: 1, 3...
    # Đảo ngược để chunk tốt thứ 2 nằm cuối prompt.
    back_part = chunks[1::2][::-1]

    return front_part + back_part


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def build_source_label(chunk: dict, index: int) -> str:
    """
    Tạo nhãn nguồn dễ đọc cho citation.
    """
    metadata = chunk.get("metadata", {})

    title = (
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("source_file")
        or f"Source {index}"
    )

    section = metadata.get("section")
    source_file = metadata.get("source_file")
    chunk_id = metadata.get("chunk_id")
    url = metadata.get("url") or metadata.get("original_url")

    parts = [str(title)]

    if section:
        parts.append(f"Section: {section}")

    if source_file and source_file not in str(title):
        parts.append(f"File: {source_file}")

    if chunk_id:
        parts.append(f"Chunk: {chunk_id}")

    if url:
        parts.append(f"URL: {url}")

    return " | ".join(parts)


def attach_citation_ids(chunks: list[dict]) -> list[dict]:
    """
    Gắn citation_id [S1], [S2] vào metadata của từng chunk.
    Không sửa object gốc để tránh side-effect.
    """
    prepared_chunks = []

    for index, chunk in enumerate(chunks, start=1):
        copied_chunk = copy.deepcopy(chunk)
        metadata = dict(copied_chunk.get("metadata", {}))

        citation_id = f"S{index}"
        source_label = build_source_label(copied_chunk, index)

        metadata["citation_id"] = citation_id
        metadata["source_label"] = source_label

        copied_chunk["metadata"] = metadata
        prepared_chunks.append(copied_chunk)

    return prepared_chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có citation_id để LLM cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        citation_id = metadata.get("citation_id", f"S{index}")
        source_label = metadata.get("source_label", build_source_label(chunk, index))
        score = chunk.get("score", 0.0)
        content = chunk.get("content", "").strip()

        context_parts.append(
            f"[{citation_id}]\n"
            f"Source: {source_label}\n"
            f"Score: {score}\n"
            f"Content:\n{content}\n"
        )

    return "\n---\n".join(context_parts)


def build_sources(chunks: list[dict]) -> list[dict]:
    """
    Tạo danh sách sources trả về cho output cuối.
    """
    sources = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")

        sources.append(
            {
                "citation_id": metadata.get("citation_id"),
                "source_label": metadata.get("source_label"),
                "score": float(chunk.get("score", 0.0)),
                "metadata": metadata,
                "content_preview": content[:500],
            }
        )

    return sources


# =============================================================================
# LLM CALL
# =============================================================================

def get_openai_compatible_client():
    """
    Tạo OpenAI-compatible client.

    Hỗ trợ:
    - OpenAI trực tiếp qua OPENAI_API_KEY
    - OpenRouter qua OPENROUTER_API_KEY + OPENAI_BASE_URL
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        return None

    from openai import OpenAI

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Gọi LLM để generate answer có citation.
    """
    client = get_openai_compatible_client()

    if client is None:
        raise RuntimeError(
            "Missing generation API key. "
            "Hãy thêm OPENAI_API_KEY hoặc OPENROUTER_API_KEY vào .env"
        )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return response.choices[0].message.content


def build_extractive_fallback_answer(query: str, sources: list[dict]) -> str:
    """
    Fallback nếu chưa có API key hoặc LLM lỗi.
    Vẫn trả lời có citation bằng cách trích evidence từ retrieved chunks.
    """
    if not sources:
        return "I cannot verify this information"

    lines = [
        "LLM generation chưa chạy được, nhưng hệ thống đã truy xuất được các bằng chứng liên quan:",
        "",
    ]

    for source in sources:
        citation_id = source.get("citation_id")
        preview = source.get("content_preview", "").replace("\n", " ").strip()

        if preview:
            lines.append(f"- {preview} [{citation_id}]")

    return "\n".join(lines).strip()


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str,
            'used_llm': bool,
            'model': str,
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên lớn hơn 0.")

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        result = {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
            "used_llm": False,
            "model": GENERATION_MODEL,
        }
        save_generation_result(query, result)
        return result

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Attach citation IDs
    cited_chunks = attach_citation_ids(reordered)

    # Step 4: Format context
    context = format_context(cited_chunks)
    sources = build_sources(cited_chunks)

    user_message = f"""
Context:
{context}

---

Question:
{query}

Instructions:
- Answer using only the context.
- Cite every factual claim with [S1], [S2], etc.
- If evidence is insufficient, answer exactly: I cannot verify this information
"""

    used_llm = True

    try:
        answer = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
        )
    except Exception as error:
        used_llm = False
        answer = build_extractive_fallback_answer(query=query, sources=sources)
        answer += f"\n\nGhi chú kỹ thuật: LLM generation chưa chạy được. Lỗi: {error}"

    retrieval_source = (
        chunks[0].get("metadata", {}).get("retrieval_pipeline")
        or chunks[0].get("metadata", {}).get("retrieval_type")
        or "hybrid"
    )

    result = {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "used_llm": used_llm,
        "model": GENERATION_MODEL,
    }

    save_generation_result(query, result)

    return result


def save_generation_result(query: str, result: dict) -> None:
    """
    Lưu kết quả generation gần nhất để demo/debug.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "query": query,
        **result,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =============================================================================
# CLI
# =============================================================================

def print_result(result: dict) -> None:
    print("\nA:")
    print(result["answer"])

    print(
        f"\n[Sources: {len(result['sources'])} chunks "
        f"| via {result['retrieval_source']} "
        f"| used_llm={result['used_llm']} "
        f"| model={result['model']}]"
    )

    print("\nSource details:")
    for source in result["sources"]:
        print(f"- [{source['citation_id']}] {source['source_label']}")


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)

        result = generate_with_citation(q)
        print_result(result)
