"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn khoảng 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or metadata.get("title") or f"Source {i}"
        doc_type = metadata.get("type", "unknown")
        section = metadata.get("section_title") or metadata.get("physical_index") or ""
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | "
            f"Section: {section} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def _source_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("title") or f"Document {index}"
    section = metadata.get("section_title") or metadata.get("physical_index")
    if section:
        return f"{source}, {section}"
    return str(source)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?。])\s+", cleaned)
    if len(sentences) == 1:
        sentences = re.split(r"(?<=\.)\s+|(?<=;)\s+", cleaned)
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) <= 30:
            continue
        # Avoid using long fragments cut from the middle/end of a chunk.
        if len(sentence) > 120 and sentence[-1] not in ".!?。":
            continue
        if any(noise in sentence.lower() for noise in ["đăng nhập", "đặt báo", "xem tất cả", "hotline"]):
            continue
        if any(boilerplate in sentence.lower() for boilerplate in ["được ban hành căn cứ", "tài liệu triển khai quan trọng"]):
            continue
        clean_sentences.append(sentence)
    return clean_sentences


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    query_tokens = _tokenize(query)
    if not chunks or not query_tokens:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    scored_sentences = []
    legal_intent = bool(query_tokens & {"luật", "pháp", "hình", "phạt", "điều", "nghị", "định", "cai", "nghiện"})
    requires_addiction_terms = bool(query_tokens & {"cai", "nghiện"})
    requires_penalty_terms = bool(query_tokens & {"hình", "phạt"})
    for chunk_index, chunk in enumerate(chunks, 1):
        source = _source_label(chunk, chunk_index)
        metadata = chunk.get("metadata", {})
        source_name = str(metadata.get("source", ""))
        legal_bonus = 0.25 if legal_intent and (metadata.get("type") == "legal" or "luat" in source_name or "nghi-dinh" in source_name) else 0.0
        for sentence in _split_sentences(chunk.get("content", "")):
            sentence_tokens = _tokenize(sentence)
            overlap = len(query_tokens & sentence_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1) + legal_bonus
            if requires_addiction_terms and (sentence_tokens & {"cai", "nghiện"}):
                score += 0.25
            if requires_penalty_terms and (sentence_tokens & {"phạt", "tù", "tội"}):
                score += 0.25
            if "nguồn:" in sentence.lower():
                score -= 0.15
            scored_sentences.append((score, sentence, source))

    if not scored_sentences:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence, source in scored_sentences:
        normalized = sentence.lower()[:120]
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(f"{sentence} [{source}]")
        if len(selected) >= 3:
            break

    return "\n".join(f"- {sentence}" for sentence in selected)


def _call_openai_generation(query: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-xxx":
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        error_type = exc.__class__.__name__
        print(f"⚠ OpenAI generation unavailable; using extractive fallback ({error_type}, status={status})")
        return None


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
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _call_openai_generation(query, context)
    if not answer:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "context": context,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
