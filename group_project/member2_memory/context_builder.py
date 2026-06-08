"""
Member 2: Thân Văn Hoàng - 2A202600582
Conversation Context Builder
=========================================
Ghép lịch sử hội thoại thành chuỗi prompt cho LLM.

Nhiệm vụ: Format conversation history để LLM hiểu ngữ cảnh
          của follow-up questions.
"""


def build_conversation_context(messages: list[dict], max_messages: int = 6) -> str:
    """
    Chuyển danh sách tin nhắn thành chuỗi context cho LLM.

    Ví dụ output:
        [Người dùng]: Hình phạt tàng trữ ma tuý là gì?
        [Trợ lý]: Theo Điều 249 BLHS 2015, hình phạt...
        [Người dùng]: Còn nếu là lần đầu vi phạm thì sao?

    Args:
        messages: Danh sách tin nhắn từ chat_memory.get_context_window()
        max_messages: Số tin nhắn tối đa đưa vào context

    Returns:
        Chuỗi văn bản thể hiện lịch sử hội thoại
    """
    if not messages:
        return ""

    # Chỉ lấy N tin nhắn gần nhất
    recent = messages[-max_messages:]

    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if role == "user":
            lines.append(f"[Người dùng]: {content}")
        elif role == "assistant":
            # Cắt ngắn câu trả lời dài để tiết kiệm token
            truncated = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"[Trợ lý]: {truncated}")

    return "\n".join(lines)


def build_system_prompt(conversation_context: str = "") -> str:
    """
    Tạo system prompt đầy đủ với lịch sử hội thoại.

    Args:
        conversation_context: Chuỗi lịch sử từ build_conversation_context()

    Returns:
        System prompt hoàn chỉnh cho LLM
    """
    base_prompt = (
        "Bạn là LawBot — trợ lý AI chuyên về pháp luật phòng chống ma tuý Việt Nam. "
        "Hãy trả lời chính xác, có trích dẫn nguồn từ văn bản pháp luật. "
        "Nếu không chắc chắn, hãy nói rõ. "
        "Luôn trả lời bằng tiếng Việt.\n\n"
    )

    if conversation_context:
        base_prompt += (
            "LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n"
            f"{conversation_context}\n\n"
            "Hãy trả lời câu hỏi mới nhất, có tham khảo ngữ cảnh hội thoại phía trên nếu cần.\n"
        )

    return base_prompt


def is_followup_question(current_query: str, history: list[dict]) -> bool:
    """
    Phát hiện xem câu hỏi hiện tại có phải follow-up không.

    Sử dụng heuristics đơn giản: câu ngắn hoặc chứa từ tham chiếu.

    Args:
        current_query: Câu hỏi hiện tại
        history: Lịch sử hội thoại

    Returns:
        True nếu khả năng cao là follow-up question
    """
    if not history:
        return False

    # Từ tham chiếu thường gặp trong follow-up
    reference_words = [
        "còn", "thế thì", "vậy", "nếu vậy", "thêm", "khác",
        "nữa", "tiếp", "giải thích", "ví dụ", "cụ thể hơn",
        "so với", "trường hợp", "nếu", "điều đó",
    ]

    query_lower = current_query.lower()

    # Câu ngắn < 20 từ hoặc chứa từ tham chiếu
    word_count = len(current_query.split())
    has_reference = any(word in query_lower for word in reference_words)

    return word_count < 15 or has_reference
