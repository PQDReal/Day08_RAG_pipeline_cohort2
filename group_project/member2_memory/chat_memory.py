"""
Member 2: Thân Văn Hoàng - 2A202600582
Chat Memory Module
==============================
Quản lý lịch sử hội thoại và trạng thái session cho RAG Chatbot.

Nhiệm vụ: Xử lý st.session_state, lưu/đọc lịch sử chat, cung cấp
          conversation context cho follow-up questions.
"""

import streamlit as st
from datetime import datetime


# ─── Constants ────────────────────────────────────────────────────────────────

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Xin chào! 👋 Tôi là **LawBot** — trợ lý AI chuyên về pháp luật "
        "phòng chống ma tuý Việt Nam.\n\n"
        "Tôi có thể giúp bạn:\n"
        "- ⚖️ Tra cứu hình phạt theo Bộ luật Hình sự\n"
        "- 📋 Giải thích Luật Phòng chống ma tuý 2021\n"
        "- 📰 Tóm tắt tin tức liên quan\n"
        "- 🔍 Tìm kiếm văn bản pháp quy\n\n"
        "Bạn muốn hỏi gì?"
    ),
    "sources": [],
    "retrieval_source": None,
    "timestamp": datetime.now().isoformat(),
}

MAX_CONTEXT_WINDOW = 10  # Số lượng tin nhắn tối đa giữ trong context


# ─── Core Functions ────────────────────────────────────────────────────────────

def init_chat_memory() -> None:
    """
    Khởi tạo bộ nhớ hội thoại trong session state.
    Gọi hàm này một lần khi ứng dụng khởi động.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [WELCOME_MESSAGE.copy()]

    if "chat_meta" not in st.session_state:
        st.session_state.chat_meta = {
            "total_messages": 0,
            "session_start": datetime.now().isoformat(),
            "last_query": None,
        }


def get_chat_history() -> list[dict]:
    """
    Trả về toàn bộ lịch sử hội thoại.

    Returns:
        List các dict với keys: role, content, sources, retrieval_source, timestamp
    """
    if "messages" not in st.session_state:
        init_chat_memory()
    return st.session_state.messages


def add_message(
    role: str,
    content: str,
    sources: list | None = None,
    retrieval_source: str | None = None,
) -> None:
    """
    Thêm tin nhắn mới vào lịch sử hội thoại.

    Args:
        role: "user" hoặc "assistant"
        content: Nội dung tin nhắn
        sources: Danh sách các nguồn tài liệu (nếu có)
        retrieval_source: Phương thức retrieval đã dùng ('hybrid', 'dense', 'pageindex', v.v.)
    """
    if "messages" not in st.session_state:
        init_chat_memory()

    msg = {
        "role": role,
        "content": content,
        "sources": sources or [],
        "retrieval_source": retrieval_source,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(msg)

    # Cập nhật metadata
    if "chat_meta" in st.session_state:
        st.session_state.chat_meta["total_messages"] += 1
        if role == "user":
            st.session_state.chat_meta["last_query"] = content


def clear_history() -> None:
    """
    Xoá toàn bộ lịch sử hội thoại và reset về trạng thái ban đầu.
    """
    st.session_state.messages = [WELCOME_MESSAGE.copy()]
    st.session_state.chat_meta = {
        "total_messages": 0,
        "session_start": datetime.now().isoformat(),
        "last_query": None,
    }


def get_context_window(n: int = MAX_CONTEXT_WINDOW) -> list[dict]:
    """
    Lấy N tin nhắn gần nhất để làm context cho follow-up questions.

    Args:
        n: Số lượng tin nhắn muốn lấy (default: MAX_CONTEXT_WINDOW)

    Returns:
        List các tin nhắn gần nhất (không tính welcome message đầu tiên)
    """
    history = get_chat_history()
    # Bỏ qua welcome message đầu tiên
    user_history = [m for m in history if not (m.get("role") == "assistant" and m == history[0])]
    return user_history[-n:] if len(user_history) > n else user_history


def get_session_stats() -> dict:
    """
    Trả về thống kê phiên hội thoại hiện tại.
    """
    if "chat_meta" not in st.session_state:
        return {}
    return st.session_state.chat_meta.copy()
