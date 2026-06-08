"""
Member 3: Phạm Quang Dũng - 2A202600703
RAG Integration Module
====================================
Kết nối giao diện chat với pipeline RAG (Task 9 + Task 10).

Nhiệm vụ: Nhận query từ UI, gọi retrieval + generation pipeline,
          trả về structured response với answer + sources.
"""

import sys
import os
from typing import TYPE_CHECKING

# ─── Path Setup ───────────────────────────────────────────────────────────────
# Đảm bảo có thể import từ thư mục src (bài cá nhân)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if TYPE_CHECKING:
    from .config import RAGConfig

from .config import DEFAULT_CONFIG, RAGConfig


# ─── Typing ───────────────────────────────────────────────────────────────────

ResponseDict = dict  # {answer: str, sources: list, retrieval_source: str, config_used: str}


# ─── Main Integration Function ────────────────────────────────────────────────

def get_rag_response(
    query: str,
    conversation_context: str = "",
    config: RAGConfig | None = None,
) -> ResponseDict:
    """
    Gọi RAG pipeline và trả về câu trả lời có citation.

    Flow:
        1. Chuẩn bị query (kết hợp với conversation context nếu là follow-up)
        2. Gọi generate_with_citation từ Task 10
        3. Format và trả về response

    Args:
        query: Câu hỏi từ người dùng
        conversation_context: Lịch sử hội thoại dạng text (từ context_builder)
        config: Cấu hình RAG (dùng DEFAULT_CONFIG nếu None)

    Returns:
        Dict với keys:
            - answer (str): Câu trả lời có citation
            - sources (list): Danh sách nguồn tài liệu
            - retrieval_source (str): Phương thức retrieval đã dùng
            - config_used (str): Tên config đã dùng
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Xây dựng augmented query nếu có conversation context
    augmented_query = _build_augmented_query(query, conversation_context)

    try:
        from src.task10_generation import generate_with_citation
        result = generate_with_citation(augmented_query, top_k=config.top_k)

        return {
            "answer": result.get("answer", "Không có câu trả lời."),
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source", "task10"),
            "config_used": config.config_name,
        }

    except NotImplementedError:
        return _fallback_response(query, config, reason="not_implemented")

    except ImportError as e:
        return _fallback_response(query, config, reason=f"import_error: {e}")

    except Exception as e:
        return _error_response(str(e), config)


def get_rag_response_with_config(
    query: str,
    conversation_context: str = "",
    config_name: str = "hybrid_rerank",
) -> ResponseDict:
    """
    Wrapper tiện lợi để chọn config theo tên (dùng trong A/B test).

    Args:
        query: Câu hỏi
        conversation_context: Lịch sử hội thoại
        config_name: Tên config ('hybrid_rerank', 'dense_only', 'hybrid_aggressive')
    """
    from .config import AVAILABLE_CONFIGS, DEFAULT_CONFIG

    # Tìm config theo tên
    chosen_config = DEFAULT_CONFIG
    for display_name, cfg in AVAILABLE_CONFIGS.items():
        if cfg.config_name == config_name or display_name == config_name:
            chosen_config = cfg
            break

    return get_rag_response(query, conversation_context, chosen_config)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _build_augmented_query(query: str, conversation_context: str) -> str:
    """Ghép conversation context vào query nếu có."""
    if not conversation_context:
        return query

    return (
        f"Ngữ cảnh hội thoại:\n{conversation_context}\n\n"
        f"Câu hỏi hiện tại: {query}"
    )


def _fallback_response(query: str, config: RAGConfig, reason: str) -> ResponseDict:
    """Trả về demo response khi pipeline chưa sẵn sàng."""
    sample_answers = {
        "ma túy": (
            "**Theo Điều 249 BLHS 2015**, tội tàng trữ trái phép chất ma tuý bị phạt:\n"
            "- **Khoản 1**: Phạt tù 1-5 năm (tàng trữ dưới ngưỡng)\n"
            "- **Khoản 2**: Phạt tù 5-10 năm (từ 1g heroin/cocaine)\n"
            "- **Khoản 3**: Phạt tù 10-15 năm (từ 5g-30g heroin)\n"
            "- **Khoản 4**: Phạt tù 15-20 năm hoặc tù chung thân (≥30g heroin)\n\n"
            "> *[Nguồn mẫu — Task 10 chưa được implement]*"
        ),
        "cai nghiện": (
            "**Luật Phòng chống ma tuý 2021** quy định 4 hình thức cai nghiện:\n"
            "1. Cai nghiện tự nguyện tại gia đình\n"
            "2. Cai nghiện tự nguyện tại cộng đồng\n"
            "3. Cai nghiện tự nguyện tại cơ sở cai nghiện\n"
            "4. Cai nghiện bắt buộc tại cơ sở cai nghiện\n\n"
            "> *[Nguồn mẫu — Task 10 chưa được implement]*"
        ),
    }

    # Tìm câu trả lời phù hợp nhất
    answer = next(
        (ans for key, ans in sample_answers.items() if key in query.lower()),
        (
            f"Đây là **câu trả lời mẫu** cho câu hỏi: *{query}*\n\n"
            "Pipeline RAG chưa được implement đầy đủ. "
            "Vui lòng hoàn thiện `src/task10_generation.py` để nhận câu trả lời thật.\n\n"
            f"> *[Lý do: {reason}]*"
        ),
    )

    return {
        "answer": answer,
        "sources": [
            {
                "content": "Bộ luật Hình sự 2015, sửa đổi 2017 — Chương XX: Các tội phạm về ma túy",
                "metadata": {"source": "BLHS_2015_Chuong_XX.pdf", "type": "legal_doc"},
                "score": 0.92,
            },
            {
                "content": "Luật Phòng, chống ma túy số 73/2021/QH14 — Chương V: Cai nghiện ma túy",
                "metadata": {"source": "Luat_PCMT_2021.pdf", "type": "legal_doc"},
                "score": 0.87,
            },
        ],
        "retrieval_source": f"demo ({reason})",
        "config_used": config.config_name,
    }


def _error_response(error_msg: str, config: RAGConfig) -> ResponseDict:
    """Trả về error response khi pipeline báo lỗi không mong đợi."""
    return {
        "answer": (
            f"⚠️ Đã xảy ra lỗi khi xử lý câu hỏi của bạn.\n\n"
            f"```\n{error_msg}\n```\n\n"
            "Vui lòng thử lại hoặc liên hệ nhóm phát triển."
        ),
        "sources": [],
        "retrieval_source": "error",
        "config_used": config.config_name,
    }
