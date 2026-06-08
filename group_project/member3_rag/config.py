"""
Member 3: Phạm Quang Dũng - 2A202600703
RAG Configuration
==============================
Cấu hình trung tâm cho retrieval pipeline.

Nhiệm vụ: Định nghĩa các tham số retrieval dưới dạng dataclass,
          dễ dàng thay đổi khi A/B test.
"""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """
    Cấu hình cho RAG pipeline.

    Attributes:
        top_k: Số lượng documents trả về
        alpha: Trọng số semantic vs lexical (1.0 = pure semantic, 0.0 = pure BM25)
        use_reranking: Có dùng reranking không
        rerank_top_k: Số lượng docs sau reranking
        use_pageindex_fallback: Có fallback sang PageIndex không
        model_name: Tên LLM model dùng để generate
        max_context_messages: Số tin nhắn lịch sử truyền vào prompt
        temperature: Temperature của LLM
        config_name: Tên config (dùng cho A/B test)
    """
    top_k: int = 5
    alpha: float = 0.5
    use_reranking: bool = True
    rerank_top_k: int = 3
    use_pageindex_fallback: bool = True
    model_name: str = "gpt-4o-mini"
    max_context_messages: int = 6
    temperature: float = 0.3
    config_name: str = "default"


# ─── Preset Configs (dùng cho A/B Comparison) ────────────────────────────────

DEFAULT_CONFIG = RAGConfig(
    top_k=5,
    alpha=0.5,
    use_reranking=True,
    rerank_top_k=3,
    config_name="hybrid_rerank",  # Config A
)

DENSE_ONLY_CONFIG = RAGConfig(
    top_k=5,
    alpha=1.0,         # Pure semantic search
    use_reranking=False,
    rerank_top_k=5,
    config_name="dense_only",  # Config B
)

AGGRESSIVE_CONFIG = RAGConfig(
    top_k=8,
    alpha=0.3,         # Lean toward BM25
    use_reranking=True,
    rerank_top_k=5,
    config_name="hybrid_aggressive",  # Config C
)

# Map tên config để dễ chọn từ UI
AVAILABLE_CONFIGS = {
    "🔀 Hybrid + Reranking (Mặc định)": DEFAULT_CONFIG,
    "🧠 Dense Only (Semantic)": DENSE_ONLY_CONFIG,
    "⚡ Hybrid Aggressive (BM25-heavy)": AGGRESSIVE_CONFIG,
}
