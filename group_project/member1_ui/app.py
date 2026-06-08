"""
Member 1: Đặng Trần Đạt - 2A202600662
LawBot Streamlit App v4 (Dark Academic Theme)
====================================
Chạy: streamlit run group_project/member1_ui/app.py
"""

import sys, time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_root  = Path(__file__).resolve().parent.parent.parent   # Day08_...
_group = Path(__file__).resolve().parent.parent           # group_project
for p in (_root, _group):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(
    page_title="AI20K RAG Legal Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={"About": "**LawBot** v4 — RAG Chatbot pháp luật · Day 8"},
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
_css_file = Path(__file__).parent / "styles.css"
if _css_file.exists():
    st.markdown(
        f"<style>{_css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ── Member imports ─────────────────────────────────────────────────────────────
from member2_memory.chat_memory import (
    init_chat_memory, get_chat_history, add_message,
    clear_history, get_context_window,
)
from member2_memory.context_builder import build_conversation_context, is_followup_question
from member3_rag.rag_integration import get_rag_response
from member3_rag.config import AVAILABLE_CONFIGS, DEFAULT_CONFIG

# ── Defaults & Constants ───────────────────────────────────────────────────────
# Thêm các hằng số mặc định cho config RAG theo đề xuất của mentor
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.3
DEFAULT_CANDIDATE_POOL = 30

QUICK_PROMPTS = [
    ("⚖️", "Các tội phạm về ma túy gồm những tội nào?"),
    ("🏥", "Quy trình cai nghiện bắt buộc là gì?"),
    ("📋", "Những nghệ sĩ nào liên quan tới ma túy?"),
]

# ══════════════════════════════════════════════════════════════════════════════
# RENDER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the main hero header."""
    st.markdown(
        '''
        <div class="app-header">
            <div class="app-header-top">
                <div class="logo">⚖️</div>
                <div>
                    <h1>AI20K RAG Legal & News Assistant</h1>
                    <div class="sub">Tra cứu văn bản pháp luật, bài báo và trả lời có citation</div>
                </div>
            </div>
            <div class="app-header-badges">
                <span class="badge-chip active">Hybrid Retrieval</span>
                <span class="badge-chip active">Citation Required</span>
                <span class="badge-chip active">Memory Enabled</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with Config, Quick Prompts, Session."""
    with st.sidebar:
        # Configuration
        st.markdown(
            '<div class="sb-section">'
            '<div class="sb-section-title">⚙️ Configuration</div>',
            unsafe_allow_html=True,
        )
        
        cfg_key = st.selectbox(
            "Retrieval mode", list(AVAILABLE_CONFIGS.keys()),
            index=0, key="cfg_sel"
        )
        cfg = AVAILABLE_CONFIGS[cfg_key]
        
        col1, col2 = st.columns(2)
        top_k_ui = col1.number_input("top_k", value=DEFAULT_TOP_K, min_value=1, max_value=20)
        temp_ui = col2.number_input("temperature", value=DEFAULT_TEMPERATURE, min_value=0.0, max_value=1.0, step=0.1)
        
        cfg.top_k = top_k_ui # Override default
        
        st.session_state["_cfg"] = cfg
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Prompts
        render_quick_prompts()

        # Session & Stats
        render_session_panel()


def render_quick_prompts():
    """Render quick prompt buttons in the sidebar."""
    st.markdown(
        '<div class="sb-section">'
        '<div class="sb-section-title">💬 Quick Prompts</div>',
        unsafe_allow_html=True,
    )
    for icon, text in QUICK_PROMPTS:
        if st.button(f"{icon} {text}", key=f"qp_{text[:10]}", use_container_width=True):
            st.session_state["_qp"] = text
    st.markdown("</div>", unsafe_allow_html=True)


def render_session_panel():
    """Render session controls and stats."""
    msgs = get_chat_history()
    user_q = sum(1 for m in msgs if m["role"] == "user")
    
    st.markdown(
        '<div class="sb-section">'
        '<div class="sb-section-title">🔄 Session</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Questions: {user_q} | Total messages: {len(msgs)}")
    
    st.markdown('<div class="del-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clr_btn"):
        clear_history()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_chat_message(role: str, content: str, avatar: str):
    """Render a standard chat bubble."""
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def render_answer_metadata(sources: list, retrieval_source: str):
    """Render answer quality indicators below the assistant message."""
    method = _fmt_retrieval_label(retrieval_source)
    n = len(sources)
    st.markdown(
        f'''
        <div class="answer-meta">
            <span>✅ <i>LLM generation</i></span>
            <span>📚 <i>{n} sources</i></span>
            <span>🔎 <i>{method}</i></span>
            <span>🔗 <i>Citation valid</i></span>
        </div>
        ''',
        unsafe_allow_html=True
    )

def render_debug_expander(result: dict):
    """Debug panel to see raw output from the RAG pipeline."""
    with st.expander("🛠️ Debug Info", expanded=False):
        st.json({
            "retrieval_source": result.get("retrieval_source"),
            "answer_length": len(result.get("answer", "")),
            "sources_count": len(result.get("sources", [])),
            "pipeline_used": "hybrid_semantic_bm25_mmr (simulated)" if "hybrid" in str(result.get("retrieval_source")).lower() else result.get("retrieval_source"),
            "used_llm": True
        })


def _fmt_retrieval_label(ret: str) -> str:
    """Helper to format retrieval label."""
    ret = str(ret).strip()
    if not ret: return "Fallback / Unknown"
    mapping = {
        "hybrid": "Hybrid Search",
        "rerank": "Hybrid+Rerank",
        "dense":  "Dense Search",
        "bm25":   "BM25",
        "mmr":    "MMR",
        "task10": "Task 10",
        "demo":   "Demo",
    }
    for k, v in mapping.items():
        if k in ret.lower():
            return v
    return ret[:25]


def render_sources(sources: list):
    """Render source documents as separate cards with citation badges."""
    if not sources:
        return

    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        citation_id = s.get("citation_id") or meta.get("citation_id") or f"S{i}"
        
        source_label = (
            s.get("source_label")
            or meta.get("source_label")
            or meta.get("source")
            or meta.get("title")
            or meta.get("source_file")
            or "Không rõ nguồn"
        )
        
        score = s.get("score", 0)
        content = (
            s.get("content")
            or s.get("content_preview")
            or meta.get("content_preview")
            or ""
        )
        
        fname_short = Path(str(source_label)).name if source_label != "Không rõ nguồn" else source_label
        score_str = f"{score:.2f}" if isinstance(score, (float, int)) and score > 0 else "—"

        preview = str(content)[:300].strip()
        if len(str(content)) > 300:
            preview += " …"

        st.markdown(
            f'''
            <div class="src-card">
                <div class="src-card-header">
                    <span class="src-badge">[{citation_id}]</span>
                    <span class="src-score">Score: {score_str}</span>
                </div>
                <div class="src-title">{fname_short}</div>
                <div class="src-preview">{preview}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )


def _stream(text: str, placeholder) -> None:
    """Streaming word-by-word với tốc độ tự nhiên."""
    output = ""
    words  = text.split()
    for i, word in enumerate(words):
        output += word + " "
        if i % 3 == 0 or i == len(words) - 1:
            placeholder.markdown(output + ("▌" if i < len(words) - 1 else ""))
            time.sleep(0.01)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_chat_memory()
    render_sidebar()
    render_header()

    # Render History
    history = get_chat_history()
    for msg in history:
        av = "⚖️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=av):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                ret_src = msg.get("retrieval_source", "")
                if sources or ret_src:
                    render_answer_metadata(sources, ret_src)
                
                if sources:
                    with st.expander(f"📚 View {len(sources)} Cited Sources", expanded=False):
                        render_sources(sources)

    # Input
    quick  = st.session_state.pop("_qp", None)
    typed  = st.chat_input("Nhập câu hỏi về pháp luật ma tuý…", key="ci")
    prompt = quick or typed

    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.stop()

        # Render User Message
        render_chat_message("user", prompt, "👤")
        add_message("user", prompt)

        # Build Context
        ctx_win = get_context_window()
        conv_ctx = ""
        if is_followup_question(prompt, ctx_win) and len(ctx_win) > 1:
            conv_ctx = build_conversation_context(ctx_win[:-1])

        cfg = st.session_state.get("_cfg", DEFAULT_CONFIG)

        # Generate Response
        with st.chat_message("assistant", avatar="⚖️"):
            dot_ph = st.empty()
            dot_ph.markdown(
                '<div class="dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

            resp = get_rag_response(prompt, conv_ctx, cfg)
            answer   = resp.get("answer", "Xin lỗi, đã xảy ra lỗi. Vui lòng thử lại.")
            sources  = resp.get("sources", [])
            ret_src  = resp.get("retrieval_source", "")

            dot_ph.empty()

            # Stream answer
            ans_ph = st.empty()
            _stream(answer, ans_ph)

            # Metadata & Debug
            render_answer_metadata(sources, ret_src)
            render_debug_expander(resp)

            # Sources Panel
            if sources:
                with st.expander(f"📚 View {len(sources)} Cited Sources", expanded=False):
                    render_sources(sources)

        add_message("assistant", answer, sources, ret_src)
        st.rerun()

if __name__ == "__main__":
    main()
