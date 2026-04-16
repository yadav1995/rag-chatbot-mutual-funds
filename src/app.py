"""
Mutual Fund FAQ Assistant — Streamlit Chat Interface

Implements the UI Layer per RAGArchitecture.md §8:
- Welcome banner with disclaimer
- Clickable example questions
- Chat window with user/assistant message bubbles
- Thread sidebar with past conversations + "New Chat"
- Citation badges below assistant messages
- "Last updated" footer on every response

Run:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.chat.thread_manager import ThreadManager

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="HDFC Mutual Fund RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yadav1995/rag-chatbot-mutual-funds',
        'About': "# Mutual Fund RAG Assistant\nA facts-only AI assistant for HDFC Mutual Fund schemes."
    }
)

# =============================================================================
# Custom CSS — Premium dark theme with glassmorphism
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
}
    
/* ── Sidebar ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(233, 69, 96, 0.2);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e94560 !important;
}

/* ── Welcome Banner ────────────────────────── */
.welcome-banner {
    background: linear-gradient(135deg, rgba(233, 69, 96, 0.15), rgba(83, 52, 131, 0.15));
    border: 1px solid rgba(233, 69, 96, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    text-align: center;
}

.welcome-banner h1 {
    background: linear-gradient(135deg, #e94560, #f39c12);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.welcome-banner .disclaimer {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 0.5rem;
}

/* ── Example Question Cards ────────────────── */
.example-container {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    justify-content: center;
}

/* ── Chat Messages & Data Tables ───────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
    animation: fadeIn 0.4s ease-out;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    padding: 1.5rem !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stTable"], .dataframe {
    border-collapse: collapse;
    margin: 1rem 0;
    border-radius: 8px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
[data-testid="stTable"] th, .dataframe th {
    background: rgba(233, 69, 96, 0.15) !important;
    color: #e94560 !important;
    font-weight: 600;
    padding: 10px !important;
}
[data-testid="stTable"] td, .dataframe td {
    padding: 10px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}
[data-testid="stTable"] tr:hover, .dataframe tr:hover {
    background: rgba(255, 255, 255, 0.07) !important;
}

/* ── Citation Badge ────────────────────────── */
.citation-badge {
    display: inline-block;
    background: rgba(233, 69, 96, 0.15);
    border: 1px solid rgba(233, 69, 96, 0.4);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    margin-top: 0.5rem;
    margin-right: 0.5rem;
    font-size: 0.8rem;
    color: #e94560;
    text-decoration: none;
    transition: all 0.2s ease;
}

.citation-badge:hover {
    background: rgba(233, 69, 96, 0.3);
    color: #f39c12;
    transform: scale(1.03);
}

.citation-badge::before {
    content: "🔗 ";
}

/* ── Thread Card ───────────────────────────── */
.thread-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.thread-card:hover {
    background: rgba(233, 69, 96, 0.1);
    border-color: rgba(233, 69, 96, 0.3);
}

.thread-card .title {
    font-size: 0.85rem;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.85);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.thread-card .date {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.4);
}

/* ── Status Indicator ──────────────────────── */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7); }
    70% { box-shadow: 0 0 0 6px rgba(39, 174, 96, 0); }
    100% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0); }
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    background: #27ae60;
    box-shadow: 0 0 6px rgba(39, 174, 96, 0.5);
    animation: pulse 2s infinite;
}

/* ── Info Footer ───────────────────────────── */
.info-footer {
    text-align: center;
    padding: 1rem;
    color: rgba(255, 255, 255, 0.3);
    font-size: 0.75rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 2rem;
}

/* ── Button Styling ────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* ── Hide Streamlit defaults ───────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if "thread_manager" not in st.session_state:
        st.session_state.thread_manager = ThreadManager()

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(use_reranker=False)

    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Mid-Cap Fund?",
    "What is the minimum SIP amount for HDFC ELSS?",
    "Who is the fund manager of HDFC Mid-Cap Fund?",
]


def create_new_thread():
    """Create a new conversation thread."""
    tm = st.session_state.thread_manager
    thread = tm.create_thread()
    st.session_state.current_thread_id = thread.thread_id
    st.session_state.messages = []


def load_thread(thread_id: str):
    """Load an existing thread's messages."""
    tm = st.session_state.thread_manager
    thread = tm.get_thread(thread_id)
    if thread:
        st.session_state.current_thread_id = thread.thread_id
        st.session_state.messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "citations": msg.citations,
            }
            for msg in thread.messages
        ]


def process_query(query: str):
    """Process a user query through the RAG pipeline."""
    tm = st.session_state.thread_manager
    pipeline = st.session_state.pipeline

    # Ensure we have an active thread
    if not st.session_state.current_thread_id:
        create_new_thread()

    thread_id = st.session_state.current_thread_id

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query, "citations": []})
    tm.add_message(thread_id, "user", query)

    # Eagerly show the user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Get conversation history for context
    history = tm.get_recent_history(thread_id, max_pairs=3)

    # Run pipeline with "Thinking" state
    with st.chat_message("assistant", avatar="📊"):
        with st.status("Thinking...", expanded=True) as status:
            status.update(label="Analyzing query intent & searching facts...", state="running")
            response = pipeline.answer(
                query=query,
                thread_id=thread_id,
                conversation_history=history,
            )
            status.update(label="Response generated!", state="complete")
        
        # Guardrail notifications
        if not response.guardrail_passed and response.guardrail_violations:
            violations = ", ".join(response.guardrail_violations)
            st.toast(f"⚠️ Guardrail Triggered: {violations}", icon="⚠️")
        
        st.markdown(response.answer)
        if response.citations:
            with st.expander("📚 View Data Sources"):
                for url in response.citations:
                    domain = url.split("/")[2] if len(url.split("/")) > 2 else url
                    st.markdown(
                        f'<a href="{url}" target="_blank" class="citation-badge">{domain}</a>',
                        unsafe_allow_html=True,
                    )

    # Add assistant message
    assistant_msg = {
        "role": "assistant",
        "content": response.answer,
        "citations": response.citations,
        "intent": response.intent,
        "guardrail_passed": response.guardrail_passed,
        "guardrail_violations": response.guardrail_violations,
        "scrape_date": response.scrape_date,
    }
    st.session_state.messages.append(assistant_msg)
    tm.add_message(thread_id, "assistant", response.answer, citations=response.citations)


# =============================================================================
# Sidebar — Thread Management
# =============================================================================

with st.sidebar:
    st.markdown("### 📊 MF FAQ Assistant")
    st.markdown(
        '<span class="status-dot"></span> <small style="color: rgba(255,255,255,0.6);">System Ready</small>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_thread()
        st.rerun()

    st.markdown("#### 💬 Conversations")

    # List existing threads
    tm = st.session_state.thread_manager
    threads = tm.list_threads(limit=15)

    if not threads:
        st.caption("No conversations yet. Start a new chat!")
    else:
        for t in threads:
            is_active = t["thread_id"] == st.session_state.current_thread_id
            label = t.get("title", "New Chat")[:40]

            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"{'▸ ' if is_active else ''}{label}",
                    key=f"thread_{t['thread_id']}",
                    use_container_width=True,
                ):
                    load_thread(t["thread_id"])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{t['thread_id']}"):
                    tm.delete_thread(t["thread_id"])
                    if st.session_state.current_thread_id == t["thread_id"]:
                        st.session_state.current_thread_id = None
                        st.session_state.messages = []
                    st.rerun()

    # Sidebar footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.7rem; color: rgba(255,255,255,0.3);">
        Facts-only RAG System<br>
        15 HDFC Schemes from Groww<br>
        <br>
        ⚠️ Not financial advice
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Main Content Area
# =============================================================================

# Show welcome banner when no messages
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-banner">
            <h1>📊 Mutual Fund FAQ Assistant</h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 1rem; margin-bottom: 0.5rem;">
                Get instant, fact-based answers about 15 HDFC Mutual Fund schemes
            </p>
            <p class="disclaimer">
                ⚠️ Facts-only — No investment advice. Data sourced from Groww.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Example questions
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, (col, question) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
        with col:
            if st.button(
                question,
                key=f"example_{i}",
                use_container_width=True,
            ):
                process_query(question)
                st.rerun()


# =============================================================================
# Chat Messages
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📊"):
        st.markdown(msg["content"])

        # Show citations in expander
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 View Data Sources"):
                for url in msg["citations"]:
                    domain = url.split("/")[2] if len(url.split("/")) > 2 else url
                    st.markdown(
                        f'<a href="{url}" target="_blank" class="citation-badge">{domain}</a>',
                        unsafe_allow_html=True,
                    )


# =============================================================================
# Chat Input
# =============================================================================

if prompt := st.chat_input("Ask about HDFC mutual fund schemes..."):
    process_query(prompt)
    st.rerun()


# =============================================================================
# Footer
# =============================================================================

if st.session_state.messages:
    scrape_date = st.session_state.pipeline._scrape_date
    st.markdown(
        f"""
        <div class="info-footer">
            Data sourced from Groww.in • Last scrape: {scrape_date} • 
            15 HDFC schemes indexed • 
            Powered by all-MiniLM-L6-v2 + GPT-4o-mini
        </div>
        """,
        unsafe_allow_html=True,
    )
