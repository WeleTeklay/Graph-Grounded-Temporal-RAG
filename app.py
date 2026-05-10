"""
LexTemporal AI - Professional Legal Intelligence Platform
Graph-Grounded Temporal RAG with Contradiction-Resilient QA
Production-Grade Streamlit Application
"""

import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import time
import uuid
import json
from dataclasses import dataclass, field

from src.config import config
from src.logger import get_logger
from src.optimized_retriever import OptimizedQueryEngine

logger = get_logger(__name__)

# ============================================================================
# PAGE CONFIGURATION - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(
    page_title="LexTemporal AI | Legal Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/WeleTeklay/Graph-Grounded-Temporal-RAG',
        'About': '''
        # LexTemporal AI
        
        **Graph-Grounded Temporal RAG for Legal Documents**
        
        Production-grade legal intelligence platform that resolves 
        contradictions across evolving documents using graph-based 
        temporal reasoning.
        
        **Core Capabilities:**
        - Hybrid Search (Vector + BM25)
        - Cross-Encoder Reranking
        - Neo4j Graph Integration
        - Temporal Contradiction Resolution
        - Local LLM (Llama 3.2)
        
        **Version:** 2.0.0
        **License:** MIT
        '''
    }
)

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class Message:
    role: str
    content: str
    sources: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: int = 0


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables - SINGLE SOURCE OF TRUTH"""
    
    defaults = {
        "messages": [],
        "engine": None,
        "manifest_df": pd.DataFrame(),
        "show_timeline": False,
        "target_date": None,
        "use_specific_date": False,
        "show_sources": True,
        "show_metrics": True,
        "is_generating": False,
        "stop_generation": False,
        "editing_message_id": None,
        "regenerate_trigger": False,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    if st.session_state.engine is None:
        with st.spinner(" Initializing LexTemporal AI..."):
            st.session_state.engine = OptimizedQueryEngine()
    
    if st.session_state.manifest_df.empty:
        manifest_path = config.paths.project_root / "document_manifest.csv"
        if manifest_path.exists():
            st.session_state.manifest_df = pd.read_csv(manifest_path)


# ============================================================================
# PROCESS UPLOAD FUNCTION (ADDED)
# ============================================================================
def process_upload(file, title: str, effective_date: str, supersedes: Optional[str]):
    """Process uploaded document - fully automatic pipeline"""
    from src.ingest import ingest_single_document
    
    with st.spinner("Processing document..."):
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        
        # Save file with original extension
        original_ext = Path(file.name).suffix.lower()
        file_path = config.paths.raw_pdfs_dir / f"{doc_id}{original_ext}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Update manifest
        manifest_path = config.paths.project_root / "document_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
        else:
            manifest = pd.DataFrame(columns=['doc_id', 'doc_title', 'effective_date', 'supersedes_doc_id'])
        
        new_row = pd.DataFrame([{
            'doc_id': doc_id,
            'doc_title': title,
            'effective_date': effective_date,
            'supersedes_doc_id': supersedes if supersedes != "None" else None
        }])
        
        manifest = pd.concat([manifest, new_row], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        st.session_state.manifest_df = manifest
        
        # Run the ingestion pipeline
        success = ingest_single_document(doc_id, effective_date)
        
        if success:
            # Refresh the engine to include new document
            st.session_state.engine = OptimizedQueryEngine()
            st.success(f" Document '{title}' processed successfully! The system is now updated.")
            time.sleep(1.5)
            st.rerun()
        else:
            st.error("Failed to process document")


# Call initialization
init_session_state()


# ============================================================================
# CUSTOM CSS - PROFESSIONAL STYLING WITH FIXES
# ============================================================================
st.markdown("""
<style>
    /* ===== FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== MAIN CONTAINER ===== */
    .main > div {
        padding: 1rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* ===== HERO HEADER ===== */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
        border-radius: 24px;
        padding: 2rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .hero-badge {
        background: rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 40px;
        color: #e2e8f0;
        font-size: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* ===== SIDEBAR STYLES - FIXED VISIBILITY ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    /* Fix for text inputs in sidebar - MAKE TEXT VISIBLE */
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Fix for date inputs */
    section[data-testid="stSidebar"] .stDateInput input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Fix for select boxes */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border-color: #475569 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
        color: #f1f5f9 !important;
    }
    
    /* Fix for file uploader */
    section[data-testid="stSidebar"] .stFileUploader > div {
        background-color: #1e293b !important;
        border: 1px dashed #475569 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader > div:hover {
        border-color: #6366f1 !important;
        background-color: #1a2740 !important;
    }
    
    /* Fix for expander headers */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 0 0 8px 8px !important;
        padding: 15px !important;
    }
    
    /* Fix for button in sidebar */
    section[data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Fix for labels in sidebar */
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        margin-bottom: 4px !important;
        display: block !important;
    }
    
    /* ===== CHAT MESSAGES ===== */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 20px 20px 6px 20px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 10px;
    }
    
    .assistant-message {
        background: #1e293b;
        color: #f1f5f9;
        padding: 14px 20px;
        border-radius: 20px 20px 20px 6px;
        max-width: 85%;
        margin-bottom: 10px;
        border: 1px solid #334155;
    }
    
    /* ===== SOURCE CARDS ===== */
    .source-card {
        background: #0f172a;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #334155;
    }
    
    .source-card:hover {
        border-color: #6366f1;
    }
    
    .source-title {
        font-weight: 700;
        color: #f1f5f9;
    }
    
    .badge-current {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
    }
    
    .badge-historical {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
    }
    
    .source-meta {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #334155;
    }
    
    /* ===== METRICS CARDS ===== */
    .metric-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .metric-icon {
        font-size: 1.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f1f5f9;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    
    /* ===== WELCOME SCREEN ===== */
    .welcome-container {
        text-align: center;
        padding: 40px 30px;
        background: #1e293b;
        border-radius: 24px;
        border: 1px solid #334155;
        margin: 20px 0;
    }
    
    .welcome-icon {
        font-size: 3rem;
    }
    
    .welcome-title {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .welcome-text {
        color: #94a3b8;
        font-size: 0.9rem;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .example-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    
    .example-card:hover {
        border-color: #6366f1;
    }
    
    /* ===== INPUT STYLING ===== */
    .stChatInput > div {
        border-radius: 40px !important;
        border: 2px solid #334155 !important;
        background: #1e293b !important;
    }
    
    .stChatInput input {
        color: #f1f5f9 !important;
    }
    
    .stChatInput input::placeholder {
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title"> LexTemporal AI</div>
        <div class="hero-subtitle">Graph-Grounded Temporal RAG for Contradiction-Resilient Legal Intelligence</div>
        <div class="hero-badges">
            <span class="hero-badge"> Graph-Grounded</span>
            <span class="hero-badge"> Temporal-Aware</span>
            <span class="hero-badge"> Hybrid Search</span>
            <span class="hero-badge"> Cross-Encoder Reranking</span>
            <span class="hero-badge"> Llama 3.2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    engine = st.session_state.engine
    manifest = st.session_state.manifest_df
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"></div>
            <div class="metric-value">{len(manifest)}</div>
            <div class="metric-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"></div>
            <div class="metric-value">{engine.collection.count()}</div>
            <div class="metric-label">Vector Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = " Connected" if engine.graph_enabled else "Limited"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon"></div>
            <div class="metric-value">Neo4j</div>
            <div class="metric-label">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon"></div>
            <div class="metric-value">Llama 3.2</div>
            <div class="metric-label">3B Parameters</div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h1 style="color: #f1f5f9; font-size: 1.5rem;"> LexTemporal</h1>
            <p style="color: #94a3b8; font-size: 0.8rem;">v2.0.0 • Production</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Upload Section
        st.markdown("### Upload New Document")
        
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['pdf', 'docx', 'txt', 'md', 'html'],
            help="Supported formats: PDF, DOCX, TXT, MD, HTML",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            with st.expander(" Document Details", expanded=True):
                doc_title = st.text_input(
                    "Title", 
                    value=uploaded_file.name.rsplit('.', 1)[0],
                    key="doc_title_input"
                )
                effective_date = st.date_input(
                    "Effective Date", 
                    value=date.today(),
                    help="Select the actual effective date of this document (e.g., 2021-01-01 for old documents)"
                )
                if effective_date < date.today():
                   st.info(f" Historical document effective {effective_date}. It will supersede documents before this date.")
                
                manifest = st.session_state.get("manifest_df", pd.DataFrame())
                doc_options = ["None"] + manifest['doc_id'].tolist() if not manifest.empty else ["None"]
                supersedes = st.selectbox(
                    "Supersedes Document", 
                    options=doc_options,
                    key="doc_supersedes_select"
                )
                
                if st.button(" Process Document", type="primary", use_container_width=True):
                    process_upload(
                        uploaded_file, 
                        doc_title, 
                        str(effective_date),
                        None if supersedes == "None" else supersedes
                    )
        
        st.divider()
        
        # Temporal Controls
        st.markdown("###  Temporal Context")
        
        use_specific = st.checkbox(
            "Use specific effective date",
            value=st.session_state.use_specific_date
        )
        st.session_state.use_specific_date = use_specific
        
        if use_specific:
            target_date = st.date_input("Effective Date", value=date.today())
            st.session_state.target_date = str(target_date)
            st.info(f"Showing documents effective ≤ {target_date}")
        else:
            st.session_state.target_date = None
            st.success("Showing current/latest versions")
        
        st.divider()
        
        # Display Options
        st.markdown("### Display Options")
        st.session_state.show_sources = st.checkbox("Show source citations", value=st.session_state.show_sources)
        st.session_state.show_metrics = st.checkbox("Show metrics dashboard", value=st.session_state.show_metrics)
        
        st.divider()
        
        # System Stats
        st.markdown("###  System Stats")
        engine = st.session_state.engine
        if engine:
            st.metric("Vector DB", f"{engine.collection.count():,} chunks")
            st.metric("Graph DB", "Connected" if engine.graph_enabled else "Disabled")
        
        st.divider()
        
        if st.button(" Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.markdown("""
        <div style="text-align: center; padding: 15px 0; color: #64748b; font-size: 0.7rem;">
            © 2026 LexTemporal AI<br/>
            Graph-Grounded Temporal RAG
        </div>
        """, unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon"></div>
        <div class="welcome-title">Welcome to LexTemporal AI</div>
        <div class="welcome-text">
            Ask questions about your legal documents. I provide accurate, 
            contradiction-resilient answers using graph-based temporal reasoning.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("####  Try these examples:")
    
    examples = [
        ("", "Current State", "What is the effective date of the original agreement?"),
        ("", "Temporal Query", "What was the penalty fee in 2021?"),
        ("", "Contradiction Test", "Has the penalty clause changed over time?"),
        ("", "Specific Clause", "What does Section 1.1 state?"),
        ("", "Governing Law", "What is the governing law?"),
        ("", "Parties", "Who are the parties involved?"),
    ]
    
    cols = st.columns(3)
    for i, (icon, title, question) in enumerate(examples):
        with cols[i % 3]:
            if st.button(f"{icon} {title}", key=f"ex_{i}", use_container_width=True):
                msg = Message(role="user", content=question)
                st.session_state.messages.append(msg)
                st.rerun()
            st.caption(question)


def render_chat_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            if msg.role == "user":
                st.markdown(msg.content)
            else:
                st.markdown(msg.content)
                
                if st.session_state.show_sources and msg.sources:
                    with st.expander(" View Sources", expanded=False):
                        for src in msg.sources[:5]:
                            is_current = src.get('is_current', True)
                            badge = "CURRENT" if is_current else "HISTORICAL"
                            st.markdown(f"""
                            ** {src.get('doc_id', 'Unknown')}** ({badge})
                            - Effective: {src.get('effective_date', 'N/A')}
                            - Score: {src.get('score', 0):.3f}
                            """)
                
                if msg.response_time_ms > 0:
                    st.caption(f" {msg.response_time_ms / 1000:.2f} seconds")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def add_message(role: str, content: str, sources: List[Dict] = None, response_time_ms: int = 0):
    msg = Message(
        role=role,
        content=content,
        sources=sources or [],
        response_time_ms=response_time_ms
    )
    st.session_state.messages.append(msg)
    return msg


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    render_hero()
    render_sidebar()
    
    if st.session_state.show_metrics:
        render_metrics()
    
    with st.container():
        if not st.session_state.messages:
            render_welcome()
        else:
            render_chat_messages()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    prompt = st.chat_input(
        "Ask anything about your legal documents...",
        key="chat_input",
        disabled=st.session_state.get("is_generating", False)
    )
    
    if prompt:
        add_message(role="user", content=prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown(" Thinking...")
            
            try:
                start_time = time.perf_counter()
                
                result = st.session_state.engine.answer(
                    prompt,
                    target_date=st.session_state.get("target_date")
                )
                
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                answer = result.get('answer', 'No answer generated.')
                sources = result.get('sources', [])
                
                response_placeholder.markdown(answer)
                st.caption(f"⚡ {elapsed_ms / 1000:.2f} seconds")
                
                if st.session_state.show_sources and sources:
                    with st.expander(" View Sources", expanded=False):
                        for src in sources[:5]:
                            is_current = src.get('is_current', True)
                            badge = "CURRENT" if is_current else "HISTORICAL"
                            st.markdown(f"""
                            ** {src.get('doc_id', 'Unknown')}** ({badge})
                            - Effective: {src.get('effective_date', 'N/A')}
                            - Score: {src.get('score', 0):.3f}
                            """)
                
                add_message(role="assistant", content=answer, sources=sources, response_time_ms=elapsed_ms)
                
            except Exception as e:
                response_placeholder.error(f" Error: {str(e)}")
                logger.error(f"Query failed: {e}")
        
        st.rerun()


if __name__ == "__main__":
    main()