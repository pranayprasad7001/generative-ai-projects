# -*- coding: utf-8 -*-
"""Streamlit UI for Agentic RAG System - Decoupled Ingestion & Instant Search"""

import streamlit as st
from pathlib import Path
import sys
import time
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("rag.log"),
        logging.StreamHandler()
    ]
)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.llmgateway_config import Config
from document_ingestion.document_processor import DocumentProcessor
from document_ingestion.chunker import ChunkStrategy
from vectorstore.vectorstore import VectorStoreManager
from graph_builder.adaptive_graph_builder import GraphBuilder

# LangSmith
LANGSMITH_TRACING = Config.LANGSMITH_TRACING
LANGSMITH_API_KEY = Config.LANGSMITH_API_KEY
LANGSMITH_PROJECT = Config.LANGSMITH_PROJECT


def format_cost(cost: float) -> str:
    """Format float cost to user-readable string without scientific notation."""
    if cost == 0.0:
        return "$0.00"
    val = f"{cost:.7f}"
    while val.endswith("0") and len(val.split(".")[1]) > 2:
        val = val[:-1]
    return f"${val}"


# Page configuration
st.set_page_config(
    page_title="Enhanced Multi-Document Search RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title with indigo-purple gradient */
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Primary buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    }
    
    /* Search button styling */
    .search-btn button {
        background-color: #10b981 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: bold !important;
        transition: background-color 0.2s ease !important;
    }
    .search-btn button:hover {
        background-color: #059669 !important;
    }
    
    /* Answer Card */
    .answer-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #10b981;
        line-height: 1.6;
    }
    
    /* Source text cards */
    .source-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #059669;
        color: #34d399;
        padding: 0.35rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Headers */
    .sidebar-header {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 0.35rem;
    }
    
    /* Sidebar Info Box */
    .sidebar-info-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 0.85rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    /* Features Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.25rem;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
        background-color: rgba(30, 41, 59, 0.6);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    .feature-card h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 1.05rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .feature-card ul {
        margin: 0;
        padding-left: 1.2rem;
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    .feature-card strong {
        color: #f1f5f9;
    }
    .feature-card li {
        margin-bottom: 0.45rem;
        line-height: 1.4;
    }
    .feature-card li:last-child {
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state and auto-connect to existing Astra DB vector store."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'num_chunks_session' not in st.session_state:
        st.session_state.num_chunks_session = 0
    if 'db_error' not in st.session_state:
        st.session_state.db_error = None
        
    # Auto-initialize VectorStore & GraphBuilder if not already loaded
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        try:
            vector_store = VectorStoreManager()
            st.session_state.vector_store = vector_store
            
            llm_generator = Config.get_llm_generator()
            llm_checker = Config.get_llm_checker()
            retriever = vector_store.get_retriever(search_type="similarity")
            
            graph_builder = GraphBuilder(
                retriever=retriever,
                llm_generator=llm_generator,
                llm_checker=llm_checker
            )
            graph_builder.build_graph()
            
            st.session_state.rag_system = graph_builder
            st.session_state.initialized = True
            st.session_state.db_error = None
        except Exception as e:
            logging.exception("Failed to auto-connect to Astra DB on startup")
            st.session_state.vector_store = None
            st.session_state.rag_system = None
            st.session_state.initialized = False
            st.session_state.db_error = str(e)


def process_ingestion(urls_input, uploaded_files, strategy_name, chunk_size, chunk_overlap):
    """Process files and URLs, append to the vector store, and refresh the active retriever."""
    try:
        temp_dir = Path("data/uploaded")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        sources = []
        if urls_input:
            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
            sources.extend(urls)
            
        saved_files = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                sources.append(str(file_path))
                saved_files.append(file_path)
                
        if not sources:
            st.error("⚠️ Please specify at least one URL or upload a file.")
            return False

        if st.session_state.vector_store is None:
            vector_store = VectorStoreManager()
            st.session_state.vector_store = vector_store
        else:
            vector_store = st.session_state.vector_store
        
        embeddings = None
        if strategy_name in ["Semantic", "Hybrid"]:
            embeddings = vector_store.embeddings
            strategy = ChunkStrategy.SEMANTIC if strategy_name == "Semantic" else ChunkStrategy.HYBRID
        else:
            strategy = ChunkStrategy.RECURSIVE

        doc_processor = DocumentProcessor(
            embeddings=embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        with st.status("🛠️ Ingesting & Indexing documents...", expanded=True) as status:
            status.update(label="📄 Extracting text from files and URLs...", state="running")
            documents = doc_processor.load_documents(sources, strategy=strategy)
            
            if not documents:
                status.update(label="❌ Ingestion failed: No text content found.", state="error")
                st.error("⚠️ No text content could be extracted from the specified sources.")
                return False
                
            status.update(label=f"💾 Indexing {len(documents)} chunks into Astra DB...", state="running")
            vector_store.create_vectorstore(documents)
            
            status.update(label="⚡ Refreshing retriever in active workflow...", state="running")
            if st.session_state.rag_system is not None:
                st.session_state.rag_system.nodes.retriever = vector_store.get_retriever()
            else:
                llm_generator = Config.get_llm_generator()
                llm_checker = Config.get_llm_checker()
                graph_builder = GraphBuilder(
                    retriever=vector_store.get_retriever(),
                    llm_generator=llm_generator,
                    llm_checker=llm_checker
                )
                graph_builder.build_graph()
                st.session_state.rag_system = graph_builder
                st.session_state.initialized = True
                
            status.update(label="✅ Indexing completed successfully!", state="complete")

        st.session_state.num_chunks_session += len(documents)
        
        # Clean up temp files
        for file_path in saved_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logging.warning("Failed to delete temp file %s: %s", file_path, e)
                
        return True
    except Exception as e:
        st.error(f"❌ Failed to ingest documents: {str(e)}")
        logging.exception("Error during ingestion process")
        return False


def main():
    init_session_state()
    
    # Title Section
    st.markdown('<div class="main-title">🤖 Enhanced Multi-Document Search RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Query indexed documents in Astra DB with adaptive self-correction, or ingest new knowledge anytime.</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📡 System Telemetry</div>', unsafe_allow_html=True)
        
        # Database Status
        if st.session_state.initialized and st.session_state.vector_store is not None:
            st.markdown(
                f"""
                <div class="sidebar-info-box">
                    <strong style="color: #34d399;">🟢 Astra DB Connected</strong><br/>
                    <span style="color: #94a3b8;">Collection:</span> <code>{Config.ASTRA_DB_COLLECTION_NAME}</code><br/>
                    <span style="color: #94a3b8;">Region:</span> <code>{Config.ASTRA_DB_API_REGION or 'us-east1'}</code>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="sidebar-info-box" style="border-color: #ef4444;">
                    <strong style="color: #f87171;">🔴 Not Connected</strong><br/>
                    <span>Check Astra DB credentials & LiteLLM Gateway</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown(
            f"""
            <div class="sidebar-info-box">
                <strong>🤖 Model Gateway Config:</strong><br/>
                • <strong>Generator:</strong> <code>{Config.LLM_MODEL_GENERATOR}</code><br/>
                • <strong>Checker:</strong> <code>{Config.LLM_MODEL_CHECKER}</code><br/>
                • <strong>Embeddings:</strong> <code>{Config.EMBEDDING_MODEL}</code><br/>
                • <strong>Reranker:</strong> <code>{Config.COHERE_RERANKER_MODEL}</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="sidebar-header">⚙️ Search Settings</div>', unsafe_allow_html=True)
        search_type_choice = st.selectbox(
            "Retrieval Search Type:",
            ["Similarity Search", "Maximal Marginal Relevance (MMR)"],
            help="Similarity Search retrieves based on nearest vector. MMR balances relevance and diversity."
        )
        
        top_k = st.slider("Top Documents (k):", min_value=2, max_value=15, value=5, step=1)
        
        st.markdown("---")
        if st.button("🔄 Reconnect Database"):
            st.session_state.vector_store = None
            st.session_state.rag_system = None
            init_session_state()
            st.rerun()
            
        if st.button("🗑️ Clear Search History"):
            st.session_state.history = []
            st.rerun()

    # Main Tabs
    tab_search, tab_ingest, tab_arch = st.tabs([
        "💬 Search & Q&A",
        "📥 Ingest Knowledge",
        "ℹ️ Architecture & System"
    ])
    
    # ==========================================
    # TAB 1: Search & Q&A
    # ==========================================
    with tab_search:
        if not st.session_state.initialized:
            st.error(f"⚠️ Search engine is not initialized. Error: {st.session_state.db_error or 'Could not connect to Astra DB'}")
            st.info("Ensure LiteLLM proxy is running (`uv run litellm --config src/config/litellm_config.yaml --port 4000`) and Astra DB credentials are valid in `.env`.")
        else:
            with st.form("search_form"):
                col_in, col_btn = st.columns([5, 1])
                with col_in:
                    question = st.text_input(
                        "Ask a question against your Astra DB knowledge base:",
                        placeholder="e.g. What is memory in AI agents?"
                    )
                with col_btn:
                    st.markdown('<div class="search-btn" style="margin-top: 1.7rem;">', unsafe_allow_html=True)
                    submit = st.form_submit_button("Search")
                    st.markdown('</div>', unsafe_allow_html=True)

            if submit and question:
                with st.spinner("🔍 Agentic retrieval and self-correcting answer generation in progress..."):
                    start_time = time.time()
                    try:
                        search_type = "mmr" if search_type_choice == "Maximal Marginal Relevance (MMR)" else "similarity"
                        st.session_state.rag_system.nodes.retriever = st.session_state.vector_store.get_retriever(
                            k=top_k,
                            search_type=search_type
                        )
                        
                        result = asyncio.run(st.session_state.rag_system.run(question))
                        elapsed_time = time.time() - start_time
                        cost = result.get('total_cost', 0.0)
                        
                        # Store in search history
                        st.session_state.history.append({
                            'question': question,
                            'answer': result.get('answer', 'No answer generated'),
                            'time': elapsed_time,
                            'cost': cost,
                            'retrieved_docs': result.get('retrieved_docs', []),
                            'external_citations': result.get('external_citations', [])
                        })
                        
                        # Display Answer
                        st.markdown("### 💡 Generated Answer")
                        st.markdown(f'<div class="answer-card">{result.get("answer", "No answer generated")}</div>', unsafe_allow_html=True)
                        
                        # Stats badge
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 1.5rem; margin-top: -0.75rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: #f8fafc; background-color: #1e293b; padding: 0.5rem 1rem; border-radius: 0.375rem; width: fit-content; border: 1px solid #475569;">
                                <span>⏱️ <strong>Latency:</strong> {elapsed_time:.2f}s</span>
                                <span style="color: #64748b;">|</span>
                                <span>💰 <strong>Estimated Cost:</strong> <code style="color: #34d399; font-weight: bold; background: none; padding: 0;">{format_cost(cost)}</code></span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # External citations
                        external_citations = result.get('external_citations', [])
                        if external_citations:
                            links = [
                                f'<a href="{url}" target="_blank" style="color: #6366f1; text-decoration: none; font-weight: 500;">{url.split("//")[-1].split("/")[0].replace("www.", "")}</a>'
                                for url in external_citations
                            ]
                            st.markdown(
                                f"""
                                <div style="font-size: 0.9rem; color: #94a3b8; margin-top: -0.75rem; margin-bottom: 1.5rem;">
                                    🔗 <strong>External Citations:</strong> {" • ".join(links)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Source documents
                        st.markdown("### 📄 Retrieved Source Documents")
                        retrieved_docs = result.get('retrieved_docs', [])
                        if retrieved_docs:
                            for i, doc in enumerate(retrieved_docs, 1):
                                src_name = doc.metadata.get("file_name") or doc.metadata.get("source") or "Unknown source"
                                chunk_num = doc.metadata.get("chunk", "N/A")
                                loader_name = doc.metadata.get("loader", "Unknown")
                                
                                with st.expander(f"Reference {i}: {src_name} (Chunk {chunk_num})"):
                                    st.write(doc.page_content)
                                    st.caption(f"Source details: Loader: {loader_name} | Full path/url: {doc.metadata.get('source')}")
                        else:
                            st.info("No local reference documents found. Model may have answered using external search or general reasoning.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during retrieval/generation: {str(e)}")
                        logging.exception("Search workflow error")
            
            # History panel
            if st.session_state.history:
                st.markdown("---")
                st.markdown("### 📜 Recent Search History")
                for idx, item in enumerate(reversed(st.session_state.history[-5:])):
                    item_cost = item.get('cost', 0.0)
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>Q: {item['question']}</strong><br/>
                            <span style="color: #cbd5e1; font-size: 0.9rem;">Answer: {item['answer'][:250]}...</span><br/>
                            <span style="color: #64748b; font-size: 0.8rem;">⏱️ Latency: {item['time']:.2f}s | 💰 Cost: {format_cost(item_cost)} | 📄 References: {len(item.get('retrieved_docs', []))} | 🔗 Citations: {len(item.get('external_citations', []))}</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

    # ==========================================
    # TAB 2: Ingest Knowledge
    # ==========================================
    with tab_ingest:
        st.markdown("### 📥 Index New Documents or URLs into Astra DB")
        st.markdown("Upload new local files or supply web URLs to expand your existing knowledge base in Astra DB without resetting active queries.")
        
        col_src1, col_src2 = st.columns(2)
        with col_src1:
            st.markdown("#### 🌐 Web URLs")
            default_urls_str = "\n".join(Config.DEFAULT_URLS)
            urls_input = st.text_area(
                "Enter URLs (one per line):",
                value="",
                height=140,
                placeholder="https://example.com/article\nhttps://en.wikipedia.org/wiki/Artificial_intelligence"
            )
            if st.button("Load Default Sample URLs"):
                urls_input = default_urls_str
                
        with col_src2:
            st.markdown("#### 📁 Local Files")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT, MD, CSV, XLSX):",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "md", "csv", "xlsx"]
            )
            
        st.markdown("#### ⚙️ Chunking Strategy Settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            strategy_name = st.selectbox(
                "Chunking Strategy:",
                ["Recursive", "Semantic", "Hybrid"],
                help="Recursive uses character splitting. Semantic uses vector embedding distances. Hybrid combines both."
            )
        with c2:
            chunk_size = st.slider("Chunk Size (characters):", 100, 2000, Config.CHUNK_SIZE, 50)
        with c3:
            chunk_overlap = st.slider("Chunk Overlap (characters):", 0, 500, Config.CHUNK_OVERLAP, 10)
            
        if st.button("🚀 Ingest & Index into Astra DB", use_container_width=True):
            if not urls_input and not uploaded_files:
                st.warning("Please provide at least one URL or upload a file to ingest.")
            else:
                success = process_ingestion(
                    urls_input=urls_input,
                    uploaded_files=uploaded_files,
                    strategy_name=strategy_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                if success:
                    st.success("🎉 Documents successfully indexed into Astra DB! You can now query them in the Search tab.")
                    st.toast("Astra DB index updated successfully!")

    # ==========================================
    # TAB 3: Architecture & Specs
    # ==========================================
    with tab_arch:
        st.markdown("""
        <h3 style="color: #f1f5f9;">Architecture & Features of this Adaptive RAG Engine</h3>
        <div class="features-grid">
            <div class="feature-card">
                <h4 style="color: #818cf8;">📄 Ingestion & Search</h4>
                <ul>
                    <li><strong>Multi-Source:</strong> Loads Web URLs, PDFs, Word, Excel, CSV, TXT, and Markdown.</li>
                    <li><strong>Ensemble Search:</strong> Blends Astra DB vector similarity with BM25 keyword matching.</li>
                    <li><strong>Cohere Reranking:</strong> Refines context relevance using advanced cross-attention models.</li>
                    <li><strong>Smart Chunking:</strong> Supports standard Recursive Character or embedding-based Semantic/Hybrid chunking.</li>
                </ul>
            </div>
            <div class="feature-card">
                <h4 style="color: #c084fc;">🤖 Agentic Self-Correction</h4>
                <ul>
                    <li><strong>LangGraph Workflow:</strong> Replaces rigid pipelines with a flexible StateGraph orchestrator.</li>
                    <li><strong>Double-Loop Correction:</strong> Grades context, rewrites weak queries, and catches hallucinations.</li>
                    <li><strong>Runaway Protection:</strong> Enforces bounded retry loops (max 3) to prevent runaway LLM calls.</li>
                    <li><strong>MCP Escalation:</strong> Seamlessly queries Tavily or Wikipedia MCP servers as search backup.</li>
                </ul>
            </div>
            <div class="feature-card">
                <h4 style="color: #fb7185;">🛡️ Safety & Guardrails</h4>
                <ul>
                    <li><strong>PII Masking:</strong> Automatic masking/redaction of emails, credit cards, phones, and SSNs.</li>
                    <li><strong>Bi-Directional Filtering:</strong> Audits incoming user queries and filters generated LLM responses.</li>
                    <li><strong>Security Graders:</strong> Integrates custom keyword filters and LLM-based policy classification.</li>
                    <li><strong>MCP Tool Sandbox:</strong> Intercepts and validates MCP tool arguments before calling APIs.</li>
                </ul>
            </div>
            <div class="feature-card">
                <h4 style="color: #34d399;">💰 Performance & Scale</h4>
                <ul>
                    <li><strong>Ingestion Idempotency:</strong> Prevents duplicate vector insertion using SHA-256 chunk hashing.</li>
                    <li><strong>Unified Gateway:</strong> Routes requests through a self-hosted LiteLLM gateway with auto-failover.</li>
                    <li><strong>Dynamic Search Modes:</strong> Switches on-the-fly between Vector Similarity and MMR modes.</li>
                    <li><strong>Cost Analytics:</strong> Tracks tokens consumed and real-time execution costs per query.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()