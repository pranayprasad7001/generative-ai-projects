# -*- coding: utf-8 -*-
"""Streamlit UI for Agentic RAG System - Enhanced Version"""

import streamlit as st
from pathlib import Path
import sys
import time
import logging

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
import asyncio
from graph_builder.adaptive_graph_builder import GraphBuilder

# LangSmith
LANGSMITH_TRACING = Config.LANGSMITH_TRACING
LANGSMITH_API_KEY = Config.LANGSMITH_API_KEY
LANGSMITH_PROJECT = Config.LANGSMITH_PROJECT

def format_cost(cost: float) -> str:
    """Format float cost to user-readable string without scientific notation."""
    if cost == 0.0:
        return "$0.00"
    # Format to 7 decimal places to capture micro-costs
    val = f"{cost:.7f}"
    # Strip trailing zeros, keeping at least 2 decimal places
    while val.endswith("0") and len(val.split(".")[1]) > 2:
        val = val[:-1]
    return f"${val}"

# Page configuration
st.set_page_config(
    page_title="\U0001f916 Enhanced Multi-Document Search RAG",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
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
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Ingestion button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    div.stButton > button:first-child:hover {
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
    
    /* Answer Card with soft border gradient feel */
    .answer-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #10b981;
    }
    
    /* Source text areas */
    .source-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    /* Sidebar Headers */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 0.5rem;
    }
    
    /* Features Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
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
        font-size: 1.1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .feature-card ul {
        margin: 0;
        padding-left: 1.2rem;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .feature-card strong {
        color: #f1f5f9;
    }
    .feature-card li {
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .feature-card li:last-child {
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'num_chunks' not in st.session_state:
        st.session_state.num_chunks = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None


def process_ingestion(urls_input, uploaded_files, strategy_name, chunk_size, chunk_overlap, search_type_choice):
    """Process files and URLs, build the vector store, and initialize RAG"""
    try:
        # Create temp folder for files
        temp_dir = Path("data/uploaded")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        sources = []
        # Parse URLs
        if urls_input:
            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
            sources.extend(urls)
            
        # Save uploaded files
        saved_files = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                sources.append(str(file_path))
                saved_files.append(file_path)
                
        if not sources:
            st.error("\u26a0\ufe0f Please specify at least one URL or upload a file.")
            return False

        # Initialize VectorStoreManager first (so we can get embeddings if needed)
        vector_store = VectorStoreManager()
        st.session_state.vector_store = vector_store
        
        # Decide embeddings to use for semantic chunking / hybrid chunking
        embeddings = None
        if strategy_name in ["Semantic", "Hybrid"]:
            embeddings = vector_store.embeddings
            strategy = ChunkStrategy.SEMANTIC if strategy_name == "Semantic" else ChunkStrategy.HYBRID
        else:
            strategy = ChunkStrategy.RECURSIVE

        # Initialize DocumentProcessor
        doc_processor = DocumentProcessor(
            embeddings=embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # We perform ingestion in streamlit using visual feedback
        with st.status("\U0001f6e0\ufe0f Building database...", expanded=True) as status:
            status.update(label="\U0001f4c4 Extracting text from files and URLs...", state="running")
            documents = doc_processor.load_documents(sources, strategy=strategy)
            
            if not documents:
                status.update(label="\u274c Ingestion failed: No text content found.", state="error")
                st.error("\u26a0\ufe0f No text content could be extracted from the specified sources.")
                return False
                
            status.update(label=f"\U0001f4be Indexing {len(documents)} chunks into Astra DB...", state="running")
            vector_store.create_vectorstore(documents)
            
            status.update(label="\u26a1 Initializing agentic retrieval graph...", state="running")
            llm = Config.get_llm()
            search_type = "mmr" if search_type_choice == "Maximal Marginal Relevance (MMR)" else "similarity"
            graph_builder = GraphBuilder(
                retriever=vector_store.get_retriever(search_type=search_type),
                llm=llm
            )
            graph_builder.build_graph()
            
            status.update(label="\u2705 Database successfully built!", state="complete")

        # Save to session state
        st.session_state.rag_system = graph_builder
        st.session_state.num_chunks = len(documents)
        st.session_state.initialized = True
        
        # Clean up temp files
        for file_path in saved_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logging.warning("Failed to delete temp file %s: %s", file_path, e)
                
        return True
    except Exception as e:
        st.error(f"\u274c Failed to build RAG database: {str(e)}")
        logging.exception("Error during ingestion process")
        return False


def main():
    init_session_state()
    
    # Title Section
    st.markdown('<div class="main-title">\U0001f916 Enhanced Multi-Document Search RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ingest custom files and URLs, build semantic index, and ask questions</div>', unsafe_allow_html=True)
    
    # Sidebar for Ingestion Controls
    with st.sidebar:
        st.markdown('<div class="sidebar-header">\U0001f6e0\ufe0f Ingestion Control Center</div>', unsafe_allow_html=True)
        
        # Tabs for URLs vs Files
        source_tab = st.radio("Choose Input Type:", ["Web URLs", "File Uploads", "Both"], horizontal=True)
        
        urls_input = ""
        uploaded_files = []
        
        if source_tab in ["Web URLs", "Both"]:
            st.markdown("#### \U0001f310 Web URLs")
            default_urls_str = "\n".join(Config.DEFAULT_URLS)
            urls_input = st.text_area(
                "Enter URLs (one per line):",
                value=default_urls_str,
                height=120,
                placeholder="https://example.com/article"
            )
            
        if source_tab in ["File Uploads", "Both"]:
            st.markdown("#### \U0001f4c4 Local Files")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT, MD, CSV, XLSX):",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "md", "csv", "xlsx"]
            )
            
        st.markdown('<div class="sidebar-header" style="margin-top: 1.5rem;">\u2699\ufe0f Settings</div>', unsafe_allow_html=True)
        
        # Chunking strategy and slider configurations
        strategy_name = st.selectbox(
            "Chunking Strategy:",
            ["Recursive", "Semantic", "Hybrid"],
            help="Recursive uses standard character splitting. Semantic uses vector embedding distances. Hybrid combines both."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size:", 100, 2000, Config.CHUNK_SIZE, 50)
        with col2:
            chunk_overlap = st.slider("Overlap:", 0, 500, Config.CHUNK_OVERLAP, 10)
            
        search_type_choice = st.selectbox(
            "Search Type:",
            ["Similarity Search", "Maximal Marginal Relevance (MMR)"],
            help="Similarity Search retrieves based on nearest vector. MMR balances relevance and diversity."
        )
            
        st.markdown("---")
        
        # Build button
        build_db = st.button("\U0001f680 Build RAG Database")
        if build_db:
            success = process_ingestion(
                urls_input=urls_input,
                uploaded_files=uploaded_files,
                strategy_name=strategy_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                search_type_choice=search_type_choice
            )
            if success:
                st.success(f"\U0001f389 RAG database ready with {st.session_state.num_chunks} chunks!")
                st.toast("Database built successfully!")
                
    # Main content panel
    if not st.session_state.initialized:
        # Inform user to build database
        st.info("\U0001f448 **Please configure your data sources and click 'Build RAG Database' in the sidebar to initialize the search engine.**")
        
        # Beautiful feature list
        st.markdown("""
        <h3 style="margin-top: 1.5rem; color: #f1f5f9;">Architecture & Features of this Premium RAG Engine</h3>
        <div class="features-grid">
            <div class="feature-card">
                <h4 style="color: #818cf8;">\U0001f4c4 Ingestion & Search</h4>
                <ul>
                    <li><strong>Multi-Source:</strong> Loads Web URLs, PDFs, Word, Excel, CSV, TXT, and Markdown.</li>
                    <li><strong>Ensemble Search:</strong> Blends Astra DB vector similarity with BM25 keyword matching.</li>
                    <li><strong>Cohere Reranking:</strong> Refines context relevance using advanced cross-attention models.</li>
                    <li><strong>Smart Chunking:</strong> Supports standard Recursive Character or embedding-based Semantic/Hybrid chunking.</li>
                </ul>
            </div>
            <div class="feature-card">
                <h4 style="color: #c084fc;">\U0001f916 Agentic Self-Correction</h4>
                <ul>
                    <li><strong>LangGraph Workflow:</strong> Replaces rigid pipelines with a flexible StateGraph orchestrator.</li>
                    <li><strong>Double-Loop Correction:</strong> Grades context, rewrites weak queries, and catches hallucinations.</li>
                    <li><strong>Runaway Protection:</strong> Enforces bounded retry loops (max 3) to prevent runaway LLM calls.</li>
                    <li><strong>MCP Escalation:</strong> Seamlessly queries Tavily or Wikipedia MCP servers as search backup.</li>
                </ul>
            </div>
            <div class="feature-card">
                <h4 style="color: #fb7185;">\U0001f6e1\ufe0f Safety & Guardrails</h4>
                <ul>
                    <li><strong>PII Masking:</strong> Automatic masking/redaction of emails, credit cards, phones, and SSNs.</li>
                    <li><strong>Bi-Directional Filtering:</strong> Audits incoming user queries and filters generated LLM responses.</li>
                    <li><strong>Security Graders:</strong> Integrates custom keyword filters and LLM-based policy classification.</li>
                    <li><strong>MCP Tool Sandbox:</strong> Intercepts and validates MCP tool arguments before calling APIs.</li>
                </ul>
            </div>
            <div class="feature-card">
                <h4 style="color: #34d399;">\U0001f4b0 Performance & Scale</h4>
                <ul>
                    <li><strong>Ingestion Idempotency:</strong> Prevents duplicate vector insertion using SHA-256 chunk hashing.</li>
                    <li><strong>Unified Gateway:</strong> Routes requests through a self-hosted LiteLLM gateway with auto-failover.</li>
                    <li><strong>Dynamic Search Modes:</strong> Switches on-the-fly between Vector Similarity and MMR modes.</li>
                    <li><strong>Cost Analytics:</strong> Tracks tokens consumed and real-time execution costs per query.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Search Box UI
        st.success(f"\U0001f7e2 Database Active: {st.session_state.num_chunks} chunks indexed (using {strategy_name} strategy).")
        
        with st.form("search_form"):
            col_in, col_btn = st.columns([5, 1])
            with col_in:
                question = st.text_input(
                    "Enter your question:",
                    placeholder="e.g. What is memory in AI agents?"
                )
            with col_btn:
                st.markdown('<div class="search-btn" style="margin-top: 1.7rem;">', unsafe_allow_html=True)
                submit = st.form_submit_button("Search")
                st.markdown('</div>', unsafe_allow_html=True)
                
        # Handle Search Execution
        if submit and question:
            with st.spinner("\U0001f50d Agentic retrieval and answer generation in progress..."):
                start_time = time.time()
                try:
                    if st.session_state.vector_store is not None:
                        search_type = "mmr" if search_type_choice == "Maximal Marginal Relevance (MMR)" else "similarity"
                        st.session_state.rag_system.nodes.retriever = st.session_state.vector_store.get_retriever(search_type=search_type)
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
                    st.markdown("### \U0001f4a1 Generated Answer")
                    st.markdown(f'<div class="answer-card">{result.get("answer", "No answer generated")}</div>', unsafe_allow_html=True)
                    
                    # Premium stats badge under answer card
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 1.5rem; margin-top: -0.75rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: #f8fafc; background-color: #1e293b; padding: 0.5rem 1rem; border-radius: 0.375rem; width: fit-content; border: 1px solid #475569;">
                            <span>\u23f1\ufe0f <strong>Latency:</strong> {elapsed_time:.2f}s</span>
                            <span style="color: #64748b;">|</span>
                            <span>\U0001f4b0 <strong>Estimated Cost:</strong> <code style="color: #34d399; font-weight: bold; background: none; padding: 0;">{format_cost(cost)}</code></span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display external citations if any
                    external_citations = result.get('external_citations', [])
                    if external_citations:
                        links = [
                            f'<a href="{url}" target="_blank" style="color: #6366f1; text-decoration: none; font-weight: 500;">{url.split("//")[-1].split("/")[0].replace("www.", "")}</a>'
                            for url in external_citations
                        ]
                        st.markdown(
                            f"""
                            <div style="font-size: 0.9rem; color: #94a3b8; margin-top: -0.75rem; margin-bottom: 1.5rem;">
                                \U0001f517 <strong>External Citations:</strong> {" \u2022 ".join(links)}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Source documents display
                    st.markdown("### \U0001f4c4 Retrieved Source Documents")
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
                        st.info("No reference documents found.")
                        
                except Exception as e:
                    st.error(f"\u274c Error during retrieval/generation: {str(e)}")
                    logging.exception("Search workflow error")
                    
        # History panel
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### \U0001f4dc Recent Search History")
            for idx, item in enumerate(reversed(st.session_state.history[-3:])):
                item_cost = item.get('cost', 0.0)
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>Q: {item['question']}</strong><br/>
                        <span style="color: #94a3b8; font-size: 0.9rem;">Answer: {item['answer'][:200]}...</span><br/>
                        <span style="color: #64748b; font-size: 0.8rem;">Latency: {item['time']:.2f}s | Cost: {format_cost(item_cost)} | References: {len(item['retrieved_docs'])} | Citations: {len(item.get('external_citations', []))}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()