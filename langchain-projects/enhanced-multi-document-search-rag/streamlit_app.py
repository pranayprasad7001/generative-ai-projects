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

# Page configuration
st.set_page_config(
    page_title="🤖 Premium RAG Search & Ingestion",
    page_icon="🔍",
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
        color: #f1f5f9;
        margin-bottom: 1rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
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


def process_ingestion(urls_input, uploaded_files, strategy_name, chunk_size, chunk_overlap):
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
            st.error("⚠️ Please specify at least one URL or upload a file.")
            return False

        # Initialize VectorStoreManager first (so we can get embeddings if needed)
        vector_store = VectorStoreManager()
        
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
        with st.status("🛠️ Building database...", expanded=True) as status:
            status.update(label="📄 Extracting text from files and URLs...", state="running")
            documents = doc_processor.load_documents(sources, strategy=strategy)
            
            if not documents:
                status.update(label="❌ Ingestion failed: No text content found.", state="error")
                st.error("⚠️ No text content could be extracted from the specified sources.")
                return False
                
            status.update(label=f"💾 Indexing {len(documents)} chunks into Astra DB...", state="running")
            vector_store.create_vectorstore(documents)
            
            status.update(label="⚡ Initializing agentic retrieval graph...", state="running")
            llm = Config.get_llm()
            graph_builder = GraphBuilder(
                retriever=vector_store.get_retriever(),
                llm=llm
            )
            graph_builder.build_graph()
            
            status.update(label="✅ Database successfully built!", state="complete")

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
        st.error(f"❌ Failed to build RAG database: {str(e)}")
        logging.exception("Error during ingestion process")
        return False


def main():
    init_session_state()
    
    # Title Section
    st.markdown('<div class="main-title">🤖 Premium RAG Search & Ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ingest custom files and URLs, build semantic index, and ask questions</div>', unsafe_allow_html=True)
    
    # Sidebar for Ingestion Controls
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🛠️ Ingestion Control Center</div>', unsafe_allow_html=True)
        
        # Tabs for URLs vs Files
        source_tab = st.radio("Choose Input Type:", ["Web URLs", "File Uploads", "Both"], horizontal=True)
        
        urls_input = ""
        uploaded_files = []
        
        if source_tab in ["Web URLs", "Both"]:
            st.markdown("#### 🌐 Web URLs")
            default_urls_str = "\n".join(Config.DEFAULT_URLS)
            urls_input = st.text_area(
                "Enter URLs (one per line):",
                value=default_urls_str,
                height=120,
                placeholder="https://example.com/article"
            )
            
        if source_tab in ["File Uploads", "Both"]:
            st.markdown("#### 📄 Local Files")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT, MD, CSV, XLSX):",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "md", "csv", "xlsx"]
            )
            
        st.markdown('<div class="sidebar-header" style="margin-top: 1.5rem;">⚙️ Settings</div>', unsafe_allow_html=True)
        
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
            
        st.markdown("---")
        
        # Build button
        build_db = st.button("🚀 Build RAG Database")
        if build_db:
            success = process_ingestion(
                urls_input=urls_input,
                uploaded_files=uploaded_files,
                strategy_name=strategy_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            if success:
                st.success(f"🎉 RAG database ready with {st.session_state.num_chunks} chunks!")
                st.toast("Database built successfully!")
                
    # Main content panel
    if not st.session_state.initialized:
        # Inform user to build database
        st.info("👈 **Please configure your data sources and click 'Build RAG Database' in the sidebar to initialize the search engine.**")
        
        # Beautiful feature list
        st.markdown("""
        ### Features of this Premium RAG Engine:
        * 📁 **Multi-Source Support**: Ingest URLs, PDFs, Word docs, CSV, Excel sheets, TXT, and Markdown files.
        * 🧠 **Dynamic Chunking**: Choose between standard **Recursive character chunking** and **Semantic chunking** using embeddings.
        * 🔍 **Hybrid Retrieval**: Employs an Ensemble Retriever combining vector similarity search (Astra DB) and keyword search (BM25).
        * 🤖 **Agentic Graph Workflow**: Powered by LangGraph to retrieve documents, reflect, and generate answers with Groq models.
        """)
    else:
        # Search Box UI
        st.success(f"🟢 Database Active: {st.session_state.num_chunks} chunks indexed (using {strategy_name} strategy).")
        
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
            with st.spinner("🔍 Agentic retrieval and answer generation in progress..."):
                start_time = time.time()
                try:
                    result = asyncio.run(st.session_state.rag_system.run(question))
                    elapsed_time = time.time() - start_time
                    
                    # Store in search history
                    st.session_state.history.append({
                        'question': question,
                        'answer': result.get('answer', 'No answer generated'),
                        'time': elapsed_time,
                        'retrieved_docs': result.get('retrieved_docs', [])
                    })
                    
                    # Display Answer
                    st.markdown("### 💡 Generated Answer")
                    st.markdown(f'<div class="answer-card">{result.get("answer", "No answer generated")}</div>', unsafe_allow_html=True)
                    
                    # Source documents display
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
                        st.info("No reference documents found.")
                        
                    st.caption(f"⏱️ Response latency: {elapsed_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ Error during retrieval/generation: {str(e)}")
                    logging.exception("Search workflow error")
                    
        # History panel
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 📜 Recent Search History")
            for idx, item in enumerate(reversed(st.session_state.history[-3:])):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>Q: {item['question']}</strong><br/>
                        <span style="color: #94a3b8; font-size: 0.9rem;">Answer: {item['answer'][:200]}...</span><br/>
                        <span style="color: #64748b; font-size: 0.8rem;">Latency: {item['time']:.2f}s | References: {len(item['retrieved_docs'])}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()