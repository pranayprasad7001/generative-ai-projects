"""FastAPI Backend Server for Enhanced Multi-Document Search RAG System.

This module provides production REST endpoints for:
- Query execution via the Adaptive LangGraph RAG workflow
- Multi-turn conversational context & coreference resolution
- Document & URL ingestion into AstraDB vector store with session isolation
- System health checks and observability stats
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

# Ensure src directory is available on path
sys.path.append(str(Path(__file__).parent / "src"))

from config.llmgateway_config import Config
from vectorstore.vectorstore import VectorStoreManager
from graph_builder.adaptive_graph_builder import GraphBuilder
from document_ingestion.document_processor import DocumentProcessor
from document_ingestion.chunker import ChunkStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rag_api")

# Global instances (initialized during startup)
vector_store_manager: Optional[VectorStoreManager] = None
rag_system: Optional[GraphBuilder] = None
doc_processor: Optional[DocumentProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown resource management."""
    global vector_store_manager, rag_system, doc_processor
    logger.info("Initializing RAG Core Services...")
    try:
        # Initialize VectorStore and DocumentProcessor
        vector_store_manager = VectorStoreManager()
        doc_processor = DocumentProcessor(embeddings=vector_store_manager.embeddings)
        
        # Initialize LLMs and GraphBuilder
        llm_generator = Config.get_llm_generator()
        llm_checker = Config.get_llm_checker()
        retriever = vector_store_manager.get_retriever()
        rag_system = GraphBuilder(
            retriever=retriever,
            llm_generator=llm_generator,
            llm_checker=llm_checker
        )
        rag_system.build_graph()
        logger.info("RAG Core Services initialized successfully.")
    except Exception as e:
        logger.error(f"Failed during startup initialization: {e}", exc_info=True)
    yield
    logger.info("Shutting down RAG Core Services...")


# Initialize FastAPI App
app = FastAPI(
    title="Enhanced Multi-Document Search RAG API",
    description="Production REST API for Adaptive RAG with Hybrid Retrieval and LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Configurable CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Schemas
# ============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class QueryRequest(BaseModel):
    question: str = Field(..., description="The query/question to answer")
    search_type: Optional[str] = Field("similarity", description="'similarity' or 'mmr'")
    k: Optional[int] = Field(5, ge=1, le=50, description="Number of documents to retrieve (1 to 50)")
    thread_id: Optional[str] = Field("default_thread", description="Thread identifier for session tracking")
    session_id: Optional[str] = Field(None, description="Optional tenant or session identifier for corpus isolation")
    messages: Optional[List[ChatMessage]] = Field(default_factory=list, description="Prior conversation history")


class QueryResponse(BaseModel):
    question: str
    answer: str
    latency_seconds: float
    total_cost: float
    retrieved_docs: List[Dict[str, Any]] = []
    external_citations: List[Union[Dict[str, Any], str]] = []
    latency_breakdown: Dict[str, float] = {}


class IngestUrlRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to ingest")
    strategy: Optional[str] = Field("recursive", description="Chunk strategy: recursive, semantic, or hybrid")
    session_id: Optional[str] = Field(None, description="Optional tenant/session ID to tag documents")


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
    sources: List[str]


class HealthResponse(BaseModel):
    status: str
    astra_db_connected: bool
    model_configured: str
    timestamp: float


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for Render, Kubernetes, and load balancers."""
    astra_ok = False
    if vector_store_manager and getattr(vector_store_manager, "vectorstore", None):
        try:
            col = getattr(vector_store_manager.vectorstore.astra_env, "collection", None)
            astra_ok = col is not None
        except Exception:
            astra_ok = False

    return HealthResponse(
        status="healthy" if astra_ok else "degraded",
        astra_db_connected=astra_ok,
        model_configured=Config.LLM_MODEL_GENERATOR,
        timestamp=time.time()
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """Execute the Adaptive LangGraph RAG pipeline on a given question."""
    if not rag_system or not vector_store_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized."
        )

    start_time = time.time()
    try:
        # Dynamically create per-query retriever without mutating shared graph state
        search_type = "mmr" if request.search_type and request.search_type.lower() == "mmr" else "similarity"
        query_retriever = vector_store_manager.get_retriever(
            search_type=search_type,
            k=request.k or 5,
            session_id=request.session_id
        )

        # Convert chat history to LangChain message instances
        history_messages = []
        for msg in request.messages or []:
            if msg.role.lower() == "assistant":
                history_messages.append(AIMessage(content=msg.content))
            else:
                history_messages.append(HumanMessage(content=msg.content))

        # Run query through compiled graph
        result = await rag_system.run(
            question=request.question,
            thread_id=request.thread_id,
            messages=history_messages,
            retriever=query_retriever
        )
        latency = time.time() - start_time

        # Format retrieved docs
        raw_docs = result.get("retrieved_docs", [])
        formatted_docs = []
        for doc in raw_docs:
            if hasattr(doc, "page_content"):
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {})
                })
            elif isinstance(doc, dict):
                formatted_docs.append(doc)

        return QueryResponse(
            question=request.question,
            answer=result.get("answer", "No answer generated."),
            latency_seconds=round(latency, 3),
            total_cost=result.get("total_cost", 0.0),
            retrieved_docs=formatted_docs,
            external_citations=result.get("external_citations", []),
            latency_breakdown=result.get("latency_breakdown", {})
        )

    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/api/v1/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_urls(request: IngestUrlRequest):
    """Ingest web URLs, chunk them, and store them into AstraDB."""
    if not vector_store_manager or not doc_processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processor or VectorStore is not initialized."
        )

    try:
        strategy_map = {
            "recursive": ChunkStrategy.RECURSIVE,
            "semantic": ChunkStrategy.SEMANTIC,
            "hybrid": ChunkStrategy.HYBRID,
        }
        chunk_strategy = strategy_map.get((request.strategy or "recursive").lower(), ChunkStrategy.RECURSIVE)
        
        # Load & chunk
        docs = doc_processor.process_urls(request.urls, strategy=chunk_strategy)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content could be extracted from the provided URLs."
            )

        if request.session_id:
            for doc in docs:
                doc.metadata["session_id"] = request.session_id

        # Ingest into vector store
        chunks_count = vector_store_manager.add_documents(docs)

        return IngestResponse(
            message="URLs ingested successfully.",
            chunks_indexed=chunks_count,
            sources=request.urls
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting URLs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL ingestion failed: {str(e)}"
        )


@app.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    session_id: Optional[str] = Form(None)
):
    """Upload a file (PDF, TXT, DOCX, MD, CSV, XLSX) and ingest it into AstraDB."""
    if not vector_store_manager or not doc_processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processor or VectorStore is not initialized."
        )

    temp_path = None
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in doc_processor.supported_loaders:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: '{file_ext}'. Supported: {list(doc_processor.supported_loaders.keys())}"
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        strategy_map = {
            "recursive": ChunkStrategy.RECURSIVE,
            "semantic": ChunkStrategy.SEMANTIC,
            "hybrid": ChunkStrategy.HYBRID,
        }
        chunk_strategy = strategy_map.get(strategy.lower(), ChunkStrategy.RECURSIVE)

        # Process document
        docs = doc_processor.load_documents([temp_path], strategy=chunk_strategy)
        for doc in docs:
            doc.metadata["source"] = file.filename
            if session_id:
                doc.metadata["session_id"] = session_id

        # Ingest into vector store
        chunks_count = vector_store_manager.add_documents(docs)

        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully.",
            chunks_indexed=chunks_count,
            sources=[file.filename]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)

