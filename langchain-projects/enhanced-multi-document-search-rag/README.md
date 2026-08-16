# Enhanced Multi-Document Search RAG

> An adaptive, self-correcting, and security-aware Retrieval-Augmented Generation system built with **LangGraph**, **AstraDB Vector Retrieval & Cohere Reranking**, **MCP-based External Search**, **LiteLLM Gateway**, **Layered Security Guardrails**, **7-Dimension Deterministic Chunk Versioning**, **Stage-by-Stage Latency Breakdowns**, **FastAPI Backend Server**, and **Streamlit UI**.

This project goes beyond fixed `retrieve → generate` pipelines. Instead of assuming retrieval and generation will always succeed, the system evaluates intermediate results using continuous confidence scores, rewrites ambiguous queries, executes critique-aware self-correction loops, escalates to external Model Context Protocol (MCP) search when local knowledge is insufficient, and enforces fail-closed security controls at all application boundaries.

The application can be operated via a **FastAPI REST API**, an interactive **Streamlit Dashboard**, or inspected visually through **LangGraph Studio**.

---

## Key Capabilities

* **Adaptive LangGraph Workflow:** Dynamic, bounded self-correction loops (`MAX_REWRITES = 3`, `MAX_GENERATIONS = 3`) that route queries intelligently based on document relevance, hallucination checks, and answer quality.
* **Modular Node Architecture (`src/nodes/`):** Clean separation into `security_nodes.py`, `retrieval_nodes.py`, `generation_nodes.py`, `evaluation_nodes.py`, and pure conditional `routing.py`.
* **Unified AstraDB Cloud Retrieval & Reranking:** Cloud-native AstraDB vector similarity and MMR retrieval with candidate oversampling (`candidate_k = max(2k, 10)`), combined with Cohere cross-encoder semantic reranking.
* **Continuous Evaluator Scoring:** Rich evaluation schemas returning continuous confidence scores (0.0 to 1.0), categorical decisions (`pass`, `retry`, `rewrite`, `fail`), and configurable thresholds.
* **Granular Trace Reasoning:** Stage-specific reasoning fields (`query_analysis`, `retrieval_reasoning`, `rewrite_reasoning`, `grounding_reasoning`, `relevance_reasoning`) for transparent LangSmith traces.
* **Structured MCP Citations:** Automatic structured source metadata extraction (`source`, `title`, `url`, `tool`, `retrieval_timestamp`) from MCP tools (Tavily, Wikipedia, arXiv).
* **7-Dimension Document Identity & Versioning:** Deterministic hashing across source, version, chunking strategy, chunk size, chunk overlap, embedding model, chunk index, and content to eliminate duplicate and stale index collisions.
* **Stage-by-Stage Latency Breakdown:** Real-time timing across query analysis, retrieval, reranker, grading, generation, MCP tools, and total latency.
* **Production REST API & Session Isolation:** FastAPI endpoints for isolated multi-tenant querying, document ingestion, and conversational history without mutating shared graph state.

---

## High-Level Architecture

```text
                         ┌───────────────────────────┐
                         │      User / Application   │
                         │                           │
                         │ Query + Runtime Input     │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌────────────────────────────┐
                         │      INPUT GUARDRAILS      │
                         │                            │
                         │ • PII Middleware           │
                         │ • Deterministic Filtering  │
                         │ • LLM Security Check       │
                         └─────────────┬──────────────┘
                                       │
                                Safe Input Only
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │       QUERY ANALYZER      │
                         │                           │
                         │ Local RAG or External     │
                         │ Search Classification     │
                         └─────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
             LOCAL RAG PATH                        EXTERNAL SEARCH
                    │                                     │
                    ▼                                     ▼
          ┌──────────────────┐                  ┌──────────────────┐
          │ AstraDB Retrieval│                  │    MCP Agent     │
          │                  │                  │                  │
          │ Vector Similarity│                  │ Tavily           │
          │ MMR Oversampling │                  │ Wikipedia        │
          │ Session Filtering│                  │ arXiv            │
          └────────┬─────────┘                  └────────┬─────────┘
                   │                                     │
                   ▼                                     │
          ┌──────────────────┐                           │
          │ Cohere Reranker  │                           │
          └────────┬─────────┘                           │
                   │                                     │
                   ▼                                     │
          ┌──────────────────┐                           │
          │ Document Grader  │                           │
          └────────┬─────────┘                           │
                   │                                     │
          ┌────────┴────────┐                            │
          │                 │                            │
       Relevant          Weak Retrieval                  │
          │                 │                            │
          ▼                 ▼                            │
       Generate        Query Rewrite                     │
          │                 │                            │
          │                 └──────► AstraDB Retrieval   │
          │                                              │
          ▼                                              │
   Hallucination Check                                   │
          │                                              │
     ┌────┴────┐                                         │
     │         │                                         │
 Grounded  Ungrounded                                    │
     │         │                                         │
     │         └──────► Regenerate (with Critique)       │
     │                                                   │
     ▼                                                   │
 Answer Relevance                                        │
     │                                                   │
 ┌───┴────┐                                              │
 │        │                                              │
Relevant Weak                                            │
 │        │                                              │
 │        └────────────► Query Rewrite                   │
 │                                                       │
 └──────────────────────┬────────────────────────────────┘
                        │
                        ▼
              ┌────────────────────────┐
              │   OUTPUT GUARDRAILS    │
              │                        │
              │ • PII Protection       │
              │ • Fail-Closed Safety   │
              │ • Sanitization Check   │
              └────────────┬───────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Response │
                  │                 │
                  │ Answer          │
                  │ Sources         │
                  │ Citations (MCP) │
                  │ Cost Breakdown  │
                  │ Latency Mapping │
                  └─────────────────┘
```

---

## StateGraph Nodes & Workflow

```text
input_query_security_check
        ↓
query_analyzer
        ↓
hybrid_retrieval (AstraDB Vector + Cohere Rerank)
        ↓
documents_grader
        ↓
query_rewriter
        ↓
answer_generator (with Critique-Aware Regeneration)
        ↓
hallucination_detector
        ↓
answer_relevance_grader
        ↓
external_search (MCP: Tavily / Wikipedia / arXiv)
        ↓
output_answer_security_check (Fail-Closed)
```

---

## Project Structure

```text
enhanced-multi-document-search-rag/
├── api.py                          # Production FastAPI REST API Server
├── streamlit_app.py                # Interactive Streamlit Web UI
├── requirements.txt                # Production Dependencies
├── Dockerfile                      # Container Build Configuration
├── docker-compose.yml              # Multi-Service Orchestration
├── src/
│   ├── config/
│   │   ├── llmgateway_config.py    # LiteLLM Gateway & RateLimited Embeddings
│   │   ├── mcp_config.py           # Model Context Protocol Client & Fallbacks
│   │   └── cost_callback.py        # Token & USD Cost Tracking Callback Handler
│   ├── document_ingestion/
│   │   ├── chunker.py              # Recursive, Semantic, & Hybrid Document Splitters
│   │   └── document_processor.py   # Multi-Format Ingestion (PDF, TXT, CSV, MD, Web)
│   ├── graph_builder/
│   │   └── adaptive_graph_builder.py # LangGraph StateGraph Construction & Execution
│   ├── nodes/
│   │   ├── __init__.py             # Module Exports
│   │   ├── security_nodes.py       # Input & Fail-Closed Output Security Guardrails
│   │   ├── retrieval_nodes.py      # Query Analyzer, Hybrid Retrieval, MCP Search, Rewriter
│   │   ├── generation_nodes.py     # Answer Generator & Critique-Aware Regeneration
│   │   ├── evaluation_nodes.py     # Continuous Document, Groundedness, & Relevance Graders
│   │   ├── routing.py              # Pure Conditional Edge Routing Functions
│   │   ├── adaptive_node.py        # Facade Unifying Modular Nodes
│   │   ├── guardrails.py           # Layered Security Middleware
│   │   └── schema.py               # Pydantic Output Schemas & Citation Models
│   ├── prompts/
│   │   └── rag_prompts.py          # Strict System & Grader Prompts
│   ├── state/
│   │   └── adaptive_state.py       # Central AdaptiveRAGState Model & Latency Fields
│   ├── studio_graph.py             # LangGraph Studio Visualization Entry Point
│   └── vectorstore/
│       └── vectorstore.py          # AstraDB Cloud Vector Store, Cohere Rerank, Deduplication
└── tests/
    ├── test_adaptive_node.py       # Node Facade & Pipeline Tests
    ├── test_api.py                 # FastAPI Endpoint Tests
    ├── test_chunker.py             # Recursive, Semantic, & Hybrid Chunker Tests
    ├── test_document_processor.py  # Multi-Format Document Ingestion Tests
    ├── test_evaluation_nodes.py    # Continuous Evaluator & Score Threshold Tests
    ├── test_generation_nodes.py    # Answer Generator & Critique Tests
    ├── test_graph_builder.py       # StateGraph Compilation Tests
    ├── test_mcp_config.py          # MCP Tool Fallback & Handler Tests
    ├── test_rag_integration.py     # End-to-End Multi-Turn Integration Tests
    ├── test_rate_limiter.py        # Rate-Limited Embedding & LLM Tests
    ├── test_retrieval_nodes.py     # Retrieval Nodes & Structured Citation Tests
    ├── test_routing.py             # Conditional Router Edge Tests
    ├── test_security_nodes.py      # Fail-Closed Security Tests
    └── test_vectorstore.py         # 7D Chunk Identity, Deduplication & Reranker Tests
```

---

## 7-Dimension Composite Document Identity

To guarantee zero-duplicate ingestion and prevent index pollution when preprocessing configurations change, chunk identity is computed deterministically:

$$\text{Chunk ID} = \text{SHA256}(\text{Source} \parallel \text{Version} \parallel \text{Strategy} \parallel \text{Size} \parallel \text{Overlap} \parallel \text{Model} \parallel \text{Index} \parallel \text{Content})$$

1. **`source`**: Path or URL of the ingested document.
2. **`doc_version`**: Explicit document version string (default: `"v1"`).
3. **`chunk_strategy`**: Splitting strategy used (`recursive`, `semantic`, `hybrid`).
4. **`chunk_size`**: Target chunk size.
5. **`chunk_overlap`**: Target chunk overlap.
6. **`embedding_model`**: Model used to vectorize the chunk (e.g. `gemini-embedding-2`).
7. **`chunk_index`**: Sequential chunk index within the document.
8. **`content`**: Exact textual body of the chunk.

---

## Latency Breakdown & Observability

Every query tracks high-resolution stage-by-stage timings:

```json
{
  "total_latency": 1.482,
  "latency_breakdown": {
    "security_input": 0.124,
    "query_analysis": 0.185,
    "hybrid_retrieval": 0.312,
    "reranker": 0.082,
    "grader_documents": 0.145,
    "generation": 0.421,
    "grader_hallucination": 0.112,
    "grader_relevance": 0.102,
    "security_output": 0.098,
    "total": 1.482
  }
}
```

---

## Getting Started

### 1. Prerequisites
* Python 3.11+
* AstraDB Account & Vector Database
* LiteLLM Gateway / API Keys (Groq, Gemini, Cohere, Tavily)

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# Vector Database
ASTRA_DB_API_ENDPOINT="https://<your-db-id>-<region>.apps.astra.datastax.com"
ASTRA_DB_API_KEY="AstraCS:..."
ASTRA_DB_COLLECTION_NAME="rag_multi_doc_collection"

# LiteLLM Proxy / Gateway
LITELLM_GATEWAY_URL="https://your-litellm-proxy.com"
LITELLM_GATEWAY_API_KEY="sk-..."
LLM_MODEL_GENERATOR="gemini/gemini-2.5-flash"
LLM_MODEL_CHECKER="gemini/gemini-2.5-flash"
EMBEDDING_MODEL="gemini/text-embedding-004"

# External Search & Reranking
COHERE_API_KEY="..."
TAVILY_API_KEY="tvly-..."

# Security & CORS
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8501"

# Observability (Optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="lsv2_pt_..."
LANGSMITH_PROJECT="enhanced-rag-adaptive"
```

### 3. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Running the Applications

#### Run the FastAPI REST Server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### Run the Streamlit Dashboard:
```bash
streamlit run streamlit_app.py
```

#### Inspect with LangGraph Studio:
```bash
langgraph dev
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check and vector database connectivity status. |
| `POST` | `/api/v1/query` | Execute Adaptive RAG pipeline with conversation history and session isolation. |
| `POST` | `/api/v1/ingest/files` | Upload and chunk documents (`.pdf`, `.txt`, `.docx`, `.md`, `.csv`, `.xlsx`). |
| `POST` | `/api/v1/ingest/urls` | Scrape, clean, and index web URLs into the vector store. |

---

## Running the Automated Test Suite

The project includes an extensive test suite with **104 unit and integration tests** covering component mocks, security guardrails, evaluator scoring thresholds, and end-to-end multi-turn conversation workflows:

```bash
pytest tests/ -v
```

---

# License

This project is part of the `generative-ai-projects` repository and is licensed under the **MIT License**.

See the repository-level [LICENSE](../../LICENSE) file for the complete license text.
