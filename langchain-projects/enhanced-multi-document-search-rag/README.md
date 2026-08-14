# Enhanced Multi-Document Search RAG

> An adaptive, self-correcting and security-aware Retrieval-Augmented Generation system built with LangGraph, hybrid retrieval, Cohere reranking, MCP-based external search, LiteLLM model routing, Redis semantic caching, multi-layer guardrails, cost tracking, and component-level testing.

This project goes beyond a fixed `retrieve → generate` RAG pipeline.

Instead of assuming that retrieval and generation will always succeed, the system evaluates intermediate results, rewrites weak queries, retries generation when necessary, escalates to external search when local knowledge is insufficient, and applies security controls at the application boundaries and around external tool interactions.

The application is exposed through Streamlit and the underlying LangGraph workflow can also be inspected independently through LangGraph Studio.

---

# Why This Project?

A basic RAG system typically follows:

```text
User Query
    ↓
Retrieve Documents
    ↓
Generate Answer
```

That approach does not explicitly handle:

- Poor retrieval
- Ambiguous queries
- Redundant retrieval results
- Hallucinated answers
- Irrelevant answers
- Missing knowledge
- External knowledge requirements
- Prompt injection / unsafe inputs
- Sensitive information in inputs or outputs
- Unsafe external tool calls
- Repeated LLM requests
- LLM cost and latency
- API failures and retries

This project addresses these concerns through an adaptive LangGraph workflow with bounded self-correction, hybrid retrieval, reranking, external MCP search, layered guardrails, semantic caching, reliability controls, and observability.

---

# Architecture

## High-Level Architecture

The application treats the **input and output boundaries as security boundaries**.

All externally supplied query input enters the workflow through the input guardrail layer, and generated responses leave the workflow only after passing through the output security layer.

External MCP tool calls are additionally protected by middleware that validates tool arguments and filters unsafe tool results.

```text
                         ┌───────────────────────────┐
                         │      User / Application    │
                         │                           │
                         │ Query + Runtime Input      │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      INPUT GUARDRAILS      │
                         │                           │
                         │ • PII Middleware           │
                         │ • Deterministic Filtering  │
                         │ • LLM Security Check       │
                         └─────────────┬─────────────┘
                                       │
                              Safe Input Only
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │       QUERY ANALYZER       │
                         │                           │
                         │ Local RAG or External      │
                         │ Search Classification      │
                         └─────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
             LOCAL RAG PATH                        EXTERNAL SEARCH
                    │                                     │
                    ▼                                     ▼
          ┌──────────────────┐                  ┌──────────────────┐
          │ Hybrid Retrieval │                  │    MCP Agent     │
          │                  │                  │                  │
          │ Dense + BM25     │                  │ Tavily           │
          │ Similarity/MMR   │                  │ Wikipedia        │
          │ Ensemble         │                  │ arXiv            │
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
          │                 └──────► Retrieval            │
          │                                             │
          ▼                                             │
   Hallucination Check                                  │
          │                                             │
     ┌────┴────┐                                        │
     │         │                                        │
 Grounded  Ungrounded                                   │
     │         │                                        │
     │         └──────► Regenerate                      │
     │                                                  │
     ▼                                                  │
 Answer Relevance                                       │
     │                                                  │
 ┌───┴────┐                                             │
 │        │                                             │
Relevant Weak                                           │
 │        │                                             │
 │        └────────────► Query Rewrite                   │
 │                                                      │
 └──────────────────────┬───────────────────────────────┘
                        │
                        ▼
              ┌────────────────────────┐
              │   OUTPUT GUARDRAILS    │
              │                        │
              │ • PII Protection       │
              │ • LLM Safety Check     │
              │ • Deterministic Filter │
              └────────────┬───────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Response │
                  │                 │
                  │ Answer          │
                  │ Sources         │
                  │ Citations       │
                  │ Cost / Latency  │
                  └─────────────────┘
```

### Important security boundary

The top-level flow is intentionally:

```text
Application Input
      ↓
Input Guardrails
      ↓
RAG / Agent Workflow
      ↓
Output Guardrails
      ↓
Application Output
```

The guardrails are not treated as an optional post-processing feature.

They form the **input and output security boundaries** of the application.

For the external-search branch, tool calls and tool results receive additional middleware protection:

```text
Input Guardrails
      ↓
External Search Agent
      ↓
Tool Argument Validation
      ↓
MCP Tool
      ↓
Tool Result Filtering
      ↓
External Answer
      ↓
Output Guardrails
      ↓
Final Response
```

---

# LangGraph Workflow

The workflow is implemented using LangGraph `StateGraph`.

Current nodes:

```text
input_query_security_check
        ↓
query_analyzer
        ↓
vector_search
        ↓
documents_grader
        ↓
query_rewriter
        ↓
answer_generator
        ↓
hallucination_detector
        ↓
answer_relevance_grader
        ↓
external_search
        ↓
output_answer_security_check
```

Not every request traverses every node.

Conditional routing determines the path based on:

- Input security
- Query classification
- Retrieval quality
- Query rewrite count
- Hallucination detection
- Answer relevance
- Generation count
- External knowledge requirements

The self-correction loops are bounded to prevent uncontrolled execution.

```text
MAX_REWRITES = 3
MAX_GENERATIONS = 3
```

---

# Adaptive RAG Flow

## 1. Input Security Check

Every user query enters through the input security layer before reaching the main RAG workflow.

The input guardrail stack contains:

1. PII middleware
2. Deterministic content filtering
3. LLM-based security classification

Unsafe requests can terminate before retrieval or generation.

---

## 2. Query Analysis

The query analyzer determines whether the request should use:

```text
Local Knowledge
      or
External Search
```

Local questions continue through the RAG pipeline.

Questions requiring external knowledge can be routed directly to the MCP-based search agent.

---

# Hybrid Retrieval

The local retrieval pipeline combines semantic and lexical retrieval.

```text
                    Query
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Dense Retrieval              BM25
   AstraDB Vector Store       Lexical Search
          │
     Similarity / MMR
          │
          └───────────┬───────────┘
                      ▼
              EnsembleRetriever
                Dense = 0.7
                BM25  = 0.3
                      │
                      ▼
                Cohere Reranker
                      │
                      ▼
                Final Context
```

## Dense Retrieval

The vector database is AstraDB.

The configured embedding model is:

```text
gemini-embedding-2
```

implemented using `GoogleGenerativeAIEmbeddings`.

## Similarity Search

Retrieves documents based primarily on semantic similarity to the query.

## MMR Search

Maximal Marginal Relevance balances relevance and diversity to reduce redundant retrieved documents.

The Streamlit interface allows switching between:

```text
Similarity Search
MMR Search
```

## BM25

BM25 provides lexical retrieval using `rank_bm25` through LangChain's `BM25Retriever`.

## Ensemble Retrieval

Dense and lexical retrieval are combined with:

```text
Dense weight = 0.7
BM25 weight  = 0.3
```

## Reranking

Cohere `rerank-english-v3.0` reranks the retrieved candidates through LangChain's `ContextualCompressionRetriever`.

---

# Retrieval Self-Correction

Retrieved documents are evaluated before answer generation.

```text
Retrieve
   ↓
Document Grader
   ↓
Are documents sufficient?
   │
 ┌─┴─────────────┐
 │               │
Yes              No
 │               │
 ▼               ▼
Generate      Query Rewrite
                  │
                  ▼
              Retrieve
```

If the retrieved context remains insufficient after the bounded rewrite attempts, the workflow can escalate to external search.

---

# Answer Verification

Generated answers are not immediately returned.

They pass through two verification stages.

## Hallucination Detection

```text
Generated Answer
      ↓
Hallucination Detector
      │
 ┌────┴─────────┐
 │              │
Grounded     Ungrounded
 │              │
 ▼              ▼
Continue     Regenerate
```

Generation retries are bounded.

```text
MAX_GENERATIONS = 3
```

## Answer Relevance

A grounded answer is subsequently evaluated for relevance to the original query.

```text
Answer
  ↓
Relevance Grader
  │
 ┌┴───────────┐
 │            │
Relevant     Weak
 │            │
 ▼            ▼
Output      Query Rewrite
Guardrail       ↓
            Retrieval
```

This allows the system to correct answers that may be factually grounded but still fail to address the user's question.

---

# Multi-Layer Guardrails

The project uses a defense-in-depth approach.

## Input Guardrails

```text
User Input
    ↓
PII Middleware
    ↓
Deterministic Content Filter
    ↓
LLM Security Classification
    ↓
Safe Input
```

PII middleware can process:

- Email addresses
- Credit card numbers
- API keys
- Phone numbers
- IP addresses
- SSNs

Depending on the detector, the configured strategies include:

```text
redact
mask
block
```

---

# External Tool Guardrails

External search is performed through MCP.

Tool interactions are not trusted automatically.

The middleware can:

- Inspect tool arguments
- Block unsafe tool arguments
- Limit the number of tool calls
- Handle tool failures
- Filter unsafe tool results

```text
Agent
  ↓
Tool Call
  ↓
Argument Validation
  ↓
MCP Tool
  ↓
Tool Result
  ↓
Result Filtering
  ↓
Agent
```

This provides an additional security boundary around external tools.

---

# Output Guardrails

Generated responses pass through the output security layer before being returned to the application.

```text
Generated / External Answer
          ↓
    Output Security Agent
          ↓
 Deterministic Middleware
          ↓
       Final Answer
```

The output layer includes:

- PII protection
- LLM-based security checking
- Deterministic sensitive-pattern filtering

Sensitive patterns include examples such as:

```text
api_key
password
secret
access_token
private_key
bearer token
client_secret
ssh key
connection string
database password
```

The final answer is therefore not returned directly from the generation or external-search node.

---

# External Search with MCP

When local retrieval cannot sufficiently answer a query, the workflow can escalate to external search.

Current MCP integrations:

- Tavily
- Wikipedia
- arXiv

The external search agent uses LangChain's `create_agent` API with `langchain-mcp-adapters`.

```text
Local RAG
   ↓
Insufficient / External Knowledge Required
   ↓
External Search Agent
   ↓
MCP
 ┌──────┬──────────┐
 ▼      ▼          ▼
Tavily Wikipedia  arXiv
   │      │          │
   └──────┴──────────┘
             ↓
      External Answer
             ↓
      Output Guardrails
             ↓
       Final Response
```

External URLs returned by tool results are surfaced in the Streamlit interface.

---

# Multi-Format Document Ingestion

The ingestion pipeline supports:

- PDF
- DOCX
- TXT
- Markdown
- CSV
- XLSX
- XLS
- Web URLs
- Directories containing supported documents

The ingestion layer uses loaders including:

- `WebBaseLoader`
- `PyMuPDFLoader`
- `TextLoader`
- `Docx2txtLoader`
- `CSVLoader`
- `UnstructuredExcelLoader`

Documents are enriched with metadata such as:

```text
source
file_name
file_type
loader
chunk
```

---

# Chunking Strategies

Three chunking strategies are available.

## Recursive

Uses:

```text
RecursiveCharacterTextSplitter
```

## Semantic

Uses:

```text
SemanticChunker
```

with the configured embedding model.

## Hybrid

Combines semantic chunking with recursive splitting.

Markdown documents additionally use:

```text
MarkdownHeaderTextSplitter
```

Default configuration:

```text
Chunk size    = 500
Chunk overlap = 50
```

---

# Idempotent Document Ingestion

Each document chunk receives a deterministic SHA-256 identifier based on:

```text
source + chunk content
```

Before insertion, the identifier is checked against AstraDB.

```text
Generate Chunk ID
      ↓
Already Exists?
   ┌──┴──┐
  Yes    No
   │      │
 Skip   Insert
```

This prevents unnecessary duplicate vector insertion when the same document content is ingested repeatedly.

---

# LiteLLM Gateway

LLM calls are routed through a self-hosted LiteLLM proxy.

This provides a centralized layer for:

- Model configuration
- Provider abstraction
- Retries
- Timeout handling
- Fallbacks
- Rate limiting
- Semantic caching
- Cost tracking

Current application model:

```text
gpt-oss-120b-groq
```

Configured models include:

```text
gpt-oss-120b-groq
gpt-oss-20b-groq
qwen3.6-27b-groq
nvidia-glm-5.2
```

Configured NVIDIA fallback chain:

```text
nvidia-glm-5.2
       ↓
gpt-oss-120b-groq
       ↓
gpt-oss-20b-groq
```

The application accesses the models through the LiteLLM gateway rather than coupling the RAG workflow directly to a specific provider.

---

# Redis Semantic Caching

Semantic caching is implemented through LiteLLM and Redis.

Current configuration:

```text
Cache type:              redis-semantic
Similarity threshold:    0.85
TTL:                     1800 seconds / 30 minutes
Cache embeddings:        Cohere Embed English v3
```

Caching is disabled by default at the gateway level and explicitly enabled for selected LLM calls.

```text
Query
  ↓
Semantic Cache
  │
 ┌┴──────┐
Hit     Miss
 │        │
 ▼        ▼
Cached    LLM
Answer    Call
            │
            ▼
          Cache
```

The objective is to reduce repeated LLM inference, latency, and cost for sufficiently similar requests.

---

# LLM Cost Tracking

Each workflow execution uses a custom `CostTrackingCallbackHandler`.

It:

1. Reads LiteLLM cost information when available.
2. Falls back to token-based cost calculation when necessary.
3. Accumulates cost across multiple LLM calls.
4. Exposes cumulative request cost through `total_cost`.

This means retries and self-correction calls can contribute to the displayed request cost.

The Streamlit interface displays:

- Request latency
- Estimated LLM cost

---

# Reliability Controls

The system includes several reliability mechanisms:

- Bounded query rewriting
- Bounded generation retries
- LiteLLM gateway retries
- Application-level retry handling
- Rate limiting
- External-search escalation
- Reranker fallback handling
- Graph construction error handling
- Idempotent document ingestion

The bounded correction loops are particularly important because an adaptive RAG system should not be allowed to retry indefinitely.

---

# Observability

Current observability includes:

- Python logging
- Per-query latency
- Estimated LLM cost
- Retrieved-document inspection
- External citation inspection
- Search history
- Optional LangSmith tracing
- LangGraph Studio inspection

The goal is to make both the workflow behavior and its operational characteristics easier to inspect.

---

# Streamlit Application

The Streamlit interface provides a configurable RAG experience.

## Input Sources

Users can select:

- Web URLs
- File uploads
- Both

## Supported Uploads

- PDF
- DOCX
- TXT
- Markdown
- CSV
- XLSX

## Chunking Controls

- Recursive
- Semantic
- Hybrid
- Adjustable chunk size
- Adjustable overlap

## Retrieval Controls

- Similarity Search
- MMR

## Query Results

The interface exposes:

- Generated answer
- Retrieved source chunks
- External citations
- Search history
- Latency
- Estimated cost

---

# LangGraph Studio

The workflow is registered through:

```text
langgraph.json
```

and can be inspected independently from Streamlit.

Run:

```bash
langgraph dev
```

This makes it possible to inspect the graph and its conditional execution independently of the UI.

---

# Testing

Component-level tests are included under `tests/`.

```text
tests/
├── test_adaptive_node.py
├── test_chunker.py
├── test_document_processor.py
├── test_graph_builder.py
└── test_vectorstore.py
```

Current tests cover areas such as:

- Adaptive node behavior
- Security checks
- Query analysis
- Graph construction and routing
- Chunking
- Document processing
- Vector store behavior

Run:

```bash
pytest
```

### RAG Evaluation — In Progress

I am currently learning **RAG evaluation and RAGAS**.

Comprehensive RAG evaluation using RAGAS and related evaluation methodologies will be implemented after completing the learning phase.

Planned evaluation areas include:

- Retrieval relevance
- Context precision
- Context recall
- Answer faithfulness
- Answer relevance
- End-to-end RAG quality
- Latency and cost comparison
- Retrieval strategy comparison

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Orchestration | LangGraph `StateGraph` |
| LLM Interface | LangChain `ChatOpenAI` |
| LLM Gateway | LiteLLM Proxy |
| Application LLM | Groq `gpt-oss-120b` |
| Additional Models | Groq `gpt-oss-20b`, Qwen 3.6 27B, NVIDIA GLM 5.2 |
| Vector Database | AstraDB |
| Embeddings | Google `gemini-embedding-2` |
| Dense Retrieval | AstraDB Similarity / MMR |
| Lexical Retrieval | BM25 |
| Retrieval Fusion | LangChain `EnsembleRetriever` |
| Reranking | Cohere `rerank-english-v3.0` |
| Semantic Cache | Redis + LiteLLM |
| Cache Embeddings | Cohere Embed English v3 |
| External Search | MCP |
| External Tools | Tavily, Wikipedia, arXiv |
| Guardrails | LangChain Agents + Middleware |
| PII Protection | `PIIMiddleware` |
| Cost Tracking | LiteLLM + Custom Callback |
| Observability | LangSmith + Python Logging |
| UI | Streamlit |
| Graph Debugging | LangGraph Studio |
| Package Management | uv |
| Testing | pytest / Component Tests |

---

# Project Structure

```text
enhanced-multi-document-search-rag/
│
├── data/
│   └── sample data / local fixtures
│
├── legacy/
│   └── previous implementations
│
├── src/
│   ├── config/
│   │   ├── cost_callback.py
│   │   ├── litellm_config.yaml
│   │   ├── llmgateway_config.py
│   │   └── mcp_config.py
│   │
│   ├── document_ingestion/
│   │   ├── document_processor.py
│   │   └── chunker.py
│   │
│   ├── graph_builder/
│   │   └── adaptive_graph_builder.py
│   │
│   ├── nodes/
│   │   ├── adaptive_node.py
│   │   ├── guardrails.py
│   │   └── schema.py
│   │
│   ├── prompts/
│   │   └── rag_prompts.py
│   │
│   ├── state/
│   │   └── adaptive_state.py
│   │
│   ├── vectorstore/
│   │   └── vectorstore.py
│   │
│   └── studio_graph.py
│
├── tests/
│   ├── test_adaptive_node.py
│   ├── test_chunker.py
│   ├── test_document_processor.py
│   ├── test_graph_builder.py
│   └── test_vectorstore.py
│
├── .python-version
├── langgraph.json
├── main.py
├── streamlit_app.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

# Setup

## Requirements

- Python `>=3.13`
- uv or pip
- AstraDB account
- Google API key
- Groq API key
- Cohere API key
- Tavily API key
- Redis instance
- Node.js / `npx` for Tavily MCP
- `uvx` for Wikipedia / arXiv MCP
- LiteLLM-compatible provider configuration

---

## Clone

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git

cd generative-ai-projects/langchain-projects/enhanced-multi-document-search-rag
```

---

## Install

Using uv:

```bash
uv sync
```

Or:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
# LLM Providers
GROQ_API_KEY=

# Google Generative AI
GOOGLE_API_KEY=

# NVIDIA NIM
NVIDIA_API_KEY=
NVIDIA_API_BASE=

# LiteLLM
LITELLM_MASTER_KEY=
LITELLM_BASE_URL=http://localhost:4000

# AstraDB
ASTRA_DB_API_KEY=
ASTRA_DB_API_ENDPOINT=
ASTRA_DB_API_REGION=

# Cohere
COHERE_API_KEY=

# Tavily
TAVILY_API_KEY=

# Redis Semantic Cache
REDIS_HOST=
REDIS_PORT=6379
REDIS_PASSWORD=

# Optional LangSmith
LANGSMITH_TRACING=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

---

# Start LiteLLM

```bash
litellm --config src/config/litellm_config.yaml --port 4000
```

---

# Run Streamlit

```bash
streamlit run streamlit_app.py
```

Then:

1. Choose Web URLs, File Uploads, or Both.
2. Select a chunking strategy.
3. Configure chunk size and overlap.
4. Select Similarity Search or MMR.
5. Build the RAG database.
6. Ask a question.
7. Inspect the generated answer, retrieved sources, citations, latency, and estimated cost.

---

# Run LangGraph Studio

```bash
langgraph dev
```

---

# Run Tests

```bash
pytest
```

---

# Design Principles

This project is built around several engineering principles:

### 1. Guard the boundaries

User/application input is inspected before entering the main workflow, while generated output is inspected before being returned.

### 2. Retrieve before trusting

Retrieved documents are graded rather than blindly passed to the generator.

### 3. Correct before escalating

The system attempts query rewriting and regeneration within bounded limits before escalating to external search.

### 4. Use the right retrieval strategy

Dense retrieval handles semantic similarity, BM25 handles lexical matching, and reranking improves the final candidate ordering.

### 5. Treat external tools as untrusted

MCP tool arguments and results are inspected through middleware.

### 6. Control LLM infrastructure centrally

LiteLLM provides a gateway for model configuration, retries, fallbacks, rate limiting, caching, and cost tracking.

### 7. Make operations observable

Latency, cost, retrieved context, citations, logs, and optional tracing provide visibility into workflow behavior.

### 8. Evaluate before optimizing

The current project focuses on building the adaptive architecture first. RAGAS-based and broader RAG evaluation is currently being learned and will be added after the evaluation methodology is properly understood.

---

# Future Improvements

Planned improvements include:

- RAGAS-based evaluation
- Retrieval and generation benchmark datasets
- Quantitative comparison of retrieval strategies
- Context precision / recall evaluation
- Faithfulness evaluation
- Answer relevance evaluation
- Broader integration testing
- Additional retrieval experiments
- More systematic latency and cost analysis

---

# Project Status

**Current status: Active development**

The core adaptive RAG architecture, hybrid retrieval, reranking, MCP external search, guardrails, LiteLLM gateway, Redis semantic caching, cost tracking, reliability controls, Streamlit interface, LangGraph Studio integration, and component-level testing are implemented.

The next major learning and implementation phase is **systematic RAG evaluation with RAGAS and related evaluation methodologies**.

------------------------------------------------------------------------

# License

This project is part of the `generative-ai-projects` repository and is licensed under the **MIT License**.

See the repository-level [LICENSE](../../LICENSE) file for the complete license text.
