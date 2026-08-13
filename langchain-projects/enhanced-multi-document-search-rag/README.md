# Enhanced Multi-Document Search RAG

> Adaptive, self-correcting RAG system built with LangGraph, hybrid
> retrieval, configurable Similarity/MMR search, Cohere reranking,
> MCP-based external search, LiteLLM model routing, Redis semantic
> caching, multi-layer guardrails, and component-level testing.

An advanced Retrieval-Augmented Generation system designed to go beyond
a fixed `retrieve → generate` pipeline.

The system dynamically analyzes each query, retrieves and evaluates
supporting documents, rewrites weak queries, verifies generated answers,
escalates to external search when local retrieval is insufficient, and
applies security controls throughout the workflow.

The application is exposed through a Streamlit interface, while the
underlying workflow can also be inspected independently through
LangGraph Studio.

------------------------------------------------------------------------

## Why This Project?

A basic RAG system generally looks like:

``` text
Question
   ↓
Retrieve documents
   ↓
Generate answer
```

This project treats retrieval and generation as potential failure points
and introduces explicit mechanisms for retrieval grading, query
rewriting, retrieval diversity, hybrid search, reranking, hallucination
detection, answer relevance checking, external knowledge escalation,
safety, caching, cost tracking, and bounded retries.

------------------------------------------------------------------------

# Architecture

``` text
                           User Query
                               │
                               ▼
                    Input Security Check
                               │
                    ┌──────────┴──────────┐
                    │                     │
                 Blocked                Safe
                    │                     │
                    ▼                     ▼
                   END              Query Analyzer
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                         Local RAG            External Search
                              │                     │
                              ▼                     ▼
                       Dense Retrieval           MCP Agent
                              │                ┌────┼────┐
                              │                ▼    ▼    ▼
                              │              Tavily Wiki arXiv
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Similarity               MMR
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         BM25 Retrieval
                              │
                              ▼
                       Ensemble Retrieval
                         0.7 Dense
                         0.3 BM25
                              │
                              ▼
                       Cohere Reranking
                              │
                              ▼
                       Document Grader
                              │
                    ┌─────────┴─────────┐
                    │                   │
                Relevant              Weak
                    │                   │
                    ▼                   ▼
             Answer Generator     Query Rewriter
                    │                   │
                    ▼                   └────► Retrieval
          Hallucination Detector
                    │
              ┌─────┴─────┐
              │           │
           Grounded    Ungrounded
              │           │
              ▼           ▼
      Answer Relevance  Regenerate
              │
        ┌─────┴─────┐
        │           │
     Relevant      Weak
        │           │
        ▼           └────► Query Rewriter
 Output Security Check
        │
        ▼
     Final Answer
```

All self-correction loops are bounded.

``` text
MAX_REWRITES = 3
MAX_GENERATIONS = 3
```

------------------------------------------------------------------------

# Key Features

## 1. Adaptive LangGraph RAG

The workflow is implemented as a conditional LangGraph `StateGraph`.

Current graph nodes:

-   `input_query_security_check`
-   `query_analyzer`
-   `vector_search`
-   `documents_grader`
-   `query_rewriter`
-   `answer_generator`
-   `hallucination_detector`
-   `answer_relevance_grader`
-   `external_search`
-   `output_answer_security_check`

Conditional routing uses input safety, query classification, retrieval
quality, hallucination detection, answer relevance, retry limits, and
external-search escalation.

------------------------------------------------------------------------

## 2. Hybrid Retrieval

The local retrieval system combines dense semantic retrieval and lexical
retrieval.

``` text
                         Query
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Dense Retrieval                 BM25
       AstraDB Vector Store       Keyword Retrieval
              │
       ┌──────┴──────┐
       ▼             ▼
 Similarity          MMR
 Search              Search
       │             │
       └──────┬──────┘
              ▼
        EnsembleRetriever
          Dense: 0.7
          BM25:  0.3
              │
              ▼
        Cohere Reranker
              │
              ▼
       Final Context
```

### Dense embeddings

Current model:

``` text
gemini-embedding-2
```

implemented with `GoogleGenerativeAIEmbeddings`.

### Lexical retrieval

BM25 is implemented with `rank_bm25` through LangChain's
`BM25Retriever`.

### Fusion

``` text
Dense retrieval weight = 0.7
BM25 weight            = 0.3
```

### Reranking

Cohere `rerank-english-v3.0` is applied through
`ContextualCompressionRetriever` after candidate retrieval and fusion.

------------------------------------------------------------------------

## 3. Similarity Search vs MMR

The Streamlit UI lets the user choose:

-   Similarity Search
-   Maximal Marginal Relevance (MMR)

### Similarity Search

Retrieves documents based primarily on vector similarity to the query.

### MMR

Balances query relevance with diversity among retrieved documents,
helping reduce redundant results.

The selected search type is applied to the AstraDB dense retriever. That
dense retriever is then combined with BM25 and reranked.

This makes it possible to compare:

``` text
Similarity + BM25 + Reranking
```

against:

``` text
MMR + BM25 + Reranking
```

------------------------------------------------------------------------

## 4. Multi-Format Document Ingestion

Supported sources:

-   PDF
-   DOCX
-   TXT
-   Markdown
-   CSV
-   XLSX
-   XLS
-   Web URLs
-   Directories containing supported files

Loaders include `WebBaseLoader`, `PyMuPDFLoader`, `TextLoader`,
`Docx2txtLoader`, `CSVLoader`, and `UnstructuredExcelLoader`.

Documents are enriched with metadata such as source, filename, file
type, loader, and chunk number.

------------------------------------------------------------------------

## 5. Multiple Chunking Strategies

Three strategies are available:

### Recursive

Uses `RecursiveCharacterTextSplitter`.

### Semantic

Uses `SemanticChunker` with the configured Google embedding model.

### Hybrid

Applies semantic chunking followed by recursive splitting.

Markdown documents additionally use `MarkdownHeaderTextSplitter`.

Default configuration:

``` text
Chunk size    = 500
Chunk overlap = 50
```

------------------------------------------------------------------------

## 6. Idempotent Document Ingestion

Each chunk receives a deterministic SHA-256 ID derived from:

``` text
source + chunk content
```

Before insertion, the ID is checked in AstraDB.

``` text
Existing ID?
   │
 ┌─┴─┐
Yes  No
 │    │
Skip Insert
```

This prevents unnecessary duplicate vectors when the same content is
ingested repeatedly.

------------------------------------------------------------------------

## 7. Self-Correcting Retrieval

Retrieved documents are graded before generation.

``` text
Retrieve
   ↓
Document Grader
   ↓
Sufficient?
 ┌─┴──────────────┐
Yes               No
 │                 │
 ▼                 ▼
Generate       Query Rewrite
                  │
                  ▼
               Retrieve
```

The rewrite loop is bounded by `MAX_REWRITES`.

If local retrieval remains insufficient, the workflow can escalate to
external search.

------------------------------------------------------------------------

## 8. Hallucination Detection & Answer Relevance

Generated answers are verified after generation.

``` text
Generate
   ↓
Hallucination Detector
   │
 ┌─┴──────────┐
Grounded   Ungrounded
   │           │
   ▼           ▼
Continue    Regenerate
```

The answer is subsequently checked for relevance to the query. An
irrelevant answer can route the workflow back toward query rewriting and
retrieval.

Generation is bounded by `MAX_GENERATIONS = 3`.

------------------------------------------------------------------------

## 9. Multi-Layer Safety Guardrails

The system uses defense-in-depth.

### Input protection

1.  PII middleware
2.  Deterministic content filtering
3.  LLM-based security classification

### PII protection

Patterns include:

-   email
-   credit card
-   API keys
-   phone numbers
-   IP addresses
-   SSNs

Strategies include:

``` text
redact
mask
block
```

### Tool protection

External MCP calls can be inspected and blocked, with tool failures
handled and unsafe tool results filtered.

### Output protection

``` text
Output Security Agent
        ↓
Deterministic Output Middleware
        ↓
Final Response
```

------------------------------------------------------------------------

## 10. MCP External Search

When local retrieval cannot sufficiently answer a query, the system can
escalate to external search through MCP.

Current integrations:

-   Tavily
-   Wikipedia
-   arXiv

The external search agent uses LangChain's `create_agent` API with
`langchain-mcp-adapters`.

``` text
Local Retrieval
      ↓
Insufficient
      ↓
External Search Agent
      ↓
MCP
 ┌────┼─────┐
 ▼    ▼     ▼
Tavily Wiki arXiv
      │
      ▼
External Answer
      │
      ▼
Citation Extraction
      │
      ▼
Output Security
```

External URLs returned by tool results are surfaced in Streamlit.

------------------------------------------------------------------------

## 11. LiteLLM Gateway

LLM calls are routed through a self-hosted LiteLLM proxy.

Current application model:

``` text
gpt-oss-120b-groq
```

Configured models:

``` text
gpt-oss-120b-groq
gpt-oss-20b-groq
qwen3.6-27b-groq
nvidia-glm-5.2
```

Configured NVIDIA fallback chain:

``` text
nvidia-glm-5.2
      ↓
gpt-oss-120b-groq
      ↓
gpt-oss-20b-groq
```

LiteLLM provides centralized model configuration, retries, timeout
handling, fallback, caching, and cost support.

------------------------------------------------------------------------

## 12. Redis Semantic Caching

Semantic caching is implemented through LiteLLM and Redis.

``` text
Cache type:              redis-semantic
Similarity threshold:    0.85
TTL:                     1800 seconds / 30 minutes
Cache embedding model:   Cohere Embed English v3
```

Caching is `default_off` at the gateway level and explicitly enabled for
selected calls.

``` text
Query
  ↓
Semantic Cache
  │
 ┌┴──────┐
Hit     Miss
 │        │
 ▼        ▼
Cached   LLM
Answer   Call
          │
          ▼
        Cache
```

The goal is to reduce repeated LLM inference, latency, and cost for
sufficiently similar requests.

------------------------------------------------------------------------

## 13. LLM Cost Tracking

Each RAG execution uses a `CostTrackingCallbackHandler`.

It:

1.  Reads LiteLLM cost headers when available.
2.  Falls back to token-based `litellm.completion_cost()`.
3.  Accumulates cost across multiple LLM calls.
4.  Exposes cumulative cost through `total_cost`.

Retries and self-correction calls can therefore contribute to the
displayed request cost.

The UI displays:

-   latency
-   estimated cost

------------------------------------------------------------------------

## 14. Reliability Controls

The system includes:

-   bounded query rewriting
-   bounded generation retries
-   LiteLLM gateway retries
-   application-level retries
-   rate limiting
-   external-search escalation
-   reranker failure fallback
-   graph construction exception handling
-   idempotent ingestion

------------------------------------------------------------------------

## 15. Observability

Current observability includes:

-   Python logging
-   per-query latency
-   estimated LLM cost
-   retrieved-document inspection
-   external citation inspection
-   search history
-   optional LangSmith tracing
-   LangGraph Studio inspection

------------------------------------------------------------------------

## 16. Streamlit Application

The UI provides:

### Inputs

-   Web URLs
-   File uploads
-   Both

### Supported uploads

-   PDF
-   DOCX
-   TXT
-   Markdown
-   CSV
-   XLSX

### Chunking controls

-   Recursive
-   Semantic
-   Hybrid
-   Adjustable chunk size
-   Adjustable overlap

### Retrieval controls

-   Similarity Search
-   MMR

### Query results

-   Generated answer
-   Retrieved source chunks
-   External citations
-   Search history
-   Latency
-   Estimated cost

------------------------------------------------------------------------

## 17. LangGraph Studio

The graph is registered through:

``` text
langgraph.json
```

and exposed through:

``` text
src/studio_graph.py
```

Run:

``` bash
langgraph dev
```

to inspect the graph independently of the Streamlit interface.

------------------------------------------------------------------------

## 18. Unit Tests

Component-level tests are included under `tests/`:

``` text
tests/
├── test_adaptive_node.py
├── test_chunker.py
├── test_document_processor.py
├── test_graph_builder.py
└── test_vectorstore.py
```

Coverage includes adaptive node behavior, security checks, query
analysis, graph construction/routing, chunking, document processing, and
vector store behavior.

------------------------------------------------------------------------

# Technology Stack

  Layer                Technology
  -------------------- --------------------------------------------------
  Orchestration        LangGraph `StateGraph`
  LLM Interface        LangChain `ChatOpenAI`
  LLM Gateway          LiteLLM Proxy
  Application LLM      Groq `gpt-oss-120b`
  Additional Models    Groq `gpt-oss-20b`, Qwen 3.6 27B, NVIDIA GLM 5.2
  Vector Database      AstraDB
  Embeddings           Google `gemini-embedding-2`
  Dense Search         AstraDB Similarity / MMR
  Keyword Search       BM25
  Retrieval Fusion     `EnsembleRetriever`
  Reranking            Cohere `rerank-english-v3.0`
  Semantic Cache       Redis Semantic Cache
  Cache Embeddings     Cohere Embed English v3
  External Search      MCP
  External Tools       Tavily, Wikipedia, arXiv
  Safety               LangChain Agents + Middleware
  PII Protection       `PIIMiddleware`
  Cost Tracking        LiteLLM + custom callback
  Observability        LangSmith + application logging
  UI                   Streamlit
  Graph Debugging      LangGraph Studio
  Package Management   uv
  Testing              Python component tests

------------------------------------------------------------------------

# Project Structure

``` text
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

------------------------------------------------------------------------

# Setup

## Requirements

-   Python `>=3.13`
-   uv or pip
-   AstraDB account
-   Google API key
-   Groq API key
-   Cohere API key
-   Tavily API key
-   Redis instance
-   Node.js / `npx` for Tavily MCP
-   `uvx` for Wikipedia/arXiv MCP
-   LiteLLM-compatible provider configuration

## Clone

``` bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git

cd generative-ai-projects/langchain-projects/enhanced-multi-document-search-rag
```

## Install

Using uv:

``` bash
uv sync
```

Or:

``` bash
pip install -r requirements.txt
```

## Environment Variables

Create `.env`:

``` env
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

## Start LiteLLM

``` bash
litellm --config src/config/litellm_config.yaml --port 4000
```

## Run Streamlit

``` bash
streamlit run streamlit_app.py
```

Then:

1.  Choose Web URLs, File Uploads, or Both.
2.  Select a chunking strategy.
3.  Configure chunk size and overlap.
4.  Select Similarity Search or MMR.
5.  Click **Build RAG Database**.
6.  Ask a question.
7.  Inspect the answer, retrieved sources, citations, latency, and
    estimated cost.

## LangGraph Studio

``` bash
langgraph dev
```

## Tests

If `pytest` is available in the environment:

``` bash
pytest
```

The repository currently focuses on component-level tests. Broader
integration testing and automated RAG evaluation are planned.

------------------------------------------------------------------------

# Current Configuration

  Parameter                  Current Value
  -------------------------- -------------------------------
  Python                     `>=3.13`
  Application LLM            `gpt-oss-120b-groq`
  Embedding Model            `gemini-embedding-2`
  Dense Search               Similarity / MMR
  Reranker                   `rerank-english-v3.0`
  Chunk Size                 `500`
  Chunk Overlap              `50`
  Retrieval K                `4`
  Reranker Top N             `5`
  Dense/BM25 Weights         `0.7 / 0.3`
  Max Query Rewrites         `3`
  Max Generations            `3`
  Temperature                `0.2`
  Top P                      `0.9`
  Max Tokens                 `7000`
  Application Max Retries    `3`
  LiteLLM Gateway Retries    `2`
  Semantic Cache Threshold   `0.85`
  Semantic Cache TTL         `1800 seconds` / `30 minutes`
  Cache Embedding            Cohere Embed English v3

------------------------------------------------------------------------

# Testing Status

### Implemented

-   [x] Adaptive node tests
-   [x] Chunker tests
-   [x] Document processor tests
-   [x] Graph builder tests
-   [x] Vector store tests

### Planned

-   [ ] Integration test suite
-   [ ] CI execution
-   [ ] Automated RAG evaluation
-   [ ] RAGAS evaluation
-   [ ] Retrieval benchmarks
-   [ ] End-to-end regression dataset

------------------------------------------------------------------------

# Roadmap

## Evaluation

-   [ ] Build a representative RAG evaluation dataset
-   [ ] Add RAGAS evaluation
-   [ ] Measure retrieval quality
-   [ ] Measure answer faithfulness
-   [ ] Measure answer relevance
-   [ ] Compare Similarity vs MMR
-   [ ] Compare dense vs hybrid retrieval
-   [ ] Measure reranking improvements
-   [ ] Benchmark latency and cost
-   [ ] Perform retrieval ablation studies

## Engineering

-   [x] Component-level unit tests
-   [ ] Expand integration tests
-   [ ] Add GitHub Actions CI
-   [ ] Add Docker deployment
-   [ ] Add FastAPI API layer
-   [ ] Add production deployment configuration

## Optimization

-   [x] LiteLLM gateway
-   [x] Model routing
-   [x] Provider fallback
-   [x] Redis semantic caching
-   [x] Per-query cost tracking
-   [ ] Cache hit-rate benchmarking
-   [ ] Retrieval latency benchmarking
-   [ ] Quality/cost trade-off analysis

------------------------------------------------------------------------

# Engineering Decisions

This project is also intended as a practical laboratory for
understanding why different RAG components are useful.

### Why BM25?

BM25 provides a lexical retrieval signal that complements semantic
retrieval, particularly for queries where exact terms and lexical
overlap are important.

### Why hybrid retrieval?

Dense and lexical retrieval have different strengths and failure modes.
Combining them provides complementary candidate signals before
reranking.

### Why MMR?

Similarity search can return highly redundant chunks. MMR adds a
relevance-versus-diversity trade-off to the dense retrieval stage.

### Why reranking?

Initial retrieval is optimized for efficiently producing candidates. A
reranker can perform a more focused query-document relevance assessment
over those candidates.

### Why query rewriting?

Weak queries can lead to weak retrieval. Rewriting provides another
attempt to express the user's information need before escalating to
external search.

### Why hallucination detection?

Relevant documents do not guarantee that the generated answer is fully
supported by those documents. A separate grounding check adds another
reliability layer.

### Why external search?

A local document collection cannot answer questions about information it
does not contain. MCP-based search provides an escalation path.

### Why semantic caching?

Semantically similar requests can otherwise trigger repeated LLM calls.
Caching can reduce redundant inference, latency, and cost.

### Why LiteLLM?

A gateway separates application logic from individual model providers
and centralizes routing, fallback, retry, caching, and cost handling.

------------------------------------------------------------------------

# Project Status

## Core system: Functional

The current implementation includes:

-   adaptive LangGraph orchestration
-   multi-format ingestion
-   recursive, semantic, and hybrid chunking
-   Google Gemini embeddings
-   AstraDB retrieval
-   Similarity Search
-   MMR Search
-   BM25
-   hybrid ensemble retrieval
-   Cohere reranking
-   query rewriting
-   retrieval grading
-   hallucination detection
-   answer relevance grading
-   MCP external search
-   Tavily / Wikipedia / arXiv
-   input/output guardrails
-   PII protection
-   tool-call/result filtering
-   bounded retries
-   LiteLLM gateway
-   model routing and fallback
-   Redis semantic caching
-   LLM cost tracking
-   latency reporting
-   LangGraph Studio
-   Streamlit UI
-   component-level tests

The next phase is focused on **measurement and production hardening**:
automated evaluation, retrieval ablation studies, integration testing,
CI/CD, containerization, and API deployment.

------------------------------------------------------------------------

# License

This project is part of the `generative-ai-projects` repository and is licensed under the **MIT License**.

See the repository-level [LICENSE](../../LICENSE) file for the complete license text.
