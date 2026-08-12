# Enhanced Multi-Document Search RAG

> **Adaptive, self-correcting RAG system built with LangGraph, hybrid retrieval, reranking, MCP-based external search, LLM gateway routing, Redis semantic caching, and multi-layer safety guardrails.**

An advanced Retrieval-Augmented Generation system designed to go beyond a fixed `retrieve → generate` pipeline.

The system dynamically analyzes each query, chooses between local document retrieval and external search, grades retrieved context, rewrites weak queries, detects hallucinations, evaluates answer relevance, and escalates to external search when the local knowledge base cannot provide sufficient evidence.

The entire workflow is orchestrated as a conditional **LangGraph `StateGraph`**, with bounded retry loops and centralized LLM access through a **self-hosted LiteLLM gateway**.

---

## ✨ What Makes This RAG Different?

Traditional RAG follows:

```text
Question
   ↓
Retrieve
   ↓
Generate
   ↓
Answer
```

This project uses an adaptive workflow:

```text
                         ┌──────────────────────┐
                         │ Input Security Check │
                         └──────────┬───────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Query Analyzer │
                           └────────┬────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                Vector Search             External Search
                       │                    MCP Tools
                       ▼
               Documents Grader
                       │
              ┌────────┴────────┐
              │                 │
           Relevant          Weak
              │                 │
              ▼                 ▼
        Answer Generator   Query Rewriter
              │                 │
              ▼                 └──────► Vector Search
      Hallucination Detector
              │
       ┌──────┴───────┐
       │              │
    Grounded      Ungrounded
       │              │
       ▼              ▼
 Answer Relevance   Regenerate
       │
       ├──────────────► External Search if retries exhausted
       │
       ▼
 Output Security Check
       │
       ▼
      END
```

Every self-correction loop is bounded to prevent uncontrolled execution.

---

# 🚀 Key Features

## 1. Adaptive LangGraph RAG

The RAG pipeline is implemented as a conditional `StateGraph` rather than a linear chain.

### Current graph nodes

- `input_query_security_check`
- `query_analyzer`
- `vector_search`
- `documents_grader`
- `query_rewriter`
- `answer_generator`
- `hallucination_detector`
- `answer_relevance_grader`
- `external_search`
- `output_answer_security_check`

The graph uses conditional routing to determine what should happen next based on security checks, retrieval quality, hallucination detection, and answer relevance.

Retry loops are bounded using:

- `MAX_REWRITES = 3`
- `MAX_GENERATIONS = 3`

This prevents infinite self-correction loops.

---

# 🔎 2. Hybrid Retrieval

The local retrieval pipeline combines:

```text
                    User Query
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Dense Retrieval          BM25
          AstraDB            Keyword Search
             │                     │
             └──────────┬──────────┘
                        ▼
                 EnsembleRetriever
                    0.7 / 0.3
                        │
                        ▼
                Cohere Reranker
                        │
                        ▼
                Final Documents
```

### Dense retrieval

Uses:

- AstraDB Vector Store
- `BAAI/bge-base-en-v1.5`

### Keyword retrieval

Uses:

- BM25
- `rank_bm25`

### Fusion

```text
Dense retrieval weight:   0.7
BM25 weight:              0.3
```

### Reranking

Cohere `rerank-english-v3.0` is applied through a contextual compression retriever to improve the precision of the final retrieved context.

---

# 📚 3. Multi-Format Document Ingestion

The ingestion pipeline supports:

- PDF
- DOCX
- TXT
- Markdown
- CSV
- XLSX
- XLS
- Web URLs
- Directories containing supported files

Documents are normalized with standardized metadata including source, filename, file type, loader, and chunk information.

---

# ✂️ 4. Multiple Chunking Strategies

The application provides three chunking strategies:

### Recursive

Uses `RecursiveCharacterTextSplitter`.

### Semantic

Uses embedding-based semantic chunking.

### Hybrid

Combines semantic chunking followed by recursive splitting.

The Streamlit interface allows the user to configure:

- Chunk size
- Chunk overlap
- Chunking strategy

Default values:

```text
Chunk size:       500
Chunk overlap:     50
```

---

# ♻️ 5. Idempotent Document Ingestion

The system prevents duplicate vector insertion using deterministic SHA-256 IDs generated from:

```text
source + document content
```

When the same document/chunk is ingested again:

```text
Document
   ↓
SHA-256 ID
   ↓
Already exists?
   ├── Yes → Skip
   └── No  → Embed + Insert
```

This makes repeated ingestion safe and avoids unnecessary duplicate vectors.

---

# 🧠 6. Self-Correcting Retrieval

Retrieved documents are evaluated before generation.

If the retrieved context is insufficient:

```text
Retrieve
   ↓
Grade Documents
   ↓
Insufficient
   ↓
Rewrite Query
   ↓
Retrieve Again
```

After the maximum number of rewrites is reached, the system escalates to external search instead of repeatedly querying the same local knowledge base.

---

# 🛡️ 7. Multi-Layer Safety Guardrails

Safety is implemented at multiple points in the system.

### Input protection

The system performs:

- deterministic keyword filtering
- LLM-based security classification
- PII detection and protection

### PII protection

PII middleware currently handles patterns such as:

- Email
- Credit card
- API keys
- Phone numbers
- IP addresses
- SSNs

Different fields use different strategies including:

```text
redact
mask
block
```

### Tool protection

External MCP tool calls are intercepted so that:

1. Tool arguments can be inspected.
2. Unsafe requests can be blocked.
3. Tool failures are handled safely.
4. Unsafe tool results can be redacted before reaching the model.

### Output protection

Generated answers pass through an output security agent followed by deterministic post-processing.

---

# 🌐 8. External Search via MCP

When local retrieval cannot sufficiently answer a question, the system can escalate to external search.

MCP integrations currently include:

- Tavily
- Wikipedia
- arXiv

The external search agent is built using LangChain's modern `create_agent` API and `langchain-mcp-adapters`.

The workflow is:

```text
Local Retrieval
      ↓
Insufficient?
      ↓
External Search Agent
      ↓
MCP Tools
 ┌────┼────┐
 ▼    ▼    ▼
Tavily Wiki arXiv
      │
      ▼
External Answer
      │
      ▼
Citation Extraction
      │
      ▼
Output Security Check
```

External URLs returned by tool results are extracted and surfaced in the Streamlit interface.

---

# 🚪 9. LiteLLM Gateway

All application LLM calls are routed through a self-hosted LiteLLM proxy.

This provides a centralized abstraction layer between the application and model providers.

```text
                    Application
                         │
                         ▼
                  LiteLLM Gateway
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           Groq                  NVIDIA NIM
       Configured models        Configured model
```

The current application default model is:

```text
gpt-oss-120b-groq
```

Additional configured models include:

- `gpt-oss-20b-groq`
- `qwen3.6-27b-groq`
- `nvidia-glm-5.2`

The LiteLLM router also contains retry and fallback configuration.

This architecture makes it possible to change providers/models without tightly coupling the application to a single vendor.

---

# 💾 10. Redis Semantic Caching

The project now includes **Redis-backed semantic caching through LiteLLM**.

The current configuration uses:

```text
Cache type:              redis-semantic
Similarity threshold:    0.85
TTL:                     1800 seconds
Cache embedding model:   Cohere Embed English v3
```

`1800 seconds` = **30 minutes**.

The cache is configured as `default_off` at the gateway level and explicitly enabled for selected LLM calls that benefit from caching.

For example, answer generation uses:

```text
use-cache = true
TTL = 1800
```

This allows semantically similar requests to reuse cached responses instead of repeatedly invoking the LLM.

---

# 💰 11. Per-Query Cost Tracking

Every RAG execution creates a `CostTrackingCallbackHandler`.

The callback:

1. Reads LiteLLM cost headers when available.
2. Falls back to token-based `litellm.completion_cost()` calculation.
3. Accumulates cost across multiple LLM calls.
4. Returns the cumulative cost as `total_cost`.

This means retries and additional model calls can contribute to the displayed request cost.

The Streamlit UI displays:

```text
Latency
Estimated Cost
```

for each query.

---

# 📊 12. Observability

The application provides:

- Python logging
- Per-query latency
- Per-query estimated cost
- Retrieved document inspection
- External citation inspection
- Search history
- Optional LangSmith tracing

This makes it possible to inspect not only the final answer but also the supporting retrieval context and execution characteristics.

---

# 🖥️ 13. Streamlit Interface

The Streamlit application provides:

### Data ingestion

- Web URLs
- File uploads
- Both simultaneously

### Supported uploads

```text
PDF
DOCX
TXT
MD
CSV
XLSX
```

### Retrieval configuration

- Recursive chunking
- Semantic chunking
- Hybrid chunking
- Adjustable chunk size
- Adjustable overlap

### Query interface

- Question input
- Generated answer
- Retrieved source chunks
- External citations
- Latency
- Estimated cost
- Recent search history

---

# 🔬 14. LangGraph Studio Support

The graph is exposed independently through:

```text
langgraph.json
src/studio_graph.py
```

This allows the RAG workflow to be loaded and inspected using LangGraph Studio / `langgraph dev`.

The graph can therefore be debugged independently from the Streamlit UI.

---

# 🏗️ Architecture Components

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| LLM Interface | LangChain `ChatOpenAI` |
| LLM Gateway | LiteLLM Proxy |
| Primary Application Model | Groq `gpt-oss-120b` |
| Additional Models | Groq `gpt-oss-20b`, Qwen, NVIDIA NIM |
| Vector Database | AstraDB |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Keyword Retrieval | BM25 |
| Retrieval Fusion | `EnsembleRetriever` |
| Reranking | Cohere `rerank-english-v3.0` |
| Semantic Cache | Redis Semantic Cache |
| Cache Embeddings | Cohere Embed English v3 |
| External Search | MCP |
| External Tools | Tavily, Wikipedia, arXiv |
| Guardrails | LangChain Agents + Middleware |
| PII Protection | `PIIMiddleware` |
| Observability | LiteLLM + optional LangSmith |
| UI | Streamlit |
| Graph Debugging | LangGraph Studio |
| Package Management | uv |

---

# 📁 Project Structure

```text
enhanced-multi-document-search-rag/
│
├── data/
│   └── ...                         # Sample data / fixtures
│
├── legacy/
│   └── ...                         # Earlier implementations kept for reference
│
├── src/
│   ├── config/
│   │   ├── cost_callback.py        # Per-run LLM cost tracking
│   │   ├── litellm_config.yaml     # Model routing, fallback and Redis cache
│   │   ├── llmgateway_config.py    # Runtime configuration + LLM client
│   │   └── mcp_config.py           # MCP server/tool configuration
│   │
│   ├── document_ingestion/
│   │   ├── document_processor.py   # Multi-format loading
│   │   └── chunker.py              # Recursive / semantic / hybrid chunking
│   │
│   ├── vectorstore/
│   │   └── vectorstore.py          # AstraDB + BM25 + ensemble + reranking
│   │
│   ├── state/
│   │   └── adaptive_state.py       # LangGraph state schema
│   │
│   ├── nodes/
│   │   ├── adaptive_node.py        # RAG workflow nodes + routing
│   │   ├── guardrails.py           # Security agents + middleware
│   │   └── schema.py               # Structured output schemas
│   │
│   ├── graph_builder/
│   │   └── adaptive_graph_builder.py
│   │
│   ├── prompts/
│   │   └── rag_prompts.py
│   │
│   └── studio_graph.py              # LangGraph Studio entry point
│
├── langgraph.json
├── streamlit_app.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

# ⚙️ Setup

## Requirements

- Python 3.13+
- uv or pip
- Node.js / `npx` for Tavily MCP
- `uvx` for Wikipedia/arXiv MCP servers
- Redis instance
- AstraDB account
- Cohere API key
- Groq API key
- Tavily API key
- LiteLLM-compatible model provider credentials

---

## 1. Clone the repository

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git

cd generative-ai-projects/langchain-projects/enhanced-multi-document-search-rag
```

---

## 2. Install dependencies

Using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## 3. Configure environment variables

Create a `.env` file:

```env
# Groq
GROQ_API_KEY=

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

# Redis semantic cache
REDIS_HOST=
REDIS_PORT=6379
REDIS_PASSWORD=

# Optional LangSmith tracing
LANGSMITH_TRACING=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

---

# 🧠 Start the LiteLLM Gateway

The gateway uses:

```text
src/config/litellm_config.yaml
```

Run:

```bash
litellm --config src/config/litellm_config.yaml --port 4000
```

The configuration contains:

- Model definitions
- Provider routing
- Retry configuration
- Fallback configuration
- Redis semantic caching
- Cache similarity threshold
- Cache TTL

---

# ▶️ Run the Streamlit Application

In a separate terminal:

```bash
streamlit run streamlit_app.py
```

Then:

1. Select URLs, files, or both.
2. Choose a chunking strategy.
3. Configure chunk size and overlap.
4. Click **Build RAG Database**.
5. Enter a question.
6. Inspect the generated answer, retrieved sources, citations, latency, and estimated cost.

---

# 🧪 LangGraph Studio

To inspect the graph independently:

```bash
langgraph dev
```

The graph is registered through:

```text
langgraph.json
```

and exposed through:

```text
src/studio_graph.py
```

---

# 🔄 End-to-End Query Flow

```text
User Query
    │
    ▼
Input Security Guardrail
    │
    ├── Blocked ───────────────► END
    │
    ▼
Query Analyzer
    │
    ├── External Search ───────► MCP Search
    │
    ▼
Vector Search
    │
    ▼
Hybrid Retrieval
    │
    ├── AstraDB Vector Search
    └── BM25
    │
    ▼
Ensemble Retrieval
    │
    ▼
Cohere Reranking
    │
    ▼
Document Grader
    │
    ├── Insufficient
    │       │
    │       ▼
    │   Query Rewriter
    │       │
    │       └──────────────► Vector Search
    │
    ▼
Answer Generator
    │
    ▼
Redis Semantic Cache
    │
    ▼
Hallucination Detector
    │
    ├── Ungrounded ─────────► Regenerate
    │
    ▼
Answer Relevance Grader
    │
    ├── Irrelevant ────────► Query Rewriter
    │
    ▼
Output Security Guardrail
    │
    ▼
Final Answer
```

If local retrieval repeatedly fails, the workflow escalates to MCP-based external search.

---

# 🛡️ Safety Architecture

The system applies defense-in-depth rather than relying on a single LLM safety prompt.

```text
User Input
    │
    ▼
PII Middleware
    │
    ▼
Deterministic Content Filter
    │
    ▼
Input Security Agent
    │
    ▼
RAG / MCP Workflow
    │
    ▼
Tool Call Filtering
    │
    ▼
Tool Result Filtering
    │
    ▼
Output Security Agent
    │
    ▼
Deterministic Output Middleware
    │
    ▼
Final Response
```

This protects both the normal RAG path and the external tool path.

---

# 📌 Current Configuration

| Parameter | Current Value |
|---|---|
| Python | `>=3.13` |
| Embedding Model | `BAAI/bge-base-en-v1.5` |
| Default LLM | `gpt-oss-120b-groq` |
| Reranker | `rerank-english-v3.0` |
| Chunk Size | `500` |
| Chunk Overlap | `50` |
| Retrieval K | `4` |
| Reranker Top N | `5` |
| Dense/BM25 Weights | `0.7 / 0.3` |
| Max Query Rewrites | `3` |
| Max Generations | `3` |
| Temperature | `0.2` |
| Max Tokens | `7000` |
| Cache Type | Redis Semantic |
| Cache Similarity Threshold | `0.85` |
| Cache TTL | `1800 seconds` / `30 minutes` |

---

# 🚧 Roadmap

The core adaptive RAG, retrieval, guardrails, MCP integration, gateway routing, cost tracking, and semantic caching functionality is implemented.

### Evaluation

- [ ] Build automated RAG evaluation dataset
- [ ] Add RAGAS evaluation pipeline
- [ ] Measure retrieval and generation quality
- [ ] Add baseline vs hybrid vs reranked vs adaptive comparisons
- [ ] Add latency and cost benchmarking

### Engineering

- [ ] Add automated test suite
- [ ] Add graph routing tests
- [ ] Add GitHub Actions CI pipeline
- [ ] Add Docker deployment
- [ ] Add FastAPI API layer

### Optimization

- [x] Redis semantic caching
- [ ] Cache hit-rate monitoring
- [ ] Retrieval latency optimization
- [ ] Cost/latency benchmarking

---

# 🎯 Project Goals

This project focuses on demonstrating practical GenAI engineering concepts rather than only building a basic chatbot.

The main engineering goals are:

- Build an adaptive RAG workflow.
- Improve retrieval using hybrid search and reranking.
- Make retrieval self-correcting.
- Reduce hallucinations through verification.
- Escalate intelligently when local knowledge is insufficient.
- Integrate external knowledge through MCP.
- Protect user input, tool calls, tool results, and generated output.
- Decouple application code from LLM providers through LiteLLM.
- Track LLM usage cost.
- Reduce repeated LLM calls through semantic caching.
- Make the workflow inspectable through LangGraph Studio.

---

# 📚 Technologies

```text
Python
LangChain
LangGraph
LangChain Agents
AstraDB
BM25
Cohere
Sentence Transformers
LiteLLM
Redis
MCP
Tavily
Wikipedia
arXiv
LangSmith
Streamlit
Pydantic
uv
```

---

# 👨‍💻 Project Status

**Core system: Functional**

The current implementation includes the adaptive LangGraph workflow, hybrid retrieval, Cohere reranking, multi-format ingestion, idempotent indexing, query rewriting, hallucination detection, answer relevance grading, MCP external search, safety guardrails, PII protection, LiteLLM routing, Redis semantic caching, cost tracking, Streamlit UI, and LangGraph Studio integration.

The remaining work is primarily focused on **evaluation, testing, CI/CD, containerization, and API deployment** rather than adding more core RAG functionality.