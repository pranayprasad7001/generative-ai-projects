# Enhanced Multi-Document Search RAG

> 🚧 **Status: In Progress.** The adaptive RAG pipeline, guardrails, and LLM gateway (with cost tracking) are built and working. Response caching and a formal evaluation suite (RAGAS) are actively being built next. See [Roadmap](#roadmap--project-status) below.

An **adaptive, self-correcting RAG system** that goes beyond a single retrieve-then-generate pass. It grades its own retrieval quality, rewrites queries that don't retrieve well, checks its own answers for hallucination and relevance, and falls back to external search (Tavily, Wikipedia, arXiv via MCP) when local documents can't answer the question — all behind input/output safety guardrails and routed through a self-hosted LLM gateway with automatic provider fallback and per-query cost tracking.

## Architecture

The system is built as a `LangGraph` `StateGraph` with conditional routing at key decision and quality checkpoints, rather than a fixed linear pipeline:

```
START
  └─▶ input_query_security_check ──(blocked)──▶ END
          │ (passed)
          ▼
     query_analyzer  (routes to vector search or external search)
          │
   ┌──────┴──────┐
   ▼             ▼
vector_search   external_search ──────────────────────┐
   │                                                    │
   ▼                                                    │
documents_grader ──(insufficient, retries exhausted)───▶│
   │ (sufficient)        │ (insufficient)                │
   │                     ▼                                │
   │              query_rewriter ──▶ vector_search         │
   ▼                                                        │
answer_generator                                            │
   │                                                         │
   ▼                                                         │
hallucination_detector ──(ungrounded, retries exhausted)────▶│
   │ (grounded)          │ (ungrounded)                       │
   │                     ▼                                    │
   │              answer_generator (retry)                    │
   ▼                                                           │
answer_relevance_grader ──(off-target, retries exhausted)─────▶│
   │ (relevant)          │ (off-target)                        │
   │                     ▼                                     │
   │              query_rewriter                               │
   ▼                                                           │
output_answer_security_check ◀───────────────────────────────┘
   │
   ▼
  END
```

Every retry loop is bounded (`MAX_REWRITES`, `MAX_GENERATIONS` in config) and escalates to external search rather than looping forever or forcing an answer from documents already judged insufficient. Every LLM call in every node runs through the LiteLLM gateway, so cost is accumulated across the whole run — including retries and external-search agent calls — and surfaced back to the caller as `total_cost`.

The graph is also exposed as a standalone `agent` graph (`src/studio_graph.py`) via `langgraph.json`, so it can be loaded, visualized, and stepped through node-by-node in **LangGraph Studio / `langgraph dev`** independently of the Streamlit app.

## Key Features

**Hybrid Retrieval**
- `EnsembleRetriever` combining dense vector search (AstraDB) with BM25 keyword search, weighted 0.7 / 0.3
- Cohere reranking (`rerank-english-v3.0`) on top of the fused candidate set for precision

**Idempotent Ingestion**
- SHA-256 content hashing per chunk before insert, so re-ingesting the same source never creates duplicate vectors
- Multi-format loaders: PDF, DOCX, TXT, MD, CSV, XLSX, and web URLs, with per-loader metadata standardization
- Three chunking strategies: recursive, semantic, and hybrid

**Self-Correcting Generation**
- Retrieval grading (are the retrieved docs actually relevant and sufficient?)
- Query rewriting when retrieval is weak, with bounded retry count
- Hallucination detection (is the answer grounded in the retrieved context?)
- Answer relevance grading (does the answer actually address the question?)
- Automatic fallback to external search (Tavily / Wikipedia / arXiv via MCP) when local retrieval can't satisfy the question after retries, with source URLs parsed out of tool results and returned as citations

**LLM Gateway & Cost Tracking**
- All model calls route through a self-hosted **LiteLLM proxy**, decoupling the app from any single provider
- Primary model: NVIDIA NIM (`glm-5.2`), with automatic router-level fallback to Groq (`gpt-oss-120b`, then `gpt-oss-20b`) on failure, plus configurable retries and timeouts
- Custom model pricing registered with LiteLLM for accurate cost calculation on gateway-routed models
- A `CostTrackingCallbackHandler` accumulates per-run cost by reading LiteLLM's cost headers where available, falling back to token-based `completion_cost` calculation otherwise — surfaced in the UI alongside latency for every query

**Guardrails**
- Deterministic keyword pre-filter on input (blocks obvious bad-faith queries *before* spending an LLM call)
- LLM-based input/output security agents for more nuanced checks
- PII middleware across email, credit card, phone, IP, SSN, and API-key patterns with per-field strategies (redact / mask / block)
- Tool-call wrapping so external search results are also filtered before re-entering the conversation

## Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (`StateGraph`, conditional edges, `InMemorySaver` checkpointing) |
| LLM Gateway | Self-hosted LiteLLM proxy — NVIDIA NIM `glm-5.2` primary, Groq `gpt-oss-120b` / `gpt-oss-20b` as router fallbacks |
| Vector store | AstraDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | Cohere `rerank-english-v3.0` |
| Keyword retrieval | BM25 (`rank_bm25`) |
| External search | MCP (Tavily, Wikipedia, arXiv) via `langchain-mcp-adapters` |
| Cost/observability | LiteLLM cost headers + `completion_cost` fallback; optional LangSmith tracing |
| UI | Streamlit |
| Dev tooling | LangGraph Studio (`langgraph.json` / `studio_graph.py`) |

## Project Structure

```
enhanced-multi-document-search-rag/
├── src/
│   ├── config/             # Env config, LiteLLM gateway client, cost
│   │                        # tracking callback, litellm_config.yaml
│   │                        # (model routing/fallback rules), MCP tool manager
│   ├── document_ingestion/ # Loaders + chunking strategies
│   ├── vectorstore/        # AstraDB + hybrid retriever + reranking
│   ├── state/               # AdaptiveRAGState (Pydantic)
│   ├── nodes/               # Graph nodes: analyzer, grader, rewriter,
│   │                        # generator, hallucination/relevance checks,
│   │                        # guardrails
│   ├── graph_builder/       # StateGraph assembly and routing
│   ├── prompts/             # System prompts for each node
│   └── studio_graph.py      # Entry point for LangGraph Studio / langgraph dev
├── legacy/                  # Earlier linear/agentic implementations,
│                            # kept for reference — not used by the app
├── langgraph.json           # LangGraph Studio graph registration
├── streamlit_app.py         # UI entry point (uses the adaptive graph)
└── data/                    # Sample docs and test fixtures
```

## Setup

**Requirements:** Python 3.13+

1. Clone the repo and install dependencies:
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

2. Create a `.env` file with:
   ```
   # Model providers behind the gateway
   NVIDIA_API_KEY=
   NVIDIA_API_BASE=
   GROQ_API_KEY=

   # LiteLLM gateway
   LITELLM_MASTER_KEY=
   LITELLM_BASE_URL=http://localhost:4000

   # Retrieval & tools
   COHERE_API_KEY=
   TAVILY_API_KEY=
   ASTRA_DB_API_KEY=
   ASTRA_DB_API_ENDPOINT=
   ASTRA_DB_API_REGION=

   # Optional tracing
   LANGSMITH_TRACING=
   LANGSMITH_API_KEY=
   LANGSMITH_PROJECT=
   ```

3. Start the LiteLLM gateway (uses `src/config/litellm_config.yaml` for model routing and fallback rules):
   ```bash
   litellm --config src/config/litellm_config.yaml --port 4000
   ```

4. In a separate terminal, run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

   From the sidebar, ingest documents via file upload or URLs, then ask questions in the main panel. Each answer is returned with retrieved source chunks, external citations (when the external-search path is used), latency, and estimated cost.

   Alternatively, run `langgraph dev` to load the same graph in LangGraph Studio for node-by-node debugging.

## Roadmap / Project Status

### ✅ Completed

- [x] Multi-format document ingestion
- [x] Recursive, semantic, and hybrid chunking
- [x] Idempotent, hash-based ingestion
- [x] Hybrid retrieval (Vector + BM25)
- [x] Cohere reranking
- [x] LangGraph StateGraph orchestration
- [x] Adaptive query routing
- [x] Retrieval grading
- [x] Query rewriting with bounded retries
- [x] Hallucination detection
- [x] Answer relevance grading
- [x] External-search fallback with citation extraction
- [x] MCP integration (Tavily, Wikipedia, arXiv)
- [x] Input security guardrails
- [x] Output security guardrails
- [x] PII protection
- [x] Tool-call / tool-result filtering
- [x] Bounded retry and escalation logic
- [x] LLM gateway (LiteLLM proxy) with provider fallback and retries
- [x] Per-run cost tracking and latency reporting
- [x] LangGraph Studio integration
- [x] Streamlit interface

### 🚧 Planned / In Progress

- [ ] Exact-match and semantic caching
- [ ] RAGAS evaluation suite
- [ ] Automated evaluation dataset
- [ ] Test suite + CI pipeline
- [ ] Docker/containerized deployment
- [ ] FastAPI API layer (dependency added, endpoints not yet implemented)

---

**This project is still in progress** — the retrieval, self-correction, and gateway/cost-tracking pipeline is functional end-to-end, but production-hardening (caching, evals, tests, containerization, a real API layer) is ongoing.
