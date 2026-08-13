# Generative AI Projects

A growing portfolio of practical **Generative AI and AI Engineering projects** built with modern LLM frameworks, RAG architectures, AI agents, tool calling, LangGraph, CrewAI, vector databases, MCP, LLM gateways, guardrails, semantic caching, observability, and cost tracking.

The repository documents a progression from foundational LLM applications toward increasingly **adaptive, reliable, safety-aware, and production-oriented AI systems**.

Each project is independently structured with its own README covering architecture, setup, implementation details, and technology choices.

---

## 🧭 What This Repository Covers

- LLM Applications
- Retrieval-Augmented Generation (RAG)
- Adaptive / Self-Correcting RAG
- Hybrid Search & Reranking
- Similarity Search & MMR
- AI Agents & Tool Calling
- LangGraph Workflows
- Multi-Agent Systems with CrewAI
- MCP-based Tool Integration
- Conversational AI
- SQL Agents
- Local LLM Applications
- Semantic Caching
- LLM Gateways & Model Routing
- Safety Guardrails & PII Protection
- LLM Cost Tracking
- LangSmith Observability
- Streamlit & Gradio Applications
- Component-level testing

The goal is not simply to demonstrate that an LLM can generate an answer, but to explore how GenAI systems can be made **grounded, reliable, observable, secure, cost-aware, testable, and maintainable**.

---

# 🚀 Featured Projects

## 🔥 Enhanced Multi-Document Search RAG

**Adaptive, self-correcting RAG system and the most advanced project in this repository.**

It combines LangGraph orchestration, hybrid retrieval, configurable Similarity/MMR search, Cohere reranking, MCP external search, LiteLLM routing, Redis semantic caching, multi-layer guardrails, PII protection, cost tracking, and component-level testing.

### Highlights

- Conditional LangGraph `StateGraph`
- Multi-format document ingestion
- Recursive, semantic, and hybrid chunking
- Google `gemini-embedding-2`
- AstraDB vector retrieval
- Similarity Search
- Maximal Marginal Relevance (MMR)
- BM25 keyword retrieval
- Dense + lexical ensemble retrieval
- Cohere reranking
- Query rewriting
- Document relevance grading
- Hallucination detection
- Answer relevance grading
- MCP-based external search
- Tavily, Wikipedia, and arXiv
- Redis semantic caching
- LiteLLM gateway
- Model routing and provider fallback
- Per-query latency and cost tracking
- PII middleware
- Input/output guardrails
- Tool-call and tool-result filtering
- LangGraph Studio
- Streamlit interface
- Component-level unit tests

**Status:** Core system functional; automated evaluation, integration testing, CI/CD, and deployment hardening remain in progress.

👉 [View Enhanced Multi-Document Search RAG](./langchain-projects/enhanced-multi-document-search-rag)

---

## 🤖 Agentic Search Engine

A tool-calling search agent that dynamically uses external search capabilities to answer user questions.

### Highlights

- LangChain agents
- Tool calling
- Live web search
- Conversation history
- Streaming responses
- Model selection
- Temperature control
- Tool activity visibility
- Streamlit interface

**Status:** Deployed

👉 [View Agentic Search Engine](./langchain-projects/agentic-search-engine)

🚀 [Live Demo](https://agentic-search-engine-with-exa.streamlit.app)

---

## 🗄️ SQL Chat Assistant

Natural-language interface for querying SQL databases through an agent.

### Highlights

- Natural language → SQL
- SQLite / MySQL
- Tool-calling agent
- Read-only database access
- SQL safety controls
- Prompt-injection awareness
- Streamlit interface

**Status:** Deployed

👉 [View SQL Chat Assistant](./langchain-projects/sql-chat-assistant)

🚀 [Live Demo](https://sql-db-chat-assistant.streamlit.app)

---

## 💬 Conversational PDF Assistant

A conversational RAG application for interacting with uploaded PDF documents while maintaining conversation context.

### Highlights

- PDF ingestion
- Vector retrieval
- Conversation history
- History-aware retrieval
- Follow-up question handling
- Source-aware answers
- Streamlit interface

**Status:** Deployed

👉 [View Conversational PDF Assistant](./langchain-projects/conversational-pdf-assistant)

🚀 [Live Demo](https://conversational-pdf-assistant.streamlit.app)

---

## 📚 AstraDB Multi-Document RAG Hub

Multi-document RAG application using AstraDB as the vector database.

### Highlights

- Multiple PDF ingestion
- AstraDB vector store
- Hugging Face inference
- Session-isolated indexing
- Retrieval-based question answering
- Streamlit interface

**Status:** Deployed

👉 [View AstraDB Multi-Doc RAG Hub](./langchain-projects/astradb-multidoc-rag-hub)

🚀 [Live Demo](https://astradb-multidoc-rag.streamlit.app)

---

## 📄 Research Paper Assistant

RAG application for querying research papers using persistent vector storage.

### Highlights

- Research paper ingestion
- ChromaDB
- Persistent vector storage
- Retrieval-based question answering
- Visible source chunks
- Streamlit interface

**Status:** Deployed

👉 [View Research Paper Assistant](./langchain-projects/research-paper-assistant-rag)

🚀 [Live Demo](https://interactive-research-paper-assistant.streamlit.app)

---

## 🧮 Math & Reasoning Agent

An agent that dynamically selects tools depending on the user's question.

### Capabilities

- Calculator
- Reasoning workflow
- Wikipedia lookup
- ReAct-style tool selection
- Streamlit interface

**Status:** Deployed

👉 [View Math & Reasoning Agent](./langchain-projects/math-reasoning-agent)

🚀 [Live Demo](https://math-reasoning-agent.streamlit.app)

---

## 🧠 Intelligent Query Router

A LangGraph-based routing system that classifies a question and dynamically selects the appropriate information source.

```text
                    User Query
                        │
                        ▼
                 Query Classification
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          AstraDB   Wikipedia    arXiv
             │          │          │
             └──────────┼──────────┘
                        ▼
                  Grounded Answer
```

### Highlights

- LangGraph
- Structured LLM output
- Dynamic routing
- AstraDB retrieval
- Wikipedia
- arXiv
- Streaming responses
- Grounded generation

**Status:** Runs locally

👉 [View Intelligent Query Router](./langchain-projects/intelligent-query-router)

---

## 💻 Multi-Language Code Assistant

A local coding assistant powered by Ollama.

### Highlights

- Local LLM inference
- Ollama
- Gradio
- Offline operation
- Multiple programming languages
- Iterative UI improvements

**Status:** Runs locally

👉 [View Multi-Language Code Assistant](./langchain-projects/multi-language-code-assistant)

---

## 🌐 URL Content Summarizer

Summarizes long-form web content and YouTube transcripts.

The application selects different summarization strategies depending on content length.

### Highlights

- URL ingestion
- YouTube transcript processing
- Stuff chain
- Map-reduce chain
- Automatic strategy selection
- Streamlit interface

**Status:** Deployed

👉 [View URL Content Summarizer](./langchain-projects/url-content-summarizer)

🚀 [Live Demo](https://url-content-summarizer.streamlit.app)

---

## 💬 Q&A Chatbot with Groq

The foundational project in this repository and an early exploration of LangChain-based LLM applications.

### Highlights

- Prompt templates
- Groq LLM
- LangChain
- Streamlit
- Basic conversational question answering

**Status:** Deployed

👉 [View Q&A Chatbot](./langchain-projects/qa-chatbot)

🚀 [Live Demo](https://question-answer-chatbot-groq.streamlit.app)

---

# 🤝 CrewAI Projects

## 📰 Agentic Daily Briefing

A multi-agent CrewAI workflow that researches current information and produces a structured daily briefing.

```text
                User Request
                     │
                     ▼
              Researcher Agent
                     │
              Web Search / Tools
                     │
                     ▼
                Research Data
                     │
                     ▼
                 Writer Agent
                     │
                     ▼
              Daily Briefing
```

### Highlights

- CrewAI
- Multi-agent architecture
- Researcher + writer roles
- Web search
- Task delegation
- Structured Markdown output

**Status:** Working end-to-end

👉 [View Agentic Daily Briefing](./crewai-projects/agentic-daily-briefing)

---

# 🏗️ Repository Structure

```text
generative-ai-projects/
│
├── langchain-projects/
│   ├── enhanced-multi-document-search-rag/
│   ├── agentic-search-engine/
│   ├── conversational-pdf-assistant/
│   ├── sql-chat-assistant/
│   ├── astradb-multidoc-rag-hub/
│   ├── research-paper-assistant-rag/
│   ├── intelligent-query-router/
│   ├── math-reasoning-agent/
│   ├── multi-language-code-assistant/
│   ├── url-content-summarizer/
│   └── qa-chatbot/
│
├── crewai-projects/
│   └── agentic-daily-briefing/
│
├── LICENSE
└── README.md
```

**Current portfolio:** 11 LangChain projects + 1 CrewAI project.

---

# 🧩 Technology Coverage

## LLM Frameworks

- LangChain
- LangGraph
- CrewAI

## Models & Providers

- Groq
- NVIDIA NIM
- Google Gemini
- Hugging Face
- Ollama
- Open-source / local models

## RAG & Retrieval

- AstraDB
- ChromaDB
- BM25
- Dense retrieval
- Similarity Search
- MMR
- Hybrid retrieval
- Ensemble retrieval
- Cohere reranking
- Semantic chunking
- Recursive chunking
- Query rewriting
- Retrieval grading

## Agents

- Tool-calling agents
- ReAct agents
- LangGraph workflows
- CrewAI multi-agent workflows
- MCP-based tools

## MCP / External Tools

- Tavily
- Wikipedia
- arXiv
- Web search

## LLM Infrastructure

- LiteLLM
- Model routing
- Provider fallback
- Redis semantic caching
- Cost tracking
- Token usage tracking

## Safety & Reliability

- PII middleware
- Input validation
- Output validation
- Tool-call filtering
- Tool-result filtering
- Guardrail agents
- Hallucination detection
- Answer relevance grading
- Bounded retries

## Interfaces

- Streamlit
- Gradio

## Observability & Testing

- LangSmith
- Application logging
- Latency tracking
- LLM cost tracking
- Component-level unit tests

---

# 📈 Project Progression

The projects represent an intentional progression in GenAI engineering:

```text
                         Foundation
                             │
                             ▼
                      Q&A Chatbot
                             │
                             ▼
                     RAG Applications
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       PDF / Research                  Multi-Document
             RAG                            RAG
              │                             │
              └──────────────┬──────────────┘
                             ▼
                      Query Routing
                             │
                             ▼
                         AI Agents
                             │
                             ▼
                       Tool Calling
                             │
                             ▼
                      Multi-Agent
                             │
                             ▼
                 Adaptive / Agentic RAG
                             │
                             ▼
                  Reliability Engineering
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
         Guardrails       Caching       Cost Tracking
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                         Testing
                             │
                             ▼
                         Evaluation
                             │
                             ▼
                     Deployment / CI
```

The repository is therefore not intended to be a collection of unrelated demos. It documents an ongoing progression from basic LLM applications toward more reliable and production-oriented AI systems.

---

# 🛡️ Reliability & Evaluation Roadmap

Several projects already implement reliability mechanisms including:

- Input/output guardrails
- PII protection
- Prompt-injection-aware workflows
- Retrieval grading
- Hallucination detection
- Answer relevance checks
- Bounded retries
- Tool filtering
- Semantic caching
- LLM cost tracking
- Provider fallback
- Component-level tests

The next focus is making system behavior **measurable and reproducible**.

## Evaluation

- [ ] Build reusable evaluation datasets
- [ ] Add RAGAS evaluations
- [ ] Measure retrieval quality
- [ ] Measure answer faithfulness
- [ ] Measure answer relevance
- [ ] Compare RAG architectures through ablation studies
- [ ] Compare Similarity vs MMR
- [ ] Benchmark retrieval latency
- [ ] Benchmark end-to-end latency
- [ ] Benchmark cost / quality trade-offs

## Engineering

- [x] Component-level unit tests in the advanced RAG project
- [ ] Expand integration tests
- [ ] Add graph-routing regression tests
- [ ] Add GitHub Actions CI
- [ ] Containerize selected applications
- [ ] Add FastAPI service layers where appropriate
- [ ] Production deployment hardening

## Optimization

- [x] LiteLLM gateway integration
- [x] Provider routing / fallback
- [x] Redis semantic caching
- [x] Per-query cost tracking
- [ ] Cache hit-rate benchmarking
- [ ] Retrieval latency benchmarking
- [ ] Quality / cost trade-off analysis

---

# 🎯 Engineering Focus

The repository focuses on the engineering principles required to build modern GenAI applications.

### 1. Grounding

Use retrieval and external tools to reduce unsupported model outputs.

### 2. Reliability

Validate retrieval quality, detect hallucinations, and retry or reroute when necessary.

### 3. Safety

Protect users, retrieved content, external tools, and generated outputs with multiple layers of validation.

### 4. Observability

Track retrieval behavior, tool usage, latency, and LLM costs.

### 5. Cost Efficiency

Use model routing, semantic caching, bounded retries, and usage tracking to control inference costs.

### 6. Modularity

Separate ingestion, retrieval, orchestration, tools, prompts, state, and interfaces so individual components can evolve independently.

### 7. Testability

Protect core components with automated tests before expanding toward integration and end-to-end evaluation.

### 8. Evaluation

Move beyond qualitative demonstrations and measure whether architectural changes actually improve system performance.

---

# 🧪 Development Philosophy

These projects follow a practical engineering loop:

```text
Build
  ↓
Understand
  ↓
Experiment
  ↓
Identify Failure Modes
  ↓
Add Reliability
  ↓
Test
  ↓
Measure
  ↓
Optimize
  ↓
Deploy
```

Rather than treating an LLM application as a single prompt, the projects explore the broader engineering system around the model.

---

# 🚀 Running a Project

Each project is self-contained.

Clone the repository:

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the desired project:

```bash
cd generative-ai-projects/<project-folder>/<project-name>
```

Then follow that project's individual `README.md`.

There is intentionally **no single universal startup command** for the entire repository. Projects may use:

- `pip`
- `uv`
- Streamlit
- Gradio
- LangGraph CLI
- LiteLLM
- Redis
- MCP servers
- AstraDB
- other project-specific services

Always use the individual project README for exact setup instructions.

---

# 🌐 Live Demos

| Project | Demo |
|---|---|
| Conversational PDF Assistant | [Open Demo](https://conversational-pdf-assistant.streamlit.app) |
| Agentic Search Engine | [Open Demo](https://agentic-search-engine-with-exa.streamlit.app) |
| SQL Chat Assistant | [Open Demo](https://sql-db-chat-assistant.streamlit.app) |
| URL Content Summarizer | [Open Demo](https://url-content-summarizer.streamlit.app) |
| AstraDB Multi-Doc RAG Hub | [Open Demo](https://astradb-multidoc-rag.streamlit.app) |
| Research Paper Assistant | [Open Demo](https://interactive-research-paper-assistant.streamlit.app) |
| Math & Reasoning Agent | [Open Demo](https://math-reasoning-agent.streamlit.app) |
| Q&A Chatbot | [Open Demo](https://question-answer-chatbot-groq.streamlit.app) |

---

# 📊 Project Categories

| Category | Projects |
|---|---|
| LLM Fundamentals | Q&A Chatbot |
| RAG | Conversational PDF, Research Paper Assistant, AstraDB Multi-Doc |
| Advanced RAG | Enhanced Multi-Document Search RAG |
| Query Routing | Intelligent Query Router |
| AI Agents | Agentic Search Engine, Math & Reasoning Agent |
| SQL Agents | SQL Chat Assistant |
| Local AI | Multi-Language Code Assistant |
| Summarization | URL Content Summarizer |
| Multi-Agent AI | Agentic Daily Briefing |
| Tool Integration | MCP / Web Search / Wikipedia / arXiv |
| AI Infrastructure | LiteLLM, Redis Semantic Cache, Cost Tracking |
| Safety | Guardrails, PII Protection, Tool Filtering |
| Testing | Component-level unit testing |

---

# 🔗 Links

- [GitHub Profile](https://github.com/pranayprasad7001)
- [LinkedIn](https://www.linkedin.com/in/pranayprasad7/)
- [Streamlit Applications](https://share.streamlit.io/user/pranayprasad7001)

---

# 📜 License

This repository is licensed under the **MIT License**.

See [LICENSE](./LICENSE) for details.
