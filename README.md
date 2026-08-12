# Generative AI Projects

A growing collection of practical **Generative AI and AI Engineering projects** built with modern LLM frameworks, RAG architectures, AI agents, tool calling, LangGraph, CrewAI, vector databases, MCP, LLM gateways, guardrails, and observability.

The repository documents my progression from foundational LLM applications to increasingly sophisticated **production-oriented GenAI systems**.

Each project is independently structured and includes its own README with architecture, setup instructions, implementation details, and technology choices.

---

## 🧭 What This Repository Covers

The projects in this repository explore different areas of modern Generative AI engineering:

- **LLM Applications**
- **Retrieval-Augmented Generation (RAG)**
- **Adaptive / Self-Correcting RAG**
- **Hybrid Search & Reranking**
- **AI Agents & Tool Calling**
- **LangGraph Workflows**
- **Multi-Agent Systems with CrewAI**
- **MCP-based Tool Integration**
- **Conversational AI**
- **SQL Agents**
- **Local LLM Applications**
- **Semantic Caching**
- **LLM Gateways & Model Routing**
- **Safety Guardrails & PII Protection**
- **LLM Cost Tracking**
- **LangSmith Observability**
- **Streamlit & Gradio Applications**

The goal is not simply to demonstrate that an LLM can generate an answer, but to explore how GenAI systems can be made **more reliable, grounded, observable, secure, cost-aware, and maintainable**.

---

# 🚀 Featured Projects

## 🔥 Enhanced Multi-Document Search RAG

**Advanced adaptive RAG system using LangGraph, hybrid retrieval, reranking, MCP, LiteLLM, Redis semantic caching, and multi-layer guardrails.**

This is currently the most advanced RAG project in the repository.

### Highlights

- Conditional LangGraph `StateGraph`
- Multi-format document ingestion
- Recursive, semantic, and hybrid chunking
- AstraDB vector retrieval
- BM25 keyword retrieval
- Dense + keyword ensemble retrieval
- Cohere reranking
- Query rewriting
- Document relevance grading
- Hallucination detection
- Answer relevance grading
- MCP-based external search
- Tavily, Wikipedia, and arXiv tools
- Redis semantic caching
- LiteLLM gateway
- Model routing and fallback
- Per-query latency and cost tracking
- PII middleware
- Input/output security guardrails
- Tool-call and tool-result filtering
- LangGraph Studio support
- Streamlit interface

**Status:** Core system functional; evaluation, automated testing, CI/CD, and deployment hardening are ongoing.

[View Enhanced Multi-Document Search RAG →](./langchain-projects/enhanced-multi-document-search-rag)

---

## 🤖 Agentic Search Engine

Tool-calling search agent that dynamically uses external search capabilities to answer questions.

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

[View Agentic Search Engine →](./langchain-projects/agentic-search-engine)

[Live Demo →](https://agentic-search-engine-with-exa.streamlit.app)

---

## 🗄️ SQL Chat Assistant

Natural-language interface for querying SQL databases through an agent.

### Highlights

- Natural language → SQL
- SQLite / MySQL support
- Tool-calling agent
- Read-only database enforcement
- SQL safety controls
- Prompt-injection awareness
- Streamlit interface

**Status:** Deployed

[View SQL Chat Assistant →](./langchain-projects/sql-chat-assistant)

[Live Demo →](https://sql-db-chat-assistant.streamlit.app)

---

## 💬 Conversational PDF Assistant

A conversational RAG application that allows users to interact with uploaded PDF documents while maintaining conversation context.

### Highlights

- PDF ingestion
- Vector retrieval
- Conversational history
- History-aware retrieval
- Follow-up question handling
- Source-aware answers
- Streamlit interface

**Status:** Deployed

[View Conversational PDF Assistant →](./langchain-projects/conversational-pdf-assistant)

[Live Demo →](https://conversational-pdf-assistant.streamlit.app)

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

[View AstraDB Multi-Doc RAG Hub →](./langchain-projects/astradb-multidoc-rag-hub)

[Live Demo →](https://astradb-multidoc-rag.streamlit.app)

---

## 📄 Research Paper Assistant

RAG application designed for querying research papers.

### Highlights

- Research paper ingestion
- Chroma vector store
- Persistent vector storage
- Retrieval-based question answering
- Visible source chunks
- Streamlit interface

**Status:** Deployed

[View Research Paper Assistant →](./langchain-projects/research-paper-assistant-rag)

[Live Demo →](https://interactive-research-paper-assistant.streamlit.app)

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

[View Math & Reasoning Agent →](./langchain-projects/math-reasoning-agent)

[Live Demo →](https://math-reasoning-agent.streamlit.app)

---

## 🧠 Intelligent Query Router

A LangGraph-based routing system that analyzes a question and dynamically selects the appropriate information source.

```text
User Query
     │
     ▼
Query Classification
     │
 ┌───┼────────┐
 ▼   ▼        ▼
AstraDB  Wikipedia  arXiv
     │
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

[View Intelligent Query Router →](./langchain-projects/intelligent-query-router)

---

## 💻 Multi-Language Code Assistant

A local coding assistant powered by Ollama.

### Highlights

- Local LLM inference
- Ollama
- Gradio interface
- Offline operation
- Multiple programming languages
- Two iterations: basic → improved UI

**Status:** Runs locally

[View Multi-Language Code Assistant →](./langchain-projects/multi-language-code-assistant)

---

## 🌐 URL Content Summarizer

Summarizes long-form web content and YouTube transcripts.

The application dynamically chooses between different summarization strategies depending on content length.

### Highlights

- URL ingestion
- YouTube transcript processing
- Stuff chain
- Map-reduce chain
- Automatic strategy selection
- Streamlit interface

**Status:** Deployed

[View URL Content Summarizer →](./langchain-projects/url-content-summarizer)

[Live Demo →](https://url-content-summarizer.streamlit.app)

---

## 💬 Q&A Chatbot with Groq

The original project in this repository and the foundation for later experimentation with LangChain and LLM applications.

### Highlights

- Prompt templates
- Groq LLM
- LangChain
- Streamlit
- Basic conversational question answering

**Status:** Deployed

[View Q&A Chatbot →](./langchain-projects/qa-chatbot)

[Live Demo →](https://question-answer-chatbot-groq.streamlit.app)

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

[View Agentic Daily Briefing →](./crewai-projects/agentic-daily-briefing)

---

# 🏗️ Repository Structure

```text
generative-ai-projects/
│
├── langchain-projects/
│   │
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
│   │
│   └── agentic-daily-briefing/
│
├── LICENSE
└── README.md
```

The repository currently contains **11 LangChain projects and 1 CrewAI project**.

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

## Retrieval & RAG

- AstraDB
- ChromaDB
- BM25
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
- LangGraph agents
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

## Observability

- LangSmith
- Application logging
- Latency tracking
- LLM cost tracking

---

# 📈 Project Progression

The projects represent an intentional progression in complexity.

```text
                    Foundation
                        │
                        ▼
                Q&A Chatbot
                        │
                        ▼
                RAG Applications
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       PDF / Research         Multi-Document
            RAG                    RAG
             │                     │
             └──────────┬──────────┘
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
              Production Concerns
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
   Guardrails        Caching         Cost Tracking
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                 Evaluation
                        │
                        ▼
                Deployment / CI
```

The projects are therefore not intended to be a collection of unrelated demos. They document an ongoing progression from **basic LLM applications toward more reliable and production-oriented AI systems**.

---

# 🛡️ Reliability & Evaluation Roadmap

The repository has evolved from experimentation toward production-oriented engineering.

Several projects already include reliability mechanisms such as:

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

The next focus is making these systems measurable.

### Evaluation

- [ ] Build reusable evaluation datasets
- [ ] Add RAGAS evaluations
- [ ] Measure retrieval quality
- [ ] Measure answer faithfulness
- [ ] Measure answer relevance
- [ ] Compare RAG architectures through ablation studies
- [ ] Benchmark latency and cost

### Engineering

- [ ] Expand automated unit tests
- [ ] Add graph-routing tests
- [ ] Add integration tests
- [ ] Add GitHub Actions CI
- [ ] Containerize selected applications
- [ ] Add FastAPI service layers where appropriate

### Optimization

- [x] LiteLLM gateway integration
- [x] Provider routing / fallback
- [x] Redis semantic caching
- [x] Per-query cost tracking
- [ ] Cache hit-rate benchmarking
- [ ] Retrieval latency benchmarking
- [ ] Cost / quality trade-off analysis

---

# 🎯 Engineering Focus

The main focus of this repository is learning and implementing the engineering principles required to build modern GenAI applications:

### 1. Grounding

Use retrieval and external tools to reduce unsupported model outputs.

### 2. Reliability

Validate retrieval quality, detect hallucinations, and retry or reroute when necessary.

### 3. Safety

Protect users, tools, retrieved content, and generated outputs with multiple layers of validation.

### 4. Observability

Track what the system retrieved, which tools were used, how long execution took, and how much the LLM calls cost.

### 5. Cost Efficiency

Use model routing, caching, bounded retries, and usage tracking to control inference costs.

### 6. Modularity

Separate ingestion, retrieval, orchestration, tools, prompts, state, and interfaces so individual components can evolve independently.

### 7. Evaluation

Move beyond qualitative demonstrations and measure whether architectural changes actually improve system performance.

---

# 🧪 Development Philosophy

These projects follow a simple progression:

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

Then follow the project's individual `README.md`.

Most projects use:

```bash
pip install -r requirements.txt
```

and Streamlit-based projects can generally be started with:

```bash
streamlit run app.py
```

However, **do not assume every project follows the same startup command**. Some projects use `uv`, CLI entry points, or additional infrastructure such as LiteLLM, Redis, MCP servers, or external databases.

Always refer to the individual project README for the exact setup.

---

# 🌐 Live Demos

Several projects are available as live Streamlit applications.

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

---

# 🔗 Links

- **GitHub:** [pranayprasad7001](https://github.com/pranayprasad7001)
- **LinkedIn:** [Pranay Prasad](https://www.linkedin.com/in/pranayprasad7/)
- **Streamlit Apps:** [View deployed applications](https://share.streamlit.io/user/pranayprasad7001)

---

# 📜 License

This repository is licensed under the **MIT License**.

See [LICENSE](./LICENSE) for details.