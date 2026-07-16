# Generative AI Projects

A collection of Generative AI applications built while learning and applying LangChain, RAG, AI agents, and modern LLM tooling — one project at a time, from simple prompt-based chatbots to multi-tool agents and retrieval systems backed by different vector stores.

Each project lives in its own folder under `langchain-projects/`, is independently deployable, and ships with its own README covering setup, tech stack, and how it works.

---

## Projects

| Project | Description | Live Demo |
|---|---|---|
| [Conversational PDF Assistant](langchain-projects/conversational-pdf-assistant) | RAG chatbot over uploaded PDFs with history-aware retrieval for follow-up questions | [Demo](https://conversational-pdf-assistant.streamlit.app) |
| [Agentic Search Engine](langchain-projects/agentic-search-engine) | Tool-calling agent that answers questions using live web search (Exa) | [Demo](https://agentic-search-engine-with-exa.streamlit.app/) |
| [SQL Chat Assistant](langchain-projects/sql-chat-assistant) | Natural language to SQL agent for SQLite/MySQL with read-only enforcement | [Demo](https://sql-db-chat-assistant.streamlit.app/) |
| [URL Content Summarizer](langchain-projects/url-content-summarizer) | Summarizes YouTube videos and web pages, auto-switching between stuff and map-reduce chains | [Demo](https://url-content-summarizer.streamlit.app/) |
| [AstraDB Multi-Doc RAG Hub](langchain-projects/astradb-multidoc-rag-hub) | Multi-PDF RAG using Astra DB as the vector store with Hugging Face inference and session-isolated indexing | [Demo](https://astradb-multidoc-rag.streamlit.app/) |
| [Research Paper Assistant (RAG)](langchain-projects/research-paper-assistant-rag) | RAG system for research papers with persistent Chroma storage and visible source chunks | [Demo](https://interactive-research-paper-assistant.streamlit.app) |
| [Math & Reasoning Agent](langchain-projects/math-reasoning-agent) | ReAct agent that dynamically picks between a calculator, reasoning chain, and Wikipedia lookup | [Demo](https://math-reasoning-agent.streamlit.app/) |
| [Multi-language Code Assistant](langchain-projects/multi-language-code-assistant) | Local, fully offline coding assistant using Ollama and Gradio (two iterations, v1 basic → v2 improved UI) | Runs locally, not deployed |
| [Q&A Chatbot with Groq](langchain-projects/qa-chatbot) | The original starter project — a lightweight prompt-template-based Q&A chatbot | [Demo](https://question-answer-chatbot-groq.streamlit.app) |

---

## What This Repo Covers

- **Retrieval-Augmented Generation (RAG)** — with two different vector store backends (ChromaDB, Astra DB) and both Groq and Hugging Face-hosted models
- **AI Agents** — tool-calling agents (web search) and ReAct agents (multi-tool: math, reasoning, Wikipedia)
- **Conversational AI** — session-based chat history, history-aware retrieval for follow-ups
- **SQL Agents** — natural language database querying with read-only safeguards and prompt-injection awareness
- **Local LLMs** — an Ollama-based offline coding assistant, no external API dependency
- **Summarization strategies** — automatic stuff vs. map-reduce chain selection based on content length
- **LangSmith tracing** — integrated across projects for observability into agent/chain execution

---

## Tech Stack

**Languages & Core:** Python

**LLM Frameworks:** LangChain, LangGraph

**Models / APIs:** Groq, Google Gemini, Hugging Face Inference API, Ollama (local)

**Vector Stores:** ChromaDB, Astra DB

**Interfaces:** Streamlit, Gradio

**Other:** SQLAlchemy, Exa Search API, Wikipedia API, LangSmith

---

## Repository Structure

```
generative-ai-projects/
│
├── langchain-projects/
│   ├── conversational-pdf-assistant/
│   ├── agentic-search-engine/
│   ├── sql-chat-assistant/
│   ├── url-content-summarizer/
│   ├── astradb-multidoc-rag-hub/
│   ├── research-paper-assistant-rag/
│   ├── math-reasoning-agent/
│   ├── multi-language-code-assistant/
│   └── qa-chatbot/
│
├── LICENSE
└── README.md
```

---

## Running a Project Locally

Each project is self-contained. General pattern:

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
cd generative-ai-projects/langchain-projects/<project-name>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Most projects need a Groq API key (free tier at [console.groq.com](https://console.groq.com/keys)); a few require additional credentials (Astra DB, Google Gemini, Exa) — see the individual project README for specifics.

---

## Links

- **GitHub:** [github.com/pranayprasad7001](https://github.com/pranayprasad7001)
- **LinkedIn:** [linkedin.com/in/pranayprasad7](https://linkedin.com/in/pranayprasad7)
- **Streamlit Apps:** [share.streamlit.io/user/pranayprasad7001](https://share.streamlit.io/user/pranayprasad7001)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.