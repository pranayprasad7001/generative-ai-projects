# Research Paper Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) application that enables users to ask questions about one or more research papers in natural language. The application retrieves relevant sections from uploaded PDF documents and uses an LLM to generate context-aware responses while displaying the supporting document chunks.

## Live Demo

- **GitHub Repository:** https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/research-paper-assistant-rag
- **Streamlit Application:** https://interactive-research-paper-assistant.streamlit.app

---

## Features

- Upload and query multiple research papers in PDF format
- Semantic search using vector embeddings
- Retrieval-Augmented Generation (RAG) pipeline
- Persistent Chroma vector database
- Local Hugging Face embedding model
- Batched document embedding for memory-efficient indexing
- Source document visualization for retrieved responses
- Interactive Streamlit interface

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Framework | LangChain |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Embeddings | Hugging Face (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Document Loader | PyPDFLoader |
| Language | Python |

---

## Project Workflow

```text
Upload PDF Documents
        │
        ▼
Extract Document Text
        │
        ▼
Split into Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Store in ChromaDB
        │
        ▼
User Query
        │
        ▼
Retrieve Relevant Chunks
        │
        ▼
Generate Response using LLM
        │
        ▼
Display Answer with Source Context
```

---

## Project Structure

```text
research-paper-assistant-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── chroma_db/
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the project directory.

```bash
cd generative-ai-projects/langchain-projects/research-paper-assistant-rag
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project directory.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## How It Works

1. Upload one or more research papers.
2. Process and embed the uploaded documents.
3. Ask questions related to the uploaded papers.
4. The application retrieves the most relevant document chunks.
5. The LLM generates an answer using the retrieved context.
6. Retrieved source chunks are displayed for reference.

---

## Current Limitations

- Supports PDF documents only.
- Uploaded documents must be processed before querying.
- Response quality depends on the retrieved context.
- Conversation history is not maintained across questions.

---

## Future Improvements

- Conversation memory
- Advanced retrieval strategies
- Metadata-based document filtering
- Streaming responses
- Improved citation support
- Support for additional document formats

---

## Acknowledgements

This project uses the following open-source libraries and services:

- LangChain
- ChromaDB
- Hugging Face
- Streamlit
- Groq API