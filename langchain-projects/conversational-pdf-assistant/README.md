# Conversational PDF Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to upload one or more PDF documents and interact with them through a conversational interface. The assistant maintains chat history, allowing follow-up questions while preserving context throughout the conversation.

## Live Demo

**Streamlit:**  
https://conversational-pdf-assistant.streamlit.app

## GitHub Repository

https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/conversational-pdf-assistant

---

## Features

- Upload and chat with multiple PDF documents
- Conversational question answering with chat history
- History-aware retrieval for contextual follow-up questions
- Semantic document search using vector embeddings
- Configurable LLM, temperature, and maximum token settings
- Session-based conversation management
- Automatic document chunking and embedding generation
- Streamlit-based user interface

---

## Tech Stack

### Frameworks & Libraries

- Python
- Streamlit
- LangChain
- LangChain Chroma
- LangChain HuggingFace
- LangChain Groq

### Models

- Groq LLMs
- BAAI/bge-small-en-v1.5 Embedding Model

### Vector Database

- ChromaDB

---

## Project Workflow

1. Upload one or more PDF documents.
2. Extract text from each document.
3. Split documents into semantic chunks.
4. Generate embeddings for every chunk.
5. Store embeddings in a Chroma vector database.
6. Retrieve the most relevant document chunks for each query.
7. Rephrase follow-up questions using conversation history.
8. Generate grounded responses using the retrieved context.
9. Maintain session-based chat history for contextual conversations.

---

## Project Structure

```
conversational-pdf-assistant/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── assets/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

### Navigate to the project

```bash
cd generative-ai-projects/langchain-projects/conversational-pdf-assistant
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_api_key
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## How It Works

- PDFs are uploaded through the Streamlit interface.
- Text is extracted using `PyPDFLoader`.
- Documents are split into manageable chunks.
- HuggingFace embeddings are generated for every chunk.
- ChromaDB stores the vector representations.
- A history-aware retriever reformulates follow-up questions using previous conversation context.
- Retrieved document chunks are passed to the LLM to generate grounded responses.
- Chat history is maintained throughout the session to support natural multi-turn conversations.

---

## Sample Use Cases

- Research papers
- Technical documentation
- User manuals
- Reports
- Academic notes
- Company documentation
- Books and reference materials

---

## Future Improvements

- Source citations for generated answers
- Streaming responses
- Conversation export
- Persistent vector database
- OCR support for scanned PDFs
- Hybrid search (keyword + semantic)
- Authentication and user management

---

## License

This project is licensed under the MIT License.