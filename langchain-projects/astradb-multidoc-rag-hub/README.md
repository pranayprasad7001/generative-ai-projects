# AstraDB Multi-Doc RAG Hub

A Retrieval-Augmented Generation (RAG) application built with LangChain, Astra DB, Hugging Face Inference API, and Streamlit. The application allows users to upload multiple PDF documents, index them into Astra DB, and ask conversational questions grounded in the uploaded content. Each user session is isolated to prevent document and conversation leakage across sessions. The application supports configurable Hugging Face models, contextual retrieval, and conversational memory.

## Features

- Upload and index multiple PDF documents
- Conversational question answering over uploaded documents
- Context-aware retrieval using chat history
- Session-isolated document indexing and retrieval
- Astra DB Vector Store integration
- Hugging Face Inference API for embeddings and language models
- Configurable Hugging Face model selection
- Automatic document chunking using RecursiveCharacterTextSplitter
- Conversation history trimming for efficient context management
- Secure credential input through the Streamlit sidebar
- Clear chat history functionality
- Temporary storage cleanup after sessions

## Tech Stack

- Python
- Streamlit
- LangChain
- Astra DB
- Hugging Face Inference API
- Hugging Face Embeddings
- LangSmith
- RecursiveCharacterTextSplitter

## Project Structure

```
astradb-multidoc-rag-hub/
│
├── app.py
├── requirements.txt
└── README.md
```

## How It Works

1. Configure Astra DB and Hugging Face credentials.
2. Select the Hugging Face model to use.
3. Upload one or more PDF documents.
4. Documents are loaded, split into chunks, and indexed into Astra DB.
5. Each chunk is tagged with a unique session identifier.
6. User questions are contextualized using conversation history.
7. The retriever searches only the current session's indexed documents.
8. The retrieved context is provided to the language model to generate grounded responses.

## Installation

Clone the repository:

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the project:

```bash
cd generative-ai-projects/langchain-projects/astradb-multidoc-rag-hub
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Required Credentials

Provide the following credentials in the Streamlit sidebar:

- Astra DB Application Token
- Astra DB API Endpoint
- Hugging Face API Token
- Hugging Face Model Repository ID

## Streamlit Application

https://astradb-multidoc-rag.streamlit.app/

## GitHub Repository

https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/astradb-multidoc-rag-hub

## Future Improvements

- Support additional document formats
- Streaming response generation
- Source citation display
- Metadata filtering options
- Hybrid retrieval strategies
- Persistent chat history
- Document management interface

## License

This project is licensed under the MIT License.