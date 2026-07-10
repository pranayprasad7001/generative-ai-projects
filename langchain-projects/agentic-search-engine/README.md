# Agentic Search Engine

An AI-powered search assistant that combines a language model with web search tools to answer user queries using up-to-date information. The application supports conversational interactions, maintains chat history during a session, and uses tool-calling to retrieve relevant information from the web before generating responses.

**GitHub Repository:**  
https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/agentic-search-engine

**Live Demo:**  
https://agentic-search-engine-with-exa.streamlit.app/

---

## Features

- AI agent capable of answering questions using live web search
- Tool-calling with LangChain Agents
- Exa Search integration for retrieving web information
- Similar webpage discovery using Exa Find Similar
- Conversational interface with session-based chat history
- Streaming-style reasoning visualization in Streamlit
- Retry mechanism for transient API failures
- Rate-limit handling with user-friendly messages
- Secure API key input through the Streamlit sidebar

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- Exa Search API

---

## Project Structure

```
agentic-search-engine/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
└── README.md
```

---

## How It Works

1. The user enters their Groq and Exa API keys.
2. A question is submitted through the chat interface.
3. A LangChain tool-calling agent determines whether a web search is required.
4. The agent uses Exa Search (or Exa Find Similar when appropriate) to retrieve relevant information.
5. The retrieved information is passed back to the language model.
6. The model generates a final response while maintaining conversation history for the current session.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the project folder:

```bash
cd generative-ai-projects/langchain-projects/agentic-search-engine
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## API Keys

This project requires:

- Groq API Key
- Exa API Key

Enter both keys in the Streamlit sidebar before using the application.

---

## Future Improvements

- Support for additional search providers
- Multi-tool agent workflows
- Source citations in responses
- Persistent conversation history
- Follow-up question suggestions
- Configurable LLM and search settings

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.