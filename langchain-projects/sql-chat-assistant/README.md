# SQL Chat Assistant

A Streamlit application that allows users to interact with SQL databases using natural language. Built with LangChain SQL Agents, Groq LLMs, and SQLAlchemy, the application can answer questions by generating and executing SQL queries on connected databases.

## Features

- Chat with SQL databases using natural language
- Supports both SQLite and MySQL databases
- Built using LangChain SQL Agent
- Multiple Groq model options
- Adjustable model temperature
- Chat history during the session
- SQL agent execution trace using Streamlit callbacks
- Read-only SQLite database support
- Error handling for database and LLM connections

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- SQLAlchemy
- SQLite
- MySQL

## Project Structure

```
sql-chat-assistant/
│
├── app.py
├── student.db
├── requirements.txt
├── README.md
└── .env (optional)
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the project folder:

```bash
cd generative-ai-projects/langchain-projects/sql-chat-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Using the Application

1. Enter your Groq API Key.
2. Select a language model.
3. Adjust the temperature if required.
4. Choose one of the following:
   - Use the provided SQLite database.
   - Connect to your own MySQL database.
5. Ask questions in natural language.

Example queries:

- Show all students.
- How many records are present?
- List students with marks greater than 80.
- What is the average score?
- Which student has the highest marks?

## Database Support

### SQLite

The application includes a sample SQLite database (`student.db`) that can be queried immediately.

### MySQL

Provide:

- Host
- Username
- Password
- Database Name

The application establishes the connection using SQLAlchemy.

## Models

The application supports multiple Groq-hosted models, including:

- qwen/qwen3-32b
- qwen/qwen3.6-27b
- llama-3.3-70b-versatile
- llama-3.1-8b-instant
- openai/gpt-oss-120b
- openai/gpt-oss-20b

## Deployment

**Streamlit**

https://sql-db-chat-assistant.streamlit.app/

**GitHub Repository**

https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/sql-chat-assistant

## Notes

- The SQLite database is opened in read-only mode.
- A warning is displayed regarding prompt injection risks when allowing an LLM to execute SQL queries.
- For production deployments, use a database account with limited privileges.

## License

This project is licensed under the MIT License.