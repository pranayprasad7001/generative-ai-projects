# Math & Reasoning Agent

A Streamlit application built with LangChain that uses a ReAct agent to solve mathematical problems, answer logical reasoning questions, and retrieve factual information from Wikipedia. The agent dynamically selects the appropriate tool based on the user's query and explains the reasoning process behind its responses.

**Live Demo:** https://math-reasoning-agent.streamlit.app/

---

## Features

- Solve mathematical calculations using LangChain's `LLMMathChain`
- Answer logical and analytical reasoning questions
- Retrieve factual information using Wikipedia
- ReAct agent for dynamic tool selection
- Conversation history
- View intermediate reasoning and tool execution steps
- Retry mechanism for transient API failures
- Google Gemini model selection
- Streamlit-based interactive interface

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- Wikipedia API
- LangSmith

---

## Project Structure

```
math-reasoning-agent/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## How It Works

1. The user enters a question through the Streamlit interface.
2. A LangChain ReAct agent analyzes the query.
3. Based on the request, the agent selects one or more tools:
   - **Calculator Tool** for mathematical computations
   - **Reasoning Tool** for logical problem solving
   - **Wikipedia Tool** for factual information retrieval
4. The selected tool executes the task.
5. The final response is returned along with optional intermediate reasoning steps.

---

## Tools Used

### Calculator Tool

Uses LangChain's `LLMMathChain` to evaluate mathematical expressions and perform calculations.

### Reasoning Tool

Uses a custom prompt with an `LLMChain` to provide step-by-step reasoning for analytical and logical questions.

### Wikipedia Tool

Uses the LangChain `WikipediaAPIWrapper` to retrieve factual information from Wikipedia.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the project.

```bash
cd generative-ai-projects/langchain-projects/math-reasoning-agent
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.streamlit/secrets.toml` file and add the following:

```toml
GOOGLE_API_KEY="your_google_api_key"

LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_PROJECT="your_project_name"
LANGSMITH_TRACING_V2="true"
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Project Highlights

- LangChain ReAct Agent
- Multi-tool agent architecture
- Dynamic tool selection
- Mathematical reasoning
- Logical reasoning
- Wikipedia search integration
- Intermediate reasoning visualization
- Google Gemini integration
- LangSmith tracing support
- Streamlit interface

---

## Repository

GitHub Repository:

https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/math-reasoning-agent

---

## Live Application

https://math-reasoning-agent.streamlit.app/

---

## License

This project is licensed under the MIT License.