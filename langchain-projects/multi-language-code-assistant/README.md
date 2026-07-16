# Multi-language Code Assistant

A local AI-powered code assistant built with **Gradio** and **Ollama** for writing, explaining, debugging, reviewing, and optimizing code across multiple programming languages. The project demonstrates the evolution from a simple request-based interface to a more feature-rich conversational assistant with dynamic local model selection and an improved user interface.

> **Note:** This project runs completely offline using locally installed Ollama models and has not been deployed.

---

## Features

- Multi-language code assistance
- Explain code snippets
- Debug programming errors
- Generate code from natural language prompts
- Review and optimize code
- Context-aware conversations
- Dynamic detection of installed Ollama models
- Select local models directly from the UI
- Adjustable temperature for response generation
- Conversation history support
- Clear conversation functionality
- Graceful error handling for Ollama connection issues
- Runs entirely offline without external APIs

---

## Tech Stack

- Python
- Gradio
- Ollama
- Requests
- JSON

---

## Project Structure

```text
multi-language-code-assistant/
│
├── app_v1_basic.py          # Basic implementation using Ollama REST API
├── app_v2_better_ui.py      # Enhanced interface using Ollama Python SDK
├── requirements.txt
└── README.md
```

---

## Version Overview

### Version 1 – Basic Implementation

The first version provides a lightweight interface that communicates directly with the local Ollama REST API using HTTP requests.

**Features**

- REST API communication with Ollama
- Manual conversation history management
- Simple Gradio interface
- Lightweight implementation
- Minimal dependencies

---

### Version 2 – Enhanced Interface

The second version improves both usability and functionality by using the official Ollama Python library together with a modern Gradio Blocks interface.

**Enhancements**

- Native Ollama Python SDK
- Automatic detection of installed models
- Model selection dropdown
- Adjustable temperature slider
- Better conversation handling
- Improved Gradio Blocks UI
- Clear chat functionality
- Better error handling
- More maintainable architecture

---

## Installation

### Clone the repository

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git

cd generative-ai-projects/langchain-projects/multi-language-code-assistant
```

---

### Create a Virtual Environment

```bash
python -m venv .venv
```

---

### Activate the Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama from:

https://ollama.com

After installation, pull any model you would like to use.

For example:

```bash
ollama pull llama3
```

or

```bash
ollama pull codellama
```

or use your own custom fine-tuned model.

---

## Running the Project

### Version 1

```bash
python app_v1_basic.py
```

---

### Version 2

```bash
python app_v2_better_ui.py
```

---

## Example Prompts

- Explain this Python code.
- Find the bug in my Java program.
- Optimize this SQL query.
- Convert this C++ code to Python.
- Write a REST API using FastAPI.
- Explain time complexity of this algorithm.
- Review my code and suggest improvements.

---

## Requirements

- Python 3.10+
- Ollama installed locally
- At least one downloaded Ollama model

---

## Future Improvements

- Streaming token responses
- Syntax highlighting
- File upload support
- Multiple conversation sessions
- Project-aware code analysis
- Export chat history
- Markdown rendering for code explanations

---

## Repository

**GitHub**

https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/multi-language-code-assistant

---

## License

This project is licensed under the MIT License.