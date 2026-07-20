# Agentic Daily Briefing

A CrewAI-based multi-agent application that researches and compiles a structured daily news briefing. The project uses specialized AI agents to gather the latest news from the web and generate a categorized markdown report.

## Features

- Multi-agent workflow using CrewAI
- Web search using Tavily Search
- Sequential task execution
- Categorized daily news briefing
- Markdown report generation
- Configurable LLMs via NVIDIA API
- Hugging Face embeddings support

## Workflow

1. Research Agent searches the web for the latest news.
2. News is collected across predefined categories.
3. Writer Agent formats the research into a structured briefing.
4. Final report is generated in Markdown.

## News Categories

- Breaking News
- India
- World
- AI & Technology
- Science

## Project Structure

```
agentic-daily-briefing/
│
├── agents.py          # Agent definitions
├── tasks.py           # CrewAI tasks
├── tools.py           # Tavily search tool
├── crew_main.py       # Crew execution
├── .env
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- CrewAI
- CrewAI Tools
- LiteLLM
- NVIDIA AI API
- Tavily Search
- Hugging Face Embeddings
- python-dotenv

## Installation

Clone the repository.

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Navigate to the project directory.

```bash
cd generative-ai-projects/crewai-projects/agentic-daily-briefing
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

Install the required dependencies.

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file and add the following:

```text
NVIDIA_API_KEY=your_api_key
NVIDIA_API_BASE=your_api_base
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token
CREWAI_TRACING_ENABLED=true
```

## Running the Project

```bash
python crew_main.py
```

The generated reports will be saved as Markdown files.

## Future Improvements

- Streamlit interface
- Custom news topics
- Additional news categories
- Scheduled daily execution
- Export to PDF
- Email delivery
- Multiple LLM provider support

## License

This project is licensed under the MIT License.