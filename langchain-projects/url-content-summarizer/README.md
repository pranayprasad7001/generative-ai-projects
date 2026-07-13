# URL Content Summarizer

A Streamlit application that summarizes content from YouTube videos and web pages using LangChain and Groq LLMs. The application automatically selects an appropriate summarization strategy based on the length of the extracted content.

## Features

- Summarize YouTube videos from their URL
- Summarize content from web pages
- Supports multiple Groq-hosted LLMs
- Automatic document loading based on URL type
- Token estimation before summarization
- Uses Stuff summarization for shorter documents
- Uses Map-Reduce summarization for longer documents
- Adjustable model temperature
- Simple Streamlit interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq
- LangChain Community Loaders
- Recursive Character Text Splitter

## Models Supported

- Llama 3.3 70B Versatile
- Llama 3.1 8B Instant
- OpenAI GPT-OSS 120B
- OpenAI GPT-OSS 20B

## Project Structure

```
url-content-summarizer/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/pranayprasad7001/generative-ai-projects.git
```

Move into the project directory:

```bash
cd generative-ai-projects/langchain-projects/url-content-summarizer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Configuration

The application requires a Groq API key.

You can either:

- Enter it directly from the Streamlit sidebar, or
- Store it in an environment variable if you choose to modify the application.

Get your API key from:

https://console.groq.com/keys

## How It Works

1. Enter your Groq API key.
2. Select a supported language model.
3. Adjust the temperature if needed.
4. Paste a YouTube or webpage URL.
5. The application:
   - Detects the URL type.
   - Loads the content.
   - Estimates token count.
   - Chooses an appropriate summarization chain.
   - Generates a concise summary.

## Streamlit Demo

https://url-content-summarizer.streamlit.app/

## GitHub Repository

https://github.com/pranayprasad7001/generative-ai-projects/tree/main/langchain-projects/url-content-summarizer

## License

This project is licensed under the MIT License.