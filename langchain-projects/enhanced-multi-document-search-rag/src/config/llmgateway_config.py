"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    ASTRA_DB_API_KEY = os.getenv("ASTRA_DB_API_KEY")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT") or os.getenv("ASTRA_DB_ENDPOINT")
    ASTRA_DB_API_REGION = os.getenv("ASTRA_DB_API_REGION")
    ASTRA_DB_COLLECTION_NAME = "rag_multi_doc_collection"

    # Model Configuration
    #LLM_MODEL = "groq:openai-gpt-oss-120b"
    # LiteLLM
    LITELLM_BASE_URL = os.getenv(
        "LITELLM_BASE_URL",
        "http://localhost:4000"
    )

    LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")

    # Logical model name defined in litellm_config.yaml
    LLM_MODEL = "gpt-oss-120b-groq"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    COHERE_RERANKER_MODEL = "rerank-english-v3.0"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    COHERE_RERANKER_TOP_N = 5
    MAX_REWRITES = 3
    MAX_GENERATIONS = 3
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize LLM through LiteLLM Gateway."""
        return ChatOpenAI(
            model=cls.LLM_MODEL,
            api_key=cls.LITELLM_API_KEY,
            base_url=cls.LITELLM_BASE_URL,
        )