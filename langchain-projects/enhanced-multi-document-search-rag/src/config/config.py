"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

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
    LLM_MODEL = "groq:openai/gpt-oss-120b"
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
        """Initialize and return the LLM model"""
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return init_chat_model(cls.LLM_MODEL)