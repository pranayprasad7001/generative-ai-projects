"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    ASTRA_DB_API_KEY = os.getenv("ASTRA_DB_API_KEY")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT") or os.getenv("ASTRA_DB_ENDPOINT")
    ASTRA_DB_API_REGION = os.getenv("ASTRA_DB_API_REGION")
    
    # Model Configuration
    LLM_MODEL = "groq:openai/gpt-oss-120b"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
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