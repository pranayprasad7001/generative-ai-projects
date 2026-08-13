"""Configuration module for Agentic RAG system"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Load environment variables
load_dotenv()

import litellm

# Register custom model pricing for LiteLLM cost calculation fallback
try:
    litellm.register_model({
        "nvidia-glm-5.2": {
            "max_tokens": 12000,
            "input_cost_per_token": 0.0000014,
            "output_cost_per_token": 0.0000042,
            "litellm_provider": "openai",
            "mode": "chat"
        },
        "gpt-oss-120b-groq": {
            "max_tokens": 8192,
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.00000075,
            "litellm_provider": "openai",
            "mode": "chat"
        },
        "gpt-oss-20b-groq": {
            "max_tokens": 8192,
            "input_cost_per_token": 0.000000075,
            "output_cost_per_token": 0.0000003,
            "litellm_provider": "openai",
            "mode": "chat"
        },
        "qwen3.6-27b-groq": {
            "max_tokens": 8192,
            "input_cost_per_token": 0.000000289,
            "output_cost_per_token": 0.0000024,
            "litellm_provider": "openai",
            "mode": "chat"
        }
    })
except Exception as e:
    
    logging.getLogger(__name__).warning("Failed to register custom models in LiteLLM: %s", e)

class Config:
    """Configuration class for RAG system"""
    
    # Langsmith Tracing
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

    # API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    ASTRA_DB_API_KEY = os.getenv("ASTRA_DB_API_KEY")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT") or os.getenv("ASTRA_DB_ENDPOINT")
    ASTRA_DB_API_REGION = os.getenv("ASTRA_DB_API_REGION")
    ASTRA_DB_COLLECTION_NAME = "rag_multi_doc_collection"

    # Model Configuration
    
    # LiteLLM
    LITELLM_BASE_URL = os.getenv(
        "LITELLM_BASE_URL",
        "http://localhost:4000"
    )

    LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")

    # Logical model name defined in litellm_config.yaml
    LLM_MODEL = "gpt-oss-120b-groq"
    EMBEDDING_MODEL = "gemini-embedding-2" #"BAAI/bge-base-en-v1.5"  
    COHERE_RERANKER_MODEL = "rerank-english-v3.0"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    COHERE_RERANKER_TOP_N = 5
    MAX_REWRITES = 3
    MAX_GENERATIONS = 3
    
    # Generation parameters
    LLM_TEMPERATURE = 0.2
    LLM_TOP_P = 0.9
    LLM_MAX_TOKENS = 7000
    LLM_RATE_LIMITER = 7000
    MAX_RETRIES = 3

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize LLM through LiteLLM Gateway."""
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=cls.LLM_RATE_LIMITER,
            max_bucket_size=100
        )
        return ChatOpenAI(
            model=cls.LLM_MODEL,
            api_key=cls.LITELLM_API_KEY,
            base_url=cls.LITELLM_BASE_URL,
            temperature=cls.LLM_TEMPERATURE,
            top_p=cls.LLM_TOP_P,
            max_tokens=cls.LLM_MAX_TOKENS,
            include_response_headers=True,
            rate_limiter=rate_limiter,
            max_retries=cls.MAX_RETRIES
        )