"""RAG State Definition for LangGraph"""

from typing import List, Literal
from pydantic import BaseModel
from langchain_classic.schema import Document


class RAGState(BaseModel):
    """State object for RAG workflow"""
    question: str
    analysis: str = ""
    tool_type: Literal["vector_search", "external", "none"]
    retrieved_docs: List[Document] = []
    answer: str = ""
