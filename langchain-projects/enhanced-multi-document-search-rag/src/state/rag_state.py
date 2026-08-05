"""RAG State Definition for LangGraph"""

from typing import List
from pydantic import BaseModel
from langchain_classic.schema import Document


class RAGState(BaseModel):
    """State object for RAG workflow"""
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
