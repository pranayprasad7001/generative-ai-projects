from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(BaseModel):
    """State for the autonomous RAG workflow."""

    messages: list[BaseMessage] = Field(
        default_factory=list,
        description="Conversation and tool messages."
    )

    original_question: str = Field(
        default="",
        description="Original question from the user."
    )

    question: str = Field(
        ...,
        description="User's current question."
    )

    analysis: str = Field(
        default="",
        description="Analysis of the user's question."
    )

    tool_type: Literal["vector_search", "external", "none"] | None = Field(
        default=None,
        description="Retrieval strategy selected for the question."
    )

    retrieved_docs: list[Document] = Field(
        default_factory=list,
        description="Documents retrieved from the vector store."
    )

    external_results: str = Field(
        default="",
        description="Results retrieved from external search tools."
    )

    answer: str = Field(
        default="",
        description="Final answer generated for the user."
    )

    rewrite_count: int = Field(
        default=0,
        description="Number of times the question has been rewritten."
    )