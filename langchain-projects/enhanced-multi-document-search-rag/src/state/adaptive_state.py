from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AdaptiveRAGState(BaseModel):
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
        description="Current question used by the workflow."
    )

    analysis: str = Field(
        default="",
        description="Analysis of the user's question."
    )

    tool_type: Literal["vector_search", "external_search"] | None = Field(
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

    retrieval_grade: Literal["yes", "no"] | None = Field(
        default=None,
        description="Whether the retrieved documents are relevant and sufficient."
    )

    hallucination_grade: Literal["yes", "no"] | None = Field(
        default=None,
        description="Whether the generated answer is grounded in the provided context."
    )

    answer_relevance_grade: Literal["yes", "no"] | None = Field(
        default=None,
        description="Whether the generated answer adequately answers the user's question."
    )

    rewrite_count: int = Field(
        default=0,
        description="Number of times the question has been rewritten."
    )

    generate_count: int = Field(
        default=0,
        description="Number of times the answer has been generated."
    )

    query_blocked: bool = Field(
        default=False,
        description="Whether the user's query was blocked by the input guardrail."
    )

    total_cost: float = Field(
        default=0.0,
        description="Cumulative cost of model usage in USD."
    )

    external_citations: list[str] = Field(
        default_factory=list,
        description="Citations/source URLs extracted from external search."
    )