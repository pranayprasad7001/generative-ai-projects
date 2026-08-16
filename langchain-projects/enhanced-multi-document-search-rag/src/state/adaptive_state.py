"""Adaptive RAG State Schema Module.

This module defines the central Pydantic state model (`AdaptiveRAGState`) passed across
all nodes and conditional edges in the LangGraph workflow. It maintains the user's query,
conversational messages, retrieval artifacts, continuous grading scores, granular reasoning traces,
and stage-by-stage latency breakdowns.
"""

from typing import Literal, Optional, List, Dict, Union

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AdaptiveRAGState(BaseModel):
    """Encapsulates the state payload flowing through the Adaptive RAG StateGraph workflow."""

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

    # Generalized analysis for backward compatibility
    analysis: str = Field(
        default="",
        description="General analysis or explanation for backward compatibility."
    )

    # Granular reasoning fields for LangSmith traces and stage-by-stage observability
    query_analysis: str = Field(
        default="",
        description="Reasoning behind query routing and tool selection."
    )

    retrieval_reasoning: str = Field(
        default="",
        description="Reasoning behind retrieved document relevance evaluation."
    )

    rewrite_reasoning: str = Field(
        default="",
        description="Reasoning behind query rewriting and coreference resolution."
    )

    grounding_reasoning: str = Field(
        default="",
        description="Reasoning behind factual groundedness and hallucination detection."
    )

    relevance_reasoning: str = Field(
        default="",
        description="Reasoning behind final answer relevance evaluation."
    )

    tool_type: Literal["hybrid_retrieval", "vector_search", "external_search"] | None = Field(
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

    retrieval_grade: Literal["yes", "no", "pass", "fail", "rewrite"] | str | None = Field(
        default=None,
        description="Whether the retrieved documents are relevant and sufficient."
    )

    retrieval_score: float | None = Field(
        default=None,
        description="Continuous relevance score (0.0 to 1.0) of retrieved documents."
    )

    hallucination_grade: Literal["yes", "no", "pass", "fail", "retry"] | str | None = Field(
        default=None,
        description="Whether the generated answer is grounded in the provided context."
    )

    hallucination_score: float | None = Field(
        default=None,
        description="Continuous groundedness score (0.0 to 1.0) of generated answer."
    )

    answer_relevance_grade: Literal["yes", "no", "pass", "fail", "retry", "rewrite"] | str | None = Field(
        default=None,
        description="Whether the generated answer adequately answers the user's question."
    )

    answer_relevance_score: float | None = Field(
        default=None,
        description="Continuous relevance score (0.0 to 1.0) of generated answer against user question."
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

    external_citations: list[dict | str] = Field(
        default_factory=list,
        description="Structured citation metadata (source, title, url, tool, retrieval_timestamp) from external search."
    )

    # Detailed Latency Breakdown
    latency_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Stage-by-stage latency mapping in seconds."
    )

    query_analysis_latency: float = Field(
        default=0.0,
        description="Latency of query analysis in seconds."
    )

    retrieval_latency: float = Field(
        default=0.0,
        description="Latency of hybrid document retrieval in seconds."
    )

    reranker_latency: float = Field(
        default=0.0,
        description="Latency of Cohere reranking in seconds."
    )

    grader_latency: float = Field(
        default=0.0,
        description="Cumulative latency of evaluator and grading nodes in seconds."
    )

    generation_latency: float = Field(
        default=0.0,
        description="Latency of answer generation in seconds."
    )

    mcp_latency: float = Field(
        default=0.0,
        description="Latency of MCP tools and external search in seconds."
    )

    total_latency: float = Field(
        default=0.0,
        description="Total end-to-end RAG pipeline latency in seconds."
    )