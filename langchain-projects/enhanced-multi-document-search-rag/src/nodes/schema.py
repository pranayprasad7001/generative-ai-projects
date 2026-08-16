"""Structured Pydantic Schemas for Adaptive RAG LLM Decision Making.

This module contains output schemas used for structured LLM parsing:
- `ToolUse`: Query classification and routing determination.
- `RetrievalGrade`: Continuous document relevance scoring & decisions.
- `QuestionRewrite`: Conversational query reformulations and coreference reasoning.
- `HallucinationGrade`: Groundedness scoring against provided context.
- `AnswerRelevanceGrade`: Answer-question completeness & semantic alignment.
- `CitationMetadata`: Structured source reference tracking from external MCP tools.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ToolUse(BaseModel):      
    """Schema representing the routing decision and justification from the query analyzer."""
    tool_type: Literal["hybrid_retrieval", "vector_search", "external_search"] = Field(
        ...,
        description="Tool type to use: 'hybrid_retrieval' (dense + BM25 + rerank) or 'external_search'"
    )
    analysis: str = Field(..., description="Analysis of the tool to use")


class RetrievalGrade(BaseModel):  
    """
    Rich retrieval evaluation schema including continuous score, categorical decision, and reasoning.
    """
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance confidence score between 0.0 (irrelevant) and 1.0 (highly relevant and sufficient)."
    )
    decision: Literal["pass", "retry", "rewrite", "fail", "yes", "no"] = Field(
        default="pass",
        description="Grading decision based on relevance score."
    )
    reasoning: str = Field(
        default="Evaluated document relevance.",
        description="Concise explanation for the grading decision and score."
    )
    grade: Optional[str] = Field(
        default=None,
        description="Backward compatibility field for binary grade ('yes' or 'no')."
    )

    @model_validator(mode="after")
    def sync_grade_and_decision(self):
        # If score was NOT explicitly provided, but grade was provided
        if "score" not in self.model_fields_set and self.grade is not None:
            if self.grade.lower() in ("yes", "pass"):
                self.score = 1.0
                self.decision = "pass"
            else:
                self.score = 0.3
                self.decision = "rewrite"

        # Score is strictly authoritative
        if self.score is not None:
            if self.score >= 0.7:
                self.grade = "yes"
                if self.decision not in ("pass", "yes"):
                    self.decision = "pass"
            else:
                self.grade = "no"
                if self.decision in ("pass", "yes"):
                    self.decision = "rewrite"
        return self


class QuestionRewrite(BaseModel): 
    """
    Schema for the rewritten question, including the rewritten query and the reasoning
    behind the rewrite.
    """
    rewritten_question: str = Field(
        description="Improved retrieval-focused version of the question."
    )
    reasoning: str = Field(
        description="Brief explanation of why the rewrite should improve retrieval."
    )


class HallucinationGrade(BaseModel):
    """
    Rich hallucination detection schema including groundedness score, decision, and reasoning.
    """
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Groundedness confidence score between 0.0 (hallucinated) and 1.0 (fully grounded in context)."
    )
    decision: Literal["pass", "retry", "fail", "yes", "no"] = Field(
        default="pass",
        description="Groundedness decision: 'pass' (grounded) or 'retry' (hallucinated)."
    )
    reasoning: str = Field(
        default="Evaluated answer groundedness.",
        description="Concise explanation of the hallucination check."
    )
    grade: Optional[str] = Field(
        default=None,
        description="Backward compatibility field for binary grade ('yes' or 'no')."
    )

    @model_validator(mode="after")
    def sync_grade_and_decision(self):
        if "score" not in self.model_fields_set and self.grade is not None:
            if self.grade.lower() in ("yes", "pass"):
                self.score = 1.0
                self.decision = "pass"
            else:
                self.score = 0.2
                self.decision = "retry"

        # Score is strictly authoritative
        if self.score is not None:
            if self.score >= 0.7:
                self.grade = "yes"
                if self.decision not in ("pass", "yes"):
                    self.decision = "pass"
            else:
                self.grade = "no"
                if self.decision in ("pass", "yes"):
                    self.decision = "retry"
        return self


class AnswerRelevanceGrade(BaseModel):
    """
    Rich answer relevance evaluation schema including relevance score, decision, and reasoning.
    """
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Answer relevance score between 0.0 (irrelevant) and 1.0 (directly and completely answers question)."
    )
    decision: Literal["pass", "retry", "rewrite", "fail", "yes", "no"] = Field(
        default="pass",
        description="Relevance decision: 'pass' (relevant) or 'rewrite' / 'retry' (irrelevant)."
    )
    reasoning: str = Field(
        default="Evaluated answer relevance.",
        description="Concise explanation of the answer relevance check."
    )
    grade: Optional[str] = Field(
        default=None,
        description="Backward compatibility field for binary grade ('yes' or 'no')."
    )

    @model_validator(mode="after")
    def sync_grade_and_decision(self):
        if "score" not in self.model_fields_set and self.grade is not None:
            if self.grade.lower() in ("yes", "pass"):
                self.score = 1.0
                self.decision = "pass"
            else:
                self.score = 0.3
                self.decision = "rewrite"

        # Score is strictly authoritative
        if self.score is not None:
            if self.score >= 0.7:
                self.grade = "yes"
                if self.decision not in ("pass", "yes"):
                    self.decision = "pass"
            else:
                self.grade = "no"
                if self.decision in ("pass", "yes"):
                    self.decision = "rewrite"
        return self


class CitationMetadata(BaseModel):
    """Structured source metadata from external search tools and MCP servers."""
    source: str = Field(default="", description="Name or identifier of the source (e.g. 'Tavily', 'Wikipedia', 'arXiv')")
    title: str = Field(default="", description="Title of the article, paper, or webpage")
    url: str = Field(default="", description="Direct URL to the reference source")
    tool: str = Field(default="", description="Tool/MCP name used for retrieval")
    retrieval_timestamp: str = Field(
        default="",
        description="ISO 8601 UTC timestamp of retrieval"
    )