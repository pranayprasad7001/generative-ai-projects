from pydantic import BaseModel, Field
from typing import Literal

class ToolUse(BaseModel):      
    """Schema representing the routing decision and justification from the query analyzer."""
    tool_type: Literal["vector_search", "external_search"] = Field(..., description="Tool type to use")
    analysis: str = Field(..., description="Analysis of the tool to use")

class RetrievalGrade(BaseModel):  
    """
    Schema for the retrieval grade, indicating whether the retrieved documents
    are relevant and sufficient.
    """
    grade: Literal["yes", "no"] = Field(
        description="Whether the retrieved documents are relevant and sufficient."
    )

    reasoning: str = Field(
        description="Concise explanation for the grading decision."
    )

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
    Schema for the hallucination grade, indicating whether the generated answer
    is fully supported by the retrieved context.
    """
    grade: Literal["yes", "no"] = Field(
        description="Whether the generated answer is fully supported by the context."
    )
    reasoning: str = Field(
        description="Concise explanation of the hallucination check."
    )

class AnswerRelevanceGrade(BaseModel):
    """
    Schema for the answer relevance grade, indicating whether the generated answer
    directly and adequately answers the user's question.
    """
    grade: Literal["yes", "no"] = Field(
        description="Whether the generated answer directly and adequately answers the user's question."
    )
    reasoning: str = Field(
        description="Concise explanation of the answer relevance check."
    )