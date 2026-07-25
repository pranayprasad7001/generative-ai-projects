from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource and extract search keywords."""

    datasource: Literal["vectorstore", "wiki_search", "arxiv_search"] = Field(
        ...,
        description="Given a user question, choose to route it to wikipedia, arxiv or a vectorstore.",
    )
    query: str = Field(
        ...,
        description="The core keyword query extracted from the user's question, stripping out all conversational filler (e.g. 'tell me about', 'find', 'search for').",
    )


def route_to_datasource(llm):
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system_prompt = """You are an expert at routing user questions to a vectorstore, arxiv or wikipedia, and extracting the clean search keywords.
        The vectorstore contains documents related to harness, thinking and attacks on llms.
        Use wikipedia for general knowledge queries and arxiv for research papers.
        Make sure the 'query' field contains ONLY the clean search keywords and no conversational phrases.
        If the query refers to a specific research paper title, wrap that title in double quotes to force an exact phrase search (e.g. '"Attention Is All You Need"')."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router    
    return question_router