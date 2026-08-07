"""Langgraph nodes for RAG workflow + Agent inside generate_content"""

import builtins
import uuid
builtins.uuid = uuid

from typing import List, Optional
from state.rag_state import RAGState
from langchain_classic.schema import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes with retriever and llm

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents using the retriever

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved documents
        """
        retrieved_docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=retrieved_docs
        )

    def _build_tools(self) -> List[Tool]: 
        """Build tools for the agent"""
        
        def retriever_tool(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []

            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}") 
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name='retriever',
            func=retriever_tool,
            description='Use this tool to search for documents relevant to the user query. Returns up to 8 best matching documents.'
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang='en')
        )

        wikipedia_tool = Tool(
            name='wikipedia',
            func=wiki.run,
            description='Use this tool to search for general knowledge information. Returns up to 3 best matching documents.'
        )
        
        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """Agent with tools"""
        tools = self._build_tools()
        
        system_prompt = (
            "You are a helpful RAG Agent"
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge."
            "Return only the final useful answer."
        )

        self._agent = create_agent(model=self.llm, tools=tools, system_prompt=system_prompt)

    def generate_response(self, state: RAGState) -> RAGState:
        """
        Generate answer using the agent with retriever and wikipedia

        Args:
            state: Current RAG state with retrieved documents

        Returns:
            Updated RAG state with generated answer
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        messages = result.get("messages", []) 
        answer: Optional[str] = None

        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer"
        )