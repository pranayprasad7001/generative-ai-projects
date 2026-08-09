"""Langgraph nodes for RAG workflow + Agent inside generate_content"""

from typing import List
from state.adaptive_state import RAGState
from langchain_classic.schema import Document
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from nodes.schema import ToolUse, RetrievalGrade, QuestionRewrite
from config.mcp_config import MCPToolManager
from config.config import Config
from prompts.rag_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    RETRIEVAL_GRADER_SYSTEM_PROMPT,
    QUESTION_REWRITER_SYSTEM_PROMPT,
    EXTERNAL_SEARCH_SYSTEM_PROMPT,
)

MAX_REWRITES = Config.MAX_REWRITES

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
        self._web_search_agent = None
        self.mcp_manager = MCPToolManager()

    def query_analyzer(self, state: RAGState) -> RAGState:
        """
        Analyze the query and determine which direction to route

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with analysis and which tool to use
        """
        system_prompt = QUERY_ANALYZER_SYSTEM_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        response = self.llm.with_structured_output(ToolUse).invoke(
            prompt.format_messages(question=state.question)
        )

        state.analysis = response.analysis
        state.tool_type = response.tool_type
        return state

    def query_router(self, state: RAGState) -> str:
        """
        Routes the query to the appropriate tool based on the analysis

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with routed query
        """
        if state.tool_type == "vector_search":
            return "vector_search"
        elif state.tool_type == "external":
            return "external_search"
        else:
            return "none_search"

    async def external_search(self, state: RAGState) -> RAGState:
        """
        Perform external search to find relevant information

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved information
        """
        
        system_prompt = EXTERNAL_SEARCH_SYSTEM_PROMPT

        tools = await self.mcp_manager.get_tools()

        if self._web_search_agent is None:
            self._web_search_agent = create_agent(
                self.llm,
                system_prompt=system_prompt,
                tools=tools
            )

        response = await self._web_search_agent.ainvoke({
            "messages": [
                ("user", state.question)
            ]
        })
        
        messages = response.get("messages", [])
        answer = messages[-1].content if messages else response.get("output", "")
        
        state.external_results = answer
        state.answer = answer
        return state

    def none_search(self, state: RAGState) -> RAGState: # TODO: Remove direct answering, web_search agent will suffice
        """
        Perform none search to find relevant information

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved information
        """
        system_prompt = """
            You are the direct-answering component of an autonomous RAG system.

            The user's question has already been classified as not requiring any
            retrieval from the vector store or external search tools.

            Your task is to answer the user's question directly using the information
            available in the conversation and your general knowledge.

            Guidelines:

            1. Do not perform any vector search.
            2. Do not use external search tools.
            3. Do not invent facts or sources.
            4. Answer clearly and directly.
            5. If the user is greeting, thanking, or engaging in casual conversation,
               respond naturally and conversationally.
            6. If the question is a simple factual or conceptual question that does not
               require retrieval, provide a concise and accurate explanation.
            7. If the question cannot be answered reliably without additional
               information, clearly state what information is missing instead of
               fabricating an answer.
            8. Match the level of detail to the user's question.

            Return only the final answer to the user's question.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        response = self.llm.invoke(
            prompt.format_messages(
                question=state.question
            )
        )

        state.answer = response.content
        return state

    def vector_search(self, state: RAGState) -> RAGState:
        """Perform vector search to find relevant documents."""

        retrieved_documents: List[Document] = self.retriever.invoke(state.question)
        state.retrieved_docs = retrieved_documents
        return state

    def grader(self, state: RAGState) -> RAGState:
        """Grade the relevance and sufficiency of retrieved documents."""

        system_prompt = RETRIEVAL_GRADER_SYSTEM_PROMPT

        prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """
            User Question:
            {question}

            Retrieved Documents:
            {documents}
            """
        )])

        documents = "\n\n".join(
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_docs, start=1)
        )

        response = self.llm.with_structured_output(RetrievalGrade).invoke(prompt.format_messages(
                question=state.question,
                documents=documents
            )
        )

        state.grade = response.grade
        state.analysis = response.reasoning

        return state

    def grader_router(self, state: RAGState) -> str:
        """
        Route to next node based on grader output
        """
        if state.grade == "yes":
            return "generate"
        if state.rewrite_count >= MAX_REWRITES: # TODO : after max tries, fallback to external search
            return "generate"
        return "rewriter"
    
    def rewriter(self, state: RAGState) -> RAGState:
        """
        Rewrite the user's question to improve retrieval accuracy
        """
        system_prompt = QUESTION_REWRITER_SYSTEM_PROMPT

        prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """
            Original Question:
            {original_question}

            Current Question:
            {current_question}
            """
        )])

        response = self.llm.with_structured_output(QuestionRewrite).invoke(
            prompt.format_messages(
                original_question=state.original_question,
                current_question=state.question,
            )
        )

        state.question = response.rewritten_question
        state.analysis = response.reasoning
        state.rewrite_count += 1

        return state

    def generator(self, state: RAGState) -> RAGState:
        """"""
        pass

    def hallucination_detector(self, state: RAGState) -> RAGState:
        """"""
        pass

    def answer_relevance_grader(self, state: RAGState) -> RAGState:
        """"""
        pass