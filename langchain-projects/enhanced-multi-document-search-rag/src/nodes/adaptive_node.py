"""Langgraph nodes for RAG workflow + Agent inside generate_content"""

from typing import List
from state.adaptive_state import RAGState
from langchain_classic.schema import Document
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from nodes.schema import ToolUse, RetrievalGrade, QuestionRewrite, HallucinationGrade, AnswerRelevanceGrade
from config.mcp_config import MCPToolManager
from config.config import Config
from prompts.rag_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    RETRIEVAL_GRADER_SYSTEM_PROMPT,
    QUESTION_REWRITER_SYSTEM_PROMPT,
    EXTERNAL_SEARCH_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    HALLUCINATION_DETECTOR_SYSTEM_PROMPT,
    ANSWER_RELEVANCE_GRADER_SYSTEM_PROMPT,
)

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
        if not state.original_question:
            state.original_question = state.question

        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_ANALYZER_SYSTEM_PROMPT),
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

        if state.tool_type == "external_search":
            return "external_search"

        raise ValueError(
            f"Invalid tool_type: {state.tool_type}"
        )

    async def external_search(self, state: RAGState) -> RAGState:
        """
        Perform external search to find relevant information

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved information
        """

        tools = await self.mcp_manager.get_tools()

        if self._web_search_agent is None:
            self._web_search_agent = create_agent(
                self.llm,
                system_prompt=EXTERNAL_SEARCH_SYSTEM_PROMPT,
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

    def vector_search(self, state: RAGState) -> RAGState:
        """Perform vector search to find relevant documents."""

        retrieved_documents: List[Document] = self.retriever.invoke(state.question)
        state.retrieved_docs = retrieved_documents
        return state

    def grader(self, state: RAGState) -> RAGState:
        """Grade the relevance and sufficiency of retrieved documents."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", RETRIEVAL_GRADER_SYSTEM_PROMPT),
            (
                "human",
            """
            User Question:
            {question}

            Retrieved Documents:
            {documents}
            """
            ),
        ])

        documents = "\n\n".join(
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_docs, start=1)
        )

        response = self.llm.with_structured_output(
            RetrievalGrade
        ).invoke(
            prompt.format_messages(
                question=state.question,
                documents=documents,
            )
        )

        state.retrieval_grade = response.grade
        state.analysis = response.reasoning

        return state
    
    def rewriter(self, state: RAGState) -> RAGState:
        """
        Rewrite the user's question to improve retrieval accuracy
        """

        prompt = ChatPromptTemplate.from_messages([
        ("system", QUESTION_REWRITER_SYSTEM_PROMPT),
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
        state.generate_count = 0
        state.retrieved_docs = []
        return state

    def generator(self, state: RAGState) -> RAGState:
        """Generate answer based on retrieved documents."""

        documents_content: str = "\n\n".join(
            doc.page_content for doc in state.retrieved_docs
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM_PROMPT),
            (
                "human",
                """
                User Question:
                {question}

                Retrieved Documents:
                {documents}
                """
            ),
        ])

        response = self.llm.invoke(prompt.format_messages(question=state.question, documents=documents_content))
        state.answer = response.content
        state.generate_count += 1
        return state

    def hallucination_detector(self, state: RAGState) -> RAGState:
        """Detects hallucinations in the generated answer."""

        documents_content: str = "\n\n".join(
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_docs, start=1)
        )

        prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_DETECTOR_SYSTEM_PROMPT),
        (
            "human",
            """
            User Question:
            {question}

            Retrieved Documents:
            {documents}

            Generated Answer:
            {answer}
            """
        ),
        ])
        
        response = self.llm.with_structured_output(HallucinationGrade).invoke(
            prompt.format_messages(question=state.question, documents=documents_content, answer=state.answer))

        state.hallucination_grade = response.grade
        state.analysis = response.reasoning

        return state

    def answer_relevance_grader(self, state: RAGState) -> RAGState:
        """Grades answer relevance to the question."""

        prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_RELEVANCE_GRADER_SYSTEM_PROMPT),
        (
            "human",
            """
            User Question:
            {question}

            Generated Answer:
            {answer}
            """
        ),
        ])

        response = self.llm.with_structured_output(AnswerRelevanceGrade).invoke(
            prompt.format_messages(question=state.question, answer=state.answer)
        )

        state.answer_relevance_grade = response.grade
        state.analysis = response.reasoning

        return state

    def grader_router(self, state: RAGState) -> str:
        """Route to the next node based on retrieval grading."""

        if state.retrieval_grade == "yes":
            return "generator"
        if state.rewrite_count >= Config.MAX_REWRITES:
            return "external_search"
        return "rewriter"

    def answer_relevance_router(self, state: RAGState) -> str:
        """
        Route to next node based on answer relevance grader output
        """
        if state.answer_relevance_grade == "yes":
            return "end"
        if state.rewrite_count >= Config.MAX_REWRITES:
            return "external_search"
        return "rewriter"

    def hallucination_router(self, state: RAGState) -> str:
        """Route based on hallucination detection."""
        if state.hallucination_grade == "yes":
            return "answer_relevance_grader"
        if state.generate_count >= Config.MAX_GENERATIONS:
            return "external_search"
        return "generator"
    
    