"""Langgraph nodes for RAG workflow + Agent inside generate_content"""

import logging
from typing import List
from state.adaptive_state import RAGState
from langchain_classic.schema import Document
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from nodes.schema import ToolUse, RetrievalGrade, QuestionRewrite, HallucinationGrade, AnswerRelevanceGrade
from config.mcp_config import MCPToolManager
from config.config import Config
from langchain_core.messages import HumanMessage, AIMessage
from nodes.guardrails import Guardrails
from prompts.rag_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    RETRIEVAL_GRADER_SYSTEM_PROMPT,
    QUESTION_REWRITER_SYSTEM_PROMPT,
    EXTERNAL_SEARCH_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    HALLUCINATION_DETECTOR_SYSTEM_PROMPT,
    ANSWER_RELEVANCE_GRADER_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes with retriever and llm

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        logger.info("Initializing RAGNodes with retriever and LLM.")
        self.retriever = retriever
        self.llm = llm
        self.guardrails = Guardrails(self.llm)
        self.input_guardrail_agent = self.guardrails.get_input_guardrail_agent()
        self.output_guardrail_agent = self.guardrails.get_output_guardrail_agent()
        self._web_search_agent = None
        self.mcp_manager = MCPToolManager()

    async def input_query_security_check(self, state: RAGState) -> RAGState:
        """Validate the user's query before entering the RAG workflow."""
        logger.info("Running input security check.")
        logger.debug("Input query content: %s", state.question)

        response = await self.input_guardrail_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=state.question)
                ]
            }
        )

        messages = response.get("messages", [])

        if not messages:
            logger.warning("Input security check failed to return messages. Blocking by default.")
            state.query_blocked = True
            state.answer = (
                "I was unable to validate your request. "
                "Please try again."
            )
            return state

        result = messages[-1].content.strip().upper()

        if "BLOCKED:" in result or result == "BLOCKED":
            logger.warning("Input security check BLOCKED the query.")
            state.query_blocked = True
            state.answer = (
                "I cannot process this request. "
                "Please rephrase your question."
            )
        else:
            logger.info("Input security check passed.")
            state.query_blocked = False

        return state

    def input_query_security_router(self, state: RAGState) -> str:
        """Route the workflow based on the input security check."""
        logger.info("Routing from input security check. Query blocked: %s", state.query_blocked)
        if state.query_blocked:
            return "end"
        return "query_analyzer"

    async def output_answer_security_check(self, state: RAGState) -> RAGState:
        """
        Review and safely rewrite the generated answer.

        The output guardrail agent performs one LLM call.
        The middleware performs deterministic post-processing only.
        """
        if not state.answer:
            logger.debug("Output security check skipped (empty answer).")
            return state

        logger.info("Running output security check.")
        response = await self.output_guardrail_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=state.answer)
                ]
            }
        )

        messages = response.get("messages", [])

        if messages:
            final_message = messages[-1]

            if isinstance(final_message, AIMessage):
                state.answer = final_message.content

        logger.info("Output security check completed.")
        return state

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

        logger.info("Analyzing query.")
        logger.debug("Query content: %s", state.question)
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_ANALYZER_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        response = self.llm.with_structured_output(ToolUse).invoke(
            prompt.format_messages(question=state.question)
        )

        state.analysis = response.analysis
        state.tool_type = response.tool_type
        logger.info("Query Analyzer output - Tool type: %s", state.tool_type)
        logger.debug("Query Analyzer reasoning: %s", state.analysis)
        return state

    def query_router(self, state: RAGState) -> str:
        """
        Routes the query to the appropriate tool based on the analysis

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with routed query
        """
        logger.info("Routing query to: %s", state.tool_type)
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
        logger.info("Executing external search.")
        logger.debug("External search query: %s", state.question)
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
        logger.info("External search completed. Answer length: %d", len(answer))
        return state

    def vector_search(self, state: RAGState) -> RAGState:
        """Perform vector search to find relevant documents."""
        logger.info("Executing vector search.")
        logger.debug("Vector search query: %s", state.question)
        retrieved_documents: List[Document] = self.retriever.invoke(state.question)
        state.retrieved_docs = retrieved_documents
        logger.info("Vector search retrieved %d documents", len(retrieved_documents))
        return state

    def grader(self, state: RAGState) -> RAGState:
        """Grade the relevance and sufficiency of retrieved documents."""
        logger.info("Grading %d retrieved documents.", len(state.retrieved_docs))
        logger.debug("Grading documents for query: %s", state.question)

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
        logger.info("Retrieval grader output - Grade: %s", state.retrieval_grade)
        logger.debug("Retrieval grader reasoning: %s", state.analysis)
        return state
    
    def rewriter(self, state: RAGState) -> RAGState:
        """
        Rewrite the user's question to improve retrieval accuracy
        """
        logger.info("Rewriting question.")
        logger.debug("Original question: %s, Current question: %s", state.original_question, state.question)

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
        logger.info("Question rewritten successfully. Rewrite count: %d", state.rewrite_count)
        logger.debug("Rewritten question content: %s", state.question)
        return state

    def generator(self, state: RAGState) -> RAGState:
        """Generate answer based on retrieved documents."""
        logger.info("Generating answer. Generate count: %d", state.generate_count + 1)

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
        logger.info("Answer generated successfully.")
        return state

    def hallucination_detector(self, state: RAGState) -> RAGState:
        """Detects hallucinations in the generated answer."""
        logger.debug("Running hallucination check for generated answer.")

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
        logger.info("Hallucination check output - Grade: %s", state.hallucination_grade)
        logger.debug("Hallucination check reasoning: %s", state.analysis)
        return state

    def answer_relevance_grader(self, state: RAGState) -> RAGState:
        """Grades answer relevance to the question."""
        logger.debug("Running answer relevance check.")

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
        logger.info("Answer relevance output - Grade: %s", state.answer_relevance_grade)
        logger.debug("Answer relevance reasoning: %s", state.analysis)
        return state

    def grader_router(self, state: RAGState) -> str:
        """Route to the next node based on retrieval grading."""
        logger.info("Routing from retrieval grader. Grade: %s, Rewrite count: %d", state.retrieval_grade, state.rewrite_count)
        if state.retrieval_grade == "yes":
            return "generator"
        if state.rewrite_count >= Config.MAX_REWRITES:
            return "external_search"
        return "rewriter"

    def answer_relevance_router(self, state: RAGState) -> str:
        """
        Route to next node based on answer relevance grader output
        """
        logger.info("Routing from relevance grader. Grade: %s, Rewrite count: %d", state.answer_relevance_grade, state.rewrite_count)
        if state.answer_relevance_grade == "yes":
            return "end"
        if state.rewrite_count >= Config.MAX_REWRITES:
            return "external_search"
        return "rewriter"

    def hallucination_router(self, state: RAGState) -> str:
        """Route based on hallucination detection."""
        logger.info("Routing from hallucination detector. Grade: %s, Generate count: %d", state.hallucination_grade, state.generate_count)
        if state.hallucination_grade == "yes":
            return "answer_relevance_grader"
        if state.generate_count >= Config.MAX_GENERATIONS:
            return "external_search"
        return "generator"
    
    