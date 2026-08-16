"""Langgraph nodes for RAG workflow + Agent inside generate_content"""

import inspect
import logging
import re
from typing import List
from state.adaptive_state import AdaptiveRAGState
from langchain_classic.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from nodes.schema import ToolUse, RetrievalGrade, QuestionRewrite, HallucinationGrade, AnswerRelevanceGrade
from config.llmgateway_config import Config
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from nodes.guardrails import Guardrails
from prompts.rag_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    RETRIEVAL_GRADER_SYSTEM_PROMPT,
    QUESTION_REWRITER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_REGENERATION_SYSTEM_PROMPT,
    HALLUCINATION_DETECTOR_SYSTEM_PROMPT,
    ANSWER_RELEVANCE_GRADER_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class AdaptiveRAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm_generator=None, llm_checker=None, llm=None):
        """
        Initialize RAG nodes with retriever, llm_generator, and llm_checker.

        Args:
            retriever: Document retriever instance
            llm_generator: Language model instance for answer generation
            llm_checker: Language model instance for structured checks/routing/grading
            llm: Fallback language model instance (for backward compatibility)
        """
        logger.info("Initializing RAGNodes with retriever, LLM Generator, and LLM Checker.")
        self.retriever = retriever
        
        # Handle backward compatibility / flexible arguments
        if llm_generator is None and llm is not None:
            self.llm_generator = llm
            self.llm_checker = llm_checker if llm_checker is not None else llm
        elif llm_generator is not None and llm_checker is None:
            # Single LLM passed as 2nd positional argument
            self.llm_generator = llm_generator
            self.llm_checker = llm_generator
        else:
            self.llm_generator = llm_generator
            self.llm_checker = llm_checker

        # Backward compatibility alias
        self.llm = self.llm_generator

        self.guardrails = Guardrails(llm_checker=self.llm_checker, llm_generator=self.llm_generator)
        self.input_guardrail_agent = self.guardrails.get_input_guardrail_agent()
        self.output_guardrail_agent = self.guardrails.get_output_guardrail_agent()
        self.external_search_agent = None

    async def input_query_security_check(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
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
            state.answer = "This request could not be processed due to security filtering."
            return state

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        content_clean = content.strip().upper()

        if "BLOCKED" in content_clean or content_clean.startswith("UNSAFE"):
            logger.warning("Input query blocked by security guardrail.")
            state.query_blocked = True
            state.answer = "I cannot process this request. Please rephrase your question."
        elif "SAFE" in content_clean or "PASSED" in content_clean:
            logger.info("Input query passed security check.")
            state.query_blocked = False
        else:
            # Check for explicit safety denial phrasing only
            denial_patterns = ["REQUEST IS UNSAFE", "DENIED", "ACCESS DENIED", "POLICY VIOLATION DETECTED"]
            if any(pattern in content_clean for pattern in denial_patterns):
                logger.warning("Input query denied by safety guardrail agent.")
                state.query_blocked = True
                state.answer = "I cannot process this request. Please rephrase your question."
            else:
                logger.info("Input query treated as SAFE by guardrail agent.")
                state.query_blocked = False

        return state

    def input_query_security_router(self, state: AdaptiveRAGState) -> str:
        """Route based on input security check result."""
        if state.query_blocked:
            logger.warning("Routing to END due to security violation.")
            return "end"
        logger.info("Routing to query_analyzer after successful security check.")
        return "query_analyzer"

    async def output_answer_security_check(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Validate and sanitize the final generated answer."""
        logger.info("Running output security check.")
        logger.debug("Raw generated answer: %s", state.answer)

        if not state.answer:
            logger.warning("Empty answer provided to output security check.")
            return state

        response = await self.output_guardrail_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=f"Review the following answer for safety and policy compliance:\n\n{state.answer}")
                ]
            }
        )

        messages = response.get("messages", [])

        if not messages:
            logger.warning("Output security check failed to return messages. Retaining raw answer.")
            return state

        last_message = messages[-1]
        sanitized_answer = last_message.content if hasattr(last_message, "content") else str(last_message)

        if sanitized_answer and sanitized_answer.strip():
            state.answer = sanitized_answer.strip()
            logger.info("Output answer validated/sanitized successfully.")
        else:
            logger.warning("Output security check returned empty content. Retaining raw answer.")

        return state

    async def query_analyzer(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
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

        response = await self.llm_checker.with_structured_output(ToolUse).ainvoke(
            prompt.format_messages(question=state.question)
        )

        state.analysis = response.analysis
        state.tool_type = response.tool_type
        logger.info("Query Analyzer output - Tool type: %s", state.tool_type)
        logger.debug("Query Analyzer reasoning: %s", state.analysis)
        return state

    async def external_search(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """
        Perform external search to find relevant information

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved information
        """
        logger.info("Executing external search.")
        logger.debug("External search query: %s", state.question)

        if self.external_search_agent is None:
            logger.info("Initializing combined guardrail agent lazily for external search...")
            self.external_search_agent = await self.guardrails.get_combined_guardrail_agent()
            logger.info("Combined guardrail agent initialized successfully with MCP tools.")

        response = await self.external_search_agent.ainvoke(
            {
                "messages": [
                    ("user", state.question)
                ]
            },
            config={"recursion_limit": 100}
        )
        
        messages = response.get("messages", [])
        raw_answer = messages[-1].content if messages else response.get("output", "")
        if isinstance(raw_answer, list):
            answer_parts = []
            for part in raw_answer:
                if isinstance(part, dict) and "text" in part:
                    answer_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    answer_parts.append(part)
                else:
                    answer_parts.append(str(part))
            answer = "\n".join(answer_parts)
        elif isinstance(raw_answer, str):
            answer = raw_answer
        else:
            answer = str(raw_answer) if raw_answer is not None else ""
        
        state.external_results = answer
        state.answer = answer

        # Extract external citations (unique URLs) from ToolMessages
        citations = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content:
                if isinstance(msg.content, str):
                    content_str = msg.content
                elif isinstance(msg.content, list):
                    parts = []
                    for item in msg.content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(str(item["text"]))
                        elif isinstance(item, str):
                            parts.append(item)
                        else:
                            parts.append(str(item))
                    content_str = "\n".join(parts)
                else:
                    content_str = str(msg.content)

                urls = re.findall(r'https?://[^\s\)\]\"\']+', content_str)
                for url in urls:
                    if url not in citations:
                        citations.append(url)

        state.external_citations = citations
        logger.info("External search completed. Answer length: %d, Citations found: %d", len(answer), len(citations))
        return state

    async def vector_search(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Perform vector search to find relevant documents."""
        logger.info("Executing vector search.")
        logger.debug("Vector search query: %s", state.question)
        if hasattr(self.retriever, "ainvoke"):
            res = self.retriever.ainvoke(state.question)
            if inspect.isawaitable(res):
                retrieved_documents: List[Document] = await res
            elif isinstance(res, list):
                retrieved_documents = res
            else:
                retrieved_documents = self.retriever.invoke(state.question)
        else:
            retrieved_documents: List[Document] = self.retriever.invoke(state.question)
        state.retrieved_docs = retrieved_documents
        logger.info("Vector search retrieved %d documents", len(retrieved_documents))
        return state

    async def documents_grader(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
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

        response = await self.llm_checker.with_structured_output(
            RetrievalGrade
        ).ainvoke(
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
    
    async def query_rewriter(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """
        Rewrite the user's question to improve retrieval accuracy and resolve conversational coreferences.
        """
        logger.info("Rewriting question.")
        logger.debug("Original question: %s, Current question: %s", state.original_question, state.question)

        chat_history_str = "No previous conversation history."
        if state.messages:
            lines = []
            for msg in state.messages[-6:]:
                role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
                lines.append(f"{role}: {getattr(msg, 'content', str(msg))}")
            if lines:
                chat_history_str = "\n".join(lines)

        prompt = ChatPromptTemplate.from_messages([
            ("system", QUESTION_REWRITER_SYSTEM_PROMPT),
            (
                "human",
                """
                Conversation History:
                {chat_history}

                Original Question:
                {original_question}

                Current Question:
                {current_question}
                """
            )
        ])

        response = await self.llm_checker.with_structured_output(QuestionRewrite).ainvoke(
            prompt.format_messages(
                chat_history=chat_history_str,
                original_question=state.original_question or state.question,
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

    async def answer_generator(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Generate or regenerate answer based on retrieved documents and self-correction critique."""
        logger.info("Generating answer. Generate count: %d", state.generate_count + 1)

        context_sections = []
        if state.retrieved_docs:
            doc_str = "\n\n".join(
                f"[Document {i}]\n{doc.page_content}"
                for i, doc in enumerate(state.retrieved_docs, start=1)
            )
            context_sections.append(f"Retrieved Documents:\n{doc_str}")
        if state.external_results:
            context_sections.append(f"External Search Results:\n{state.external_results}")

        documents_content: str = "\n\n".join(context_sections) if context_sections else "No relevant documents found."

        # Self-correction regeneration branch if previously flagged for hallucination
        if state.generate_count > 0 and state.hallucination_grade == "no" and state.analysis:
            logger.info("Executing critique-aware self-correction answer regeneration.")
            prompt = ChatPromptTemplate.from_messages([
                ("system", GENERATOR_REGENERATION_SYSTEM_PROMPT),
                (
                    "human",
                    """
                    User Question:
                    {question}

                    Retrieved Context:
                    {context}

                    Previous Draft Answer:
                    {previous_answer}

                    Hallucination Critique & Feedback:
                    {critique}
                    """
                ),
            ])
            response = await self.llm_generator.bind(
                extra_body={"cache": {"use-cache": False}}
            ).ainvoke(
                prompt.format_messages(
                    question=state.question,
                    context=documents_content,
                    previous_answer=state.answer,
                    critique=state.analysis
                )
            )
        else:
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

            response = await self.llm_generator.bind(
                extra_body={"cache": {"use-cache": True, "ttl": 1800}}
            ).ainvoke(prompt.format_messages(question=state.question, documents=documents_content))

        state.answer = response.content
        state.generate_count += 1
        logger.info("Answer generated successfully. Total generations: %d", state.generate_count)
        return state

    async def hallucination_detector(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Detects hallucinations in the generated answer against both local docs and external search results."""
        logger.debug("Running hallucination check for generated answer.")

        context_sections = []
        if state.retrieved_docs:
            doc_str = "\n\n".join(
                f"[Document {i}]\n{doc.page_content}"
                for i, doc in enumerate(state.retrieved_docs, start=1)
            )
            context_sections.append(f"Retrieved Documents:\n{doc_str}")
        if state.external_results:
            context_sections.append(f"External Search Results:\n{state.external_results}")

        documents_content: str = "\n\n".join(context_sections) if context_sections else "No relevant context provided."

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
        
        response = await self.llm_checker.with_structured_output(HallucinationGrade).ainvoke(
            prompt.format_messages(question=state.question, documents=documents_content, answer=state.answer))

        state.hallucination_grade = response.grade
        state.analysis = response.reasoning
        logger.info("Hallucination check output - Grade: %s", state.hallucination_grade)
        logger.debug("Hallucination check reasoning: %s", state.analysis)
        return state

    async def answer_relevance_grader(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
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

        response = await self.llm_checker.with_structured_output(AnswerRelevanceGrade).ainvoke(
            prompt.format_messages(question=state.question, answer=state.answer)
        )

        state.answer_relevance_grade = response.grade
        state.analysis = response.reasoning
        logger.info("Answer relevance output - Grade: %s", state.answer_relevance_grade)
        logger.debug("Answer relevance reasoning: %s", state.analysis)
        return state


    def query_router(self, state: AdaptiveRAGState) -> str:
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

    def grader_router(self, state: AdaptiveRAGState) -> str:
        """Route to the next node based on retrieval grading."""
        logger.info("Routing from retrieval grader. Grade: %s, Rewrite count: %d", state.retrieval_grade, state.rewrite_count)
        if state.retrieval_grade == "yes":
            return "answer_generator"
        if state.rewrite_count >= Config.MAX_REWRITES:
            return "external_search"
        return "query_rewriter"

    def hallucination_router(self, state: AdaptiveRAGState) -> str:
        """Route based on hallucination detection."""
        logger.info("Routing from hallucination detector. Grade: %s, Generate count: %d", state.hallucination_grade, state.generate_count)
        if state.hallucination_grade == "yes":
            return "answer_relevance_grader"
        if state.generate_count >= Config.MAX_GENERATIONS:
            if state.tool_type == "external_search" or state.external_results:
                return "output_answer_security_check"
            return "external_search"
        return "answer_generator"

    def answer_relevance_router(self, state: AdaptiveRAGState) -> str:
        """
        Route to next node based on answer relevance grader output
        """
        logger.info("Routing from relevance grader. Grade: %s, Rewrite count: %d", state.answer_relevance_grade, state.rewrite_count)
        if state.answer_relevance_grade == "yes":
            return "output_answer_security_check"
        if state.rewrite_count >= Config.MAX_REWRITES or state.tool_type == "external_search" or state.external_results:
            if state.tool_type == "external_search" or state.external_results:
                return "output_answer_security_check"
            return "external_search"
        return "query_rewriter"