"""Retrieval and query analysis nodes for Adaptive RAG."""

import inspect
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, List
from state.adaptive_state import AdaptiveRAGState
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from nodes.schema import ToolUse, QuestionRewrite, CitationMetadata
from nodes.guardrails import Guardrails
from prompts.rag_prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    QUESTION_REWRITER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class RetrievalNodes:
    """Nodes responsible for query routing analysis, vector search, MCP search, and question rewriting."""

    def __init__(self, retriever, llm_checker, llm_generator=None, guardrails: Guardrails | None = None):
        self.retriever = retriever
        self.llm_checker = llm_checker
        self.llm_generator = llm_generator if llm_generator is not None else llm_checker
        self.guardrails = guardrails or Guardrails(llm_checker=self.llm_checker, llm_generator=self.llm_generator)
        self.external_search_agent: Any = None

    async def query_analyzer(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """
        Analyze the query and determine which direction to route (vector search vs external search).
        """
        t0 = time.perf_counter()
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

        state.query_analysis = response.analysis
        state.analysis = response.analysis
        state.tool_type = response.tool_type
        elapsed = time.perf_counter() - t0
        state.query_analysis_latency = round(elapsed, 4)
        state.latency_breakdown["query_analysis"] = round(elapsed, 4)
        logger.info("Query Analyzer output - Tool type: %s (%.3fs)", state.tool_type, elapsed)
        logger.debug("Query Analyzer reasoning: %s", state.query_analysis)
        return state

    async def hybrid_retrieval(self, state: AdaptiveRAGState, config: RunnableConfig | None = None) -> AdaptiveRAGState:
        """Perform dense vector retrieval and Cohere reranking to find relevant documents."""
        t0 = time.perf_counter()
        logger.info("Executing hybrid retrieval.")
        logger.debug("Hybrid retrieval query: %s", state.question)

        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        active_retriever = configurable.get("retriever") or self.retriever

        if hasattr(active_retriever, "ainvoke"):
            res = active_retriever.ainvoke(state.question)
            if inspect.isawaitable(res):
                retrieved_documents: List[Document] = await res
            elif isinstance(res, list):
                retrieved_documents = res
            else:
                retrieved_documents = active_retriever.invoke(state.question)
        else:
            retrieved_documents: List[Document] = active_retriever.invoke(state.question)
        state.retrieved_docs = retrieved_documents
        elapsed = time.perf_counter() - t0
        state.retrieval_latency = round(elapsed, 4)
        state.latency_breakdown["hybrid_retrieval"] = round(elapsed, 4)
        logger.info("Hybrid retrieval retrieved %d documents in %.3fs", len(retrieved_documents), elapsed)
        return state

    # Backward compatibility alias
    vector_search = hybrid_retrieval

    async def external_search(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Perform external web/MCP search to find relevant information."""
        t0 = time.perf_counter()
        logger.info("Executing external search.")
        logger.debug("External search query: %s", state.question)

        agent = self.external_search_agent
        if agent is None:
            logger.info("Initializing combined guardrail agent lazily for external search...")
            agent = await self.guardrails.get_combined_guardrail_agent()
            self.external_search_agent = agent
            logger.info("Combined guardrail agent initialized successfully with MCP tools.")

        if agent is None:
            raise RuntimeError("Failed to initialize external search agent.")

        response = await agent.ainvoke(
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

        # Extract structured external citations from ToolMessages
        citations = []
        seen_keys = set()
        retrieval_time = datetime.now(timezone.utc).isoformat()

        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content:
                tool_name = getattr(msg, "name", "") or getattr(msg, "tool_name", "") or "external_search"

                # Check for structured JSON content
                parsed_data = None
                if isinstance(msg.content, (list, dict)):
                    parsed_data = msg.content
                elif isinstance(msg.content, str) and msg.content.strip().startswith(("{", "[")):
                    try:
                        parsed_data = json.loads(msg.content)
                    except Exception:
                        parsed_data = None

                extracted_structured = False
                if isinstance(parsed_data, list):
                    for item in parsed_data:
                        if isinstance(item, dict):
                            item_url = item.get("url") or item.get("link") or ""
                            item_title = item.get("title") or item.get("name") or ""
                            item_src = item.get("source") or tool_name
                            dedup_key = item_url or item_title
                            if dedup_key and dedup_key not in seen_keys:
                                seen_keys.add(dedup_key)
                                extracted_structured = True
                                citations.append(CitationMetadata(
                                    source=item_src,
                                    title=item_title,
                                    url=item_url,
                                    tool=tool_name,
                                    retrieval_timestamp=retrieval_time
                                ).model_dump())
                elif isinstance(parsed_data, dict):
                    results_list = parsed_data.get("results") or parsed_data.get("data")
                    if isinstance(results_list, list):
                        for item in results_list:
                            if isinstance(item, dict):
                                item_url = item.get("url") or item.get("link") or ""
                                item_title = item.get("title") or item.get("name") or ""
                                item_src = item.get("source") or tool_name
                                dedup_key = item_url or item_title
                                if dedup_key and dedup_key not in seen_keys:
                                    seen_keys.add(dedup_key)
                                    extracted_structured = True
                                    citations.append(CitationMetadata(
                                        source=item_src,
                                        title=item_title,
                                        url=item_url,
                                        tool=tool_name,
                                        retrieval_timestamp=retrieval_time
                                    ).model_dump())

                if not extracted_structured:
                    # If no structured list was extracted, parse text content for URLs / references
                    content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                    urls = re.findall(r'https?://[^\s\)\]\"\']+', content_str)
                    for url in urls:
                        if url not in seen_keys:
                            seen_keys.add(url)
                            citations.append(CitationMetadata(
                                source=tool_name,
                                title=tool_name.replace("_", " ").title(),
                                url=url,
                                tool=tool_name,
                                retrieval_timestamp=retrieval_time
                            ).model_dump())

        state.external_citations = citations
        elapsed = time.perf_counter() - t0
        state.mcp_latency = round(elapsed, 4)
        state.latency_breakdown["mcp_external_search"] = round(elapsed, 4)
        logger.info("External search completed in %.3fs. Answer length: %d, Citations found: %d", elapsed, len(answer), len(citations))
        return state

    async def query_rewriter(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """
        Rewrite the user's question to improve retrieval accuracy and resolve conversational coreferences.
        """
        t0 = time.perf_counter()
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
        state.rewrite_explanation = response.reasoning
        state.analysis = response.reasoning
        state.rewrite_count += 1
        state.generate_count = 0
        state.retrieved_docs = []
        elapsed = time.perf_counter() - t0
        state.latency_breakdown["query_rewriter"] = round(elapsed, 4)
        logger.info("Question rewritten successfully in %.3fs. Rewrite count: %d", elapsed, state.rewrite_count)
        logger.debug("Rewritten question content: %s", state.question)
        return state
