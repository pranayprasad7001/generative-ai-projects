import time
import uuid
import logging
from typing import Optional, Any
from langgraph.graph import StateGraph, START, END
from state.adaptive_state import AdaptiveRAGState
from nodes.adaptive_node import AdaptiveRAGNodes
from langgraph.checkpoint.memory import InMemorySaver
from config.cost_callback import CostTrackingCallbackHandler

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds and manages the langgraph workflow"""

    def __init__(self, retriever, llm_generator=None, llm_checker=None, llm=None):
        """
        Initialize the graph builder

        Args:
            retriever: Document retriever instance
            llm_generator: Language model instance for answer generation
            llm_checker: Language model instance for checks/structured outputs
            llm: Fallback language model instance (for backward compatibility)
        """
        logger.info("Initializing GraphBuilder with retriever, LLM Generator, and LLM Checker.")
        self.nodes = AdaptiveRAGNodes(
            retriever=retriever,
            llm_generator=llm_generator,
            llm_checker=llm_checker,
            llm=llm
        )
        self.graph = None
        self.checkpointer = InMemorySaver()

    def build_graph(self, use_checkpointer: bool = True):
        """
        Build the RAG workflow graph

        Returns:
        Compiled graph instance
        """
        logger.info("Building StateGraph workflow...")
        # Create state graph
        builder = StateGraph(AdaptiveRAGState)

        # Add nodes
        builder.add_node("input_query_security_check", self.nodes.input_query_security_check)
        builder.add_node("query_analyzer", self.nodes.query_analyzer)
        builder.add_node("hybrid_retrieval", self.nodes.hybrid_retrieval)
        builder.add_node("documents_grader", self.nodes.documents_grader)
        builder.add_node("query_rewriter", self.nodes.query_rewriter)
        builder.add_node("answer_generator", self.nodes.answer_generator)
        builder.add_node("hallucination_detector", self.nodes.hallucination_detector)
        builder.add_node("answer_relevance_grader", self.nodes.answer_relevance_grader)
        builder.add_node("external_search", self.nodes.external_search)
        builder.add_node("output_answer_security_check", self.nodes.output_answer_security_check)
        
        # Add edges
        builder.add_edge(START, "input_query_security_check")

        builder.add_conditional_edges(
            "input_query_security_check",
            self.nodes.input_query_security_router,
            {
                "end": END,
                "query_analyzer": "query_analyzer",
            },
        )

        builder.add_conditional_edges(
            "query_analyzer",
            self.nodes.query_router,
            {
                "hybrid_retrieval": "hybrid_retrieval",
                "external_search": "external_search",
            },
        )

        builder.add_edge("hybrid_retrieval", "documents_grader")

        builder.add_conditional_edges(
            "documents_grader",
            self.nodes.grader_router,
            {
                "answer_generator": "answer_generator",
                "query_rewriter": "query_rewriter",
                "external_search": "external_search",
            },
        )

        builder.add_edge("query_rewriter", "hybrid_retrieval")

        builder.add_edge("answer_generator", "hallucination_detector")

        builder.add_conditional_edges(
            "hallucination_detector",
            self.nodes.hallucination_router,
            {
                "answer_relevance_grader": "answer_relevance_grader",
                "answer_generator": "answer_generator",
                "external_search": "external_search",
            },
        )

        builder.add_conditional_edges(
            "answer_relevance_grader",
            self.nodes.answer_relevance_router,
            {
                "output_answer_security_check": "output_answer_security_check",
                "query_rewriter": "query_rewriter",
                "external_search": "external_search",
            },
        )

        builder.add_edge("external_search", "hallucination_detector")
        builder.add_edge("output_answer_security_check", END)

        # Compile the graph
        if use_checkpointer and self.checkpointer:
            self.graph = builder.compile(checkpointer=self.checkpointer)
        else:
            self.graph = builder.compile()
        logger.info("StateGraph workflow successfully compiled.")
        return self.graph

    def clear_checkpointer(self):
        """Reset in-memory checkpointer state to prevent memory growth."""
        self.checkpointer = InMemorySaver()
        if self.graph is not None:
            self.build_graph()
        logger.info("Checkpointer state cleared.")

    async def run(
        self,
        question: str,
        thread_id: Optional[str] = None,
        messages: Optional[list] = None,
        retriever: Optional[Any] = None
    ) -> dict:
        """
        Run the Adaptive RAG workflow with a question and optional conversation history

        Args:
            question: Question to ask
            thread_id: Optional thread identifier (defaults to 'default_session')
            messages: Optional list of past BaseMessages for multi-turn history
            retriever: Optional per-query retriever to avoid mutating shared graph state

        Returns:
            Dictionary with answer and other details
        """
        if self.graph is None:
            logger.info(
                "Graph is not compiled yet. Compiling before run..."
            )
            self.build_graph()

        resolved_thread_id = thread_id or "default_session"

        logger.info(
            "Running Adaptive RAG workflow for query: %s (thread_id: %s, history_messages: %d)",
            repr(question),
            resolved_thread_id,
            len(messages) if messages else 0
        )

        initial_state = AdaptiveRAGState(
            question=question,
            messages=messages or []
        )
        cost_callback = CostTrackingCallbackHandler()
        start_time = time.perf_counter()

        configurable = {
            "thread_id": resolved_thread_id
        }
        if retriever is not None:
            configurable["retriever"] = retriever

        try:
            result = await self.graph.ainvoke(
                initial_state,
                config={
                    "configurable": configurable,
                    "callbacks": [cost_callback]
                },
            )

            total_elapsed = round(time.perf_counter() - start_time, 4)

            # Update the result state with total latency and cost
            if isinstance(result, dict):
                result["total_cost"] = cost_callback.total_cost
                result["total_latency"] = total_elapsed
                breakdown = result.get("latency_breakdown", {})
                breakdown["total"] = total_elapsed
                result["latency_breakdown"] = breakdown
            elif hasattr(result, "total_cost"):
                try:
                    result.total_cost = cost_callback.total_cost
                    result.total_latency = total_elapsed
                    result.latency_breakdown["total"] = total_elapsed
                except Exception:
                    pass

            logger.info(
                "Adaptive RAG workflow run completed in %.3fs. Cumulative Cost: $%s",
                total_elapsed,
                cost_callback.total_cost,
            )
            return result

        except Exception as e:
            logger.error("Error during Adaptive RAG workflow execution: %s", str(e), exc_info=True)
            total_elapsed = round(time.perf_counter() - start_time, 4)
            return {
                "question": question,
                "answer": "⚠️ An error occurred while processing your request. Please try again or rephrase your query.",
                "retrieved_docs": [],
                "external_citations": [],
                "total_cost": cost_callback.total_cost,
                "total_latency": total_elapsed,
                "latency_breakdown": {"total": total_elapsed}
            }