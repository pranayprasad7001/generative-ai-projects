import logging
from langgraph.graph import StateGraph, START, END
from state.adaptive_state import AdaptiveRAGState
from nodes.adaptive_node import AdaptiveRAGNodes
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds and manages the langgraph workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize the graph builder

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        logger.info("Initializing GraphBuilder with retriever and LLM.")
        self.nodes = AdaptiveRAGNodes(retriever, llm)
        self.graph = None
        self.checkpointer = InMemorySaver()

    def build_graph(self):
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
        builder.add_node("vector_search", self.nodes.vector_search)
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
                "vector_search": "vector_search",
                "external_search": "external_search",
            },
        )

        builder.add_edge("vector_search", "documents_grader")

        builder.add_conditional_edges(
            "documents_grader",
            self.nodes.grader_router,
            {
                "answer_generator": "answer_generator",
                "query_rewriter": "query_rewriter",
                "external_search": "external_search",
            },
        )

        builder.add_edge("query_rewriter", "vector_search")

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

        builder.add_edge("external_search", "output_answer_security_check")
        builder.add_edge("output_answer_security_check", END)

        # Compile the graph
        self.graph = builder.compile(checkpointer=self.checkpointer)
        logger.info("StateGraph workflow successfully compiled.")
        return self.graph

    async def run(self, question: str) -> dict:
        """
        Run the Adaptive RAG workflow with a question

        Args:
            question: Question to ask

        Returns:
            Dictionary with answer and other details
        """
        if self.graph is None:
            logger.info(
                "Graph is not compiled yet. Compiling before run..."
            )
            self.build_graph()

        logger.info(
            "Running Adaptive RAG workflow for query: %s",
            repr(question),
        )

        initial_state = AdaptiveRAGState(question=question)

        result = await self.graph.ainvoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": "default_thread"
                }
            },
        )

        logger.info(
            "Adaptive RAG workflow run completed successfully."
        )

        return result