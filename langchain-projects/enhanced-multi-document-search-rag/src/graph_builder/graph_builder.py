import logging
from langgraph.graph import StateGraph, START, END
from state.rag_state import RAGState
from nodes.nodes import RAGNodes

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
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build_graph(self):
        """
        Build the RAG workflow graph

        Returns:
        Compiled graph instance
        """
        logger.info("Building StateGraph workflow...")
        # Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_response)
        
        # Add edges
        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        # Compile the graph
        self.graph = builder.compile()
        logger.info("StateGraph workflow successfully compiled.")
        return self.graph

    def run(self, question: str) -> dict:
        """
        Run the RAG workflow with a question

        Args:
            question: Question to ask

        Returns:
            Dictionary with answer and other details
        """
        if self.graph is None:
            logger.info("Graph is not compiled yet. Compiling before run...")
            self.build_graph()

        logger.info("Running RAG workflow for query: %s", repr(question))
        initial_state = RAGState(question=question)
        result = self.graph.invoke(initial_state)
        logger.info("RAG workflow run completed successfully.")
        return result