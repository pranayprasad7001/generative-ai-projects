import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from state.adaptive_state import AdaptiveRAGState

class TestGraphBuilder(unittest.IsolatedAsyncioTestCase):

    @patch("nodes.adaptive_node.Guardrails")
    def setUp(self, mock_guardrails_class):
        # Mock Guardrails instance to prevent MCP connection attempts or agent builds
        self.mock_guardrails = MagicMock()
        mock_guardrails_class.return_value = self.mock_guardrails
        
        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        
        from graph_builder.adaptive_graph_builder import GraphBuilder
        self.builder = GraphBuilder(self.mock_retriever, self.mock_llm)

    def test_build_graph(self):
        graph = self.builder.build_graph(use_checkpointer=False)
        self.assertIsNotNone(graph)
        self.assertIsNotNone(self.builder.graph)
        
        # Verify compiled graph has the nodes we expect
        # We can check graph.nodes dictionary keys
        expected_nodes = [
            "input_query_security_check",
            "query_analyzer",
            "vector_search",
            "documents_grader",
            "query_rewriter",
            "answer_generator",
            "hallucination_detector",
            "answer_relevance_grader",
            "external_search",
            "output_answer_security_check"
        ]
        for node in expected_nodes:
            self.assertIn(node, self.builder.graph.nodes)

    @patch("graph_builder.adaptive_graph_builder.CostTrackingCallbackHandler")
    async def test_run_success(self, mock_callback_class):
        mock_callback = MagicMock()
        mock_callback.total_cost = 0.0015
        mock_callback_class.return_value = mock_callback

        # Compile the graph
        self.builder.build_graph(use_checkpointer=False)
        
        # Mock the graph invocation
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "question": "What is AI?",
            "answer": "Artificial Intelligence refers to...",
            "retrieved_docs": [],
            "external_citations": []
        }
        self.builder.graph = mock_graph

        result = await self.builder.run("What is AI?")

        self.assertEqual(result["question"], "What is AI?")
        self.assertEqual(result["answer"], "Artificial Intelligence refers to...")
        self.assertEqual(result["total_cost"], 0.0015)
        mock_graph.ainvoke.assert_called_once()

    @patch("graph_builder.adaptive_graph_builder.CostTrackingCallbackHandler")
    async def test_run_error_fallback(self, mock_callback_class):
        mock_callback = MagicMock()
        mock_callback.total_cost = 0.0
        mock_callback_class.return_value = mock_callback

        self.builder.build_graph(use_checkpointer=False)
        
        # Make invocation fail
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = Exception("Connection Refused")
        self.builder.graph = mock_graph

        result = await self.builder.run("What is AI?")

        self.assertEqual(result["question"], "What is AI?")
        self.assertIn("error occurred during request processing", result["answer"])
        self.assertEqual(result["total_cost"], 0.0)

if __name__ == "__main__":
    unittest.main()
