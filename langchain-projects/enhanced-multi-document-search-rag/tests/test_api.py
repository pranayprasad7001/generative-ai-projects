import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fastapi.testclient import TestClient
from api import app


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI REST API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("api.vector_store_manager")
    def test_health_endpoint(self, mock_vsm):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("nemotron-3-ultra-550b-a55b", data["model_configured"])

    @patch("api.rag_system")
    @patch("api.vector_store_manager")
    def test_query_rag_success(self, mock_vsm, mock_rag):
        mock_retriever = MagicMock()
        mock_vsm.get_retriever.return_value = mock_retriever
        
        mock_rag.run = AsyncMock(return_value={
            "question": "What is AI?",
            "answer": "AI is Artificial Intelligence.",
            "total_cost": 0.0005,
            "retrieved_docs": [],
            "external_citations": []
        })

        payload = {
            "question": "What is AI?",
            "search_type": "similarity",
            "k": 3,
            "thread_id": "test_thread",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        response = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], "What is AI?")
        self.assertEqual(data["answer"], "AI is Artificial Intelligence.")
        self.assertEqual(data["total_cost"], 0.0005)


if __name__ == "__main__":
    unittest.main()
