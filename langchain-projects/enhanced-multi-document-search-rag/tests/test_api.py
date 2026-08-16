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

    @patch("api.rag_system")
    @patch("api.vector_store_manager")
    def test_query_rag_error_sanitized(self, mock_vsm, mock_rag):
        mock_vsm.get_retriever.return_value = MagicMock()
        mock_rag.run = AsyncMock(side_effect=Exception("Internal database password secret leaked!"))

        payload = {"question": "What is AI?"}
        response = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertNotIn("database password secret", data["detail"])
        self.assertIn("An error occurred while processing the query", data["detail"])

    @patch("api.API_SECRET_KEY", "super-secret-test-key")
    def test_api_key_auth_enforcement(self):
        # 1. Missing API key should return 401
        res = self.client.post("/api/v1/query", json={"question": "test"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid or missing API key", res.json()["detail"])

        # 2. Invalid API key should return 401
        res = self.client.post("/api/v1/query", json={"question": "test"}, headers={"X-API-Key": "wrong-key"})
        self.assertEqual(res.status_code, 401)

    @patch("api.doc_processor")
    @patch("api.vector_store_manager")
    def test_ingest_url_ssrf_blocked(self, mock_vsm, mock_dp):
        # Loopback URL
        payload = {"urls": ["http://127.0.0.1:8000/internal"]}
        response = self.client.post("/api/v1/ingest/url", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("disallowed private/internal IP", response.json()["detail"])

    @patch("api.doc_processor")
    @patch("api.vector_store_manager")
    @patch("api.MAX_FILE_SIZE_BYTES", 100)
    def test_ingest_file_size_limit(self, mock_vsm, mock_dp):
        mock_dp.supported_loaders = {".txt": MagicMock()}
        file_content = b"A" * 500  # Exceeds mock 100 byte limit
        response = self.client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.txt", file_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("File exceeds maximum allowed size limit", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
