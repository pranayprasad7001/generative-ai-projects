import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.llmgateway_config import Config, RateLimitedOpenAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter


class TestRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Tests for Rate Limiter sharing and behavior across components."""

    def test_singleton_llm_rate_limiter(self):
        """Verify that get_llm_checker and get_llm_generator share the exact same rate limiter instance."""
        limiter1 = Config.get_llm_rate_limiter()
        limiter2 = Config.get_llm_rate_limiter()
        self.assertIs(limiter1, limiter2)

        checker = Config.get_llm_checker()
        generator = Config.get_llm_generator()
        self.assertIs(checker.rate_limiter, generator.rate_limiter)
        self.assertIs(checker.rate_limiter, limiter1)

    def test_singleton_embedding_rate_limiter(self):
        """Verify that get_embeddings shares the same rate limiter instance."""
        limiter1 = Config.get_embedding_rate_limiter()
        limiter2 = Config.get_embedding_rate_limiter()
        self.assertIs(limiter1, limiter2)

        embeddings1 = Config.get_embeddings()
        embeddings2 = Config.get_embeddings()
        self.assertIs(embeddings1.rate_limiter, embeddings2.rate_limiter)
        self.assertIs(embeddings1.rate_limiter, limiter1)

    def test_rate_limited_embeddings_sync(self):
        """Verify that embed_documents acquires the rate limiter and embed_query delegates without double acquire."""
        mock_limiter = MagicMock(spec=InMemoryRateLimiter)
        embeddings = RateLimitedOpenAIEmbeddings(
            model="gemini-embedding-2",
            api_key="test_key",
            base_url="http://127.0.0.1:4000",
            chunk_size=5,
            rate_limiter=mock_limiter
        )

        mock_client = MagicMock()
        mock_client.create.return_value = {
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        }
        embeddings.client = mock_client

        # Test embed_documents
        res = embeddings.embed_documents(["doc1", "doc2"])
        self.assertEqual(len(res), 2)
        mock_limiter.acquire.assert_called_once()

        # Reset mock call counts
        mock_limiter.reset_mock()
        mock_client.reset_mock()
        mock_client.create.return_value = {"data": [{"embedding": [0.5, 0.6]}]}

        # Test embed_query
        q_res = embeddings.embed_query("test query")
        self.assertEqual(q_res, [0.5, 0.6])
        # embed_query delegates to embed_documents so acquire is called exactly once
        mock_limiter.acquire.assert_called_once()

    async def test_rate_limited_embeddings_async(self):
        """Verify that aembed_documents and aembed_query properly acquire the rate limiter asynchronously."""
        mock_limiter = MagicMock(spec=InMemoryRateLimiter)
        embeddings = RateLimitedOpenAIEmbeddings(
            model="gemini-embedding-2",
            api_key="test_key",
            base_url="http://127.0.0.1:4000",
            chunk_size=5,
            rate_limiter=mock_limiter
        )

        mock_async_client = MagicMock()
        async def mock_async_create(*args, **kwargs):
            return {"data": [{"embedding": [0.1, 0.2]}]}
        mock_async_client.create = mock_async_create
        embeddings.async_client = mock_async_client

        # Test aembed_documents
        res = await embeddings.aembed_documents(["doc1"])
        self.assertEqual(len(res), 1)
        mock_limiter.aacquire.assert_called_once()

        mock_limiter.reset_mock()

        # Test aembed_query
        q_res = await embeddings.aembed_query("test async query")
        self.assertEqual(q_res, [0.1, 0.2])
        mock_limiter.aacquire.assert_called_once()


if __name__ == "__main__":
    unittest.main()
