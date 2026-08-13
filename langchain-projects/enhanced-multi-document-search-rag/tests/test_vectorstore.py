import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from langchain_classic.schema import Document

class TestVectorStoreManager(unittest.TestCase):
    
    @patch("vectorstore.vectorstore.CohereRerank")
    @patch("vectorstore.vectorstore.AstraDBVectorStore")
    @patch("vectorstore.vectorstore.GoogleGenerativeAIEmbeddings")
    def setUp(self, mock_google_embeddings, mock_astradb, mock_cohere):
        self.mock_google_embeddings = mock_google_embeddings
        self.mock_astradb = mock_astradb
        self.mock_cohere = mock_cohere
        
        # Instantiate VectorStoreManager
        from vectorstore.vectorstore import VectorStoreManager
        self.manager = VectorStoreManager()

    def test_init(self):
        self.assertIsNotNone(self.manager.embeddings)
        self.assertIsNotNone(self.manager.vectorstore)
        self.assertIsNotNone(self.manager.cohere_reranker)

    def test_add_documents_to_vectorstore_new_docs(self):
        mock_collection = MagicMock()
        # Simulate that documents don't exist yet in DB
        mock_collection.find_one.return_value = None
        self.manager.vectorstore.astra_env.collection = mock_collection

        docs = [Document(page_content="test page", metadata={"source": "test.txt"})]
        self.manager._add_documents_to_vectorstore(docs, self.manager.vectorstore)

        mock_collection.find_one.assert_called_once()
        self.manager.vectorstore.add_documents.assert_called_once_with(docs, ids=[unittest.mock.ANY])

    def test_add_documents_to_vectorstore_duplicates(self):
        mock_collection = MagicMock()
        # Simulate that documents already exist
        mock_collection.find_one.return_value = {"_id": "exists"}
        self.manager.vectorstore.astra_env.collection = mock_collection

        docs = [Document(page_content="test page", metadata={"source": "test.txt"})]
        self.manager._add_documents_to_vectorstore(docs, self.manager.vectorstore)

        mock_collection.find_one.assert_called_once()
        self.manager.vectorstore.add_documents.assert_not_called()

    @patch("vectorstore.vectorstore.BM25Retriever")
    @patch("vectorstore.vectorstore.EnsembleRetriever")
    @patch("vectorstore.vectorstore.ContextualCompressionRetriever")
    def test_create_retriever(self, mock_compression_retriever, mock_ensemble, mock_bm25):
        # Mock vector store's as_retriever
        mock_retriever = MagicMock()
        self.manager.vectorstore.as_retriever.return_value = mock_retriever

        # Run create_retriever
        ret = self.manager.create_retriever(self.manager.vectorstore, k=5, search_type="mmr")

        # Verify search arguments passed to vectorstore's retriever
        self.manager.vectorstore.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 5})
        self.assertEqual(self.manager._current_k, 5)
        self.assertEqual(self.manager._current_search_type, "mmr")

    def test_rerank_documents_multiple(self):
        docs = [
            Document(page_content="first content"),
            Document(page_content="second content")
        ]
        self.manager.cohere_reranker.compress_documents.return_value = [docs[1], docs[0]]
        
        reranked = self.manager.rerank_documents("query", docs)
        
        self.manager.cohere_reranker.compress_documents.assert_called_once_with(documents=docs, query="query")
        self.assertEqual(reranked[0], docs[1])

    def test_rerank_documents_single_or_empty(self):
        docs_empty = []
        reranked_empty = self.manager.rerank_documents("query", docs_empty)
        self.assertEqual(reranked_empty, [])

        docs_single = [Document(page_content="only one")]
        reranked_single = self.manager.rerank_documents("query", docs_single)
        self.assertEqual(reranked_single, docs_single)

    @patch("vectorstore.vectorstore.BM25Retriever")
    @patch("vectorstore.vectorstore.EnsembleRetriever")
    @patch("vectorstore.vectorstore.ContextualCompressionRetriever")
    def test_retrieve(self, mock_compression_retriever, mock_ensemble_retriever, mock_bm25):
        # Mock vectorstore as_retriever
        self.manager.vectorstore.as_retriever.return_value = MagicMock()
        
        # Mock ensemble_retriever
        mock_ensemble_inst = MagicMock()
        mock_ensemble_inst.invoke.return_value = [Document(page_content="result doc")]
        mock_ensemble_retriever.return_value = mock_ensemble_inst

        # Mock rerank
        self.manager.cohere_reranker.compress_documents.return_value = [Document(page_content="result doc")]

        res = self.manager.retrieve("query", k=3, search_type="mmr")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].page_content, "result doc")

if __name__ == "__main__":
    unittest.main()
