import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from langchain_classic.schema import Document
from vectorstore.vectorstore import VectorStoreManager, compute_chunk_identity

class TestVectorStoreManager(unittest.TestCase):
    
    @patch("vectorstore.vectorstore.CohereRerank")
    @patch("vectorstore.vectorstore.AstraDBVectorStore")
    @patch("vectorstore.vectorstore.Config.get_embeddings")
    def setUp(self, mock_get_embeddings, mock_astradb, mock_cohere):
        self.mock_get_embeddings = mock_get_embeddings
        self.mock_astradb = mock_astradb
        self.mock_cohere = mock_cohere
        
        # Instantiate VectorStoreManager
        self.manager = VectorStoreManager()

    def test_init(self):
        self.assertIsNotNone(self.manager.embeddings)
        self.assertIsNotNone(self.manager.vectorstore)
        self.assertIsNotNone(self.manager.cohere_reranker)

    def test_add_documents_to_vectorstore_new_docs(self):
        mock_collection = MagicMock()
        # Simulate that batch find returns empty list (no documents exist yet)
        mock_collection.find.return_value = []
        self.manager.vectorstore.astra_env.collection = mock_collection

        docs = [Document(page_content="test page", metadata={"source": "test.txt"})]
        self.manager._add_documents_to_vectorstore(docs, self.manager.vectorstore)

        mock_collection.find.assert_called_once()
        self.manager.vectorstore.add_documents.assert_called_once_with(docs, ids=[unittest.mock.ANY])

    def test_add_documents_to_vectorstore_duplicates(self):
        mock_collection = MagicMock()
        doc = Document(page_content="test page", metadata={"source": "test.txt"})
        doc_id = compute_chunk_identity(doc)
        mock_collection.find.return_value = [{"_id": doc_id}]
        self.manager.vectorstore.astra_env.collection = mock_collection

        self.manager._add_documents_to_vectorstore([doc], self.manager.vectorstore)

        mock_collection.find.assert_called_once()
        self.manager.vectorstore.add_documents.assert_not_called()

    def test_compute_chunk_identity_dimensions(self):
        doc_base = Document(page_content="Hello world", metadata={"source": "doc.pdf", "doc_version": "v1", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50, "chunk": 0})
        base_id = compute_chunk_identity(doc_base)

        # 1. Same metadata produces identical hash
        doc_same = Document(page_content="Hello world", metadata={"source": "doc.pdf", "doc_version": "v1", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50, "chunk": 0})
        self.assertEqual(base_id, compute_chunk_identity(doc_same))

        # 2. Different document version produces different hash
        doc_v2 = Document(page_content="Hello world", metadata={"source": "doc.pdf", "doc_version": "v2", "chunk_strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50, "chunk": 0})
        self.assertNotEqual(base_id, compute_chunk_identity(doc_v2))

        # 3. Different chunking strategy produces different hash
        doc_semantic = Document(page_content="Hello world", metadata={"source": "doc.pdf", "doc_version": "v1", "chunk_strategy": "semantic", "chunk_size": 500, "chunk_overlap": 50, "chunk": 0})
        self.assertNotEqual(base_id, compute_chunk_identity(doc_semantic))

        # 4. Different chunk size produces different hash
        doc_larger_chunk = Document(page_content="Hello world", metadata={"source": "doc.pdf", "doc_version": "v1", "chunk_strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 50, "chunk": 0})
        self.assertNotEqual(base_id, compute_chunk_identity(doc_larger_chunk))

        # 5. Different embedding model produces different hash
        id_emb1 = compute_chunk_identity(doc_base, default_embedding_model="gemini-embedding-2")
        id_emb2 = compute_chunk_identity(doc_base, default_embedding_model="text-embedding-3-small")
        self.assertNotEqual(id_emb1, id_emb2)

    @patch("vectorstore.vectorstore.BM25Retriever")
    @patch("vectorstore.vectorstore.EnsembleRetriever")
    @patch("vectorstore.vectorstore.ContextualCompressionRetriever")
    def test_create_retriever(self, mock_compression_retriever, mock_ensemble, mock_bm25):
        # Mock vector store's as_retriever
        mock_retriever = MagicMock()
        self.manager.vectorstore.as_retriever.return_value = mock_retriever
        self.manager.documents = [Document(page_content="sample text")]

        # Run create_retriever with k=5 (oversampled candidate_k = 10)
        ret = self.manager.create_retriever(self.manager.vectorstore, k=5, search_type="mmr")

        # Verify search arguments passed to vectorstore's retriever (candidate_k=10)
        self.manager.vectorstore.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 10})
        self.assertEqual(self.manager.cohere_reranker.top_n, 5)
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
        
        # Mock compression retriever invoke return
        mock_comp_inst = MagicMock()
        mock_comp_inst.invoke.return_value = [Document(page_content="result doc")]
        mock_compression_retriever.return_value = mock_comp_inst

        res = self.manager.retrieve("query", k=3, search_type="mmr")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].page_content, "result doc")

    def test_add_documents_returns_count(self):
        with patch.object(self.manager, "create_vectorstore") as mock_create:
            docs = [Document(page_content="doc1"), Document(page_content="doc2")]
            count = self.manager.add_documents(docs)
            self.assertEqual(count, 2)
            mock_create.assert_called_once_with(docs)

    @patch("vectorstore.vectorstore.BM25Retriever")
    @patch("vectorstore.vectorstore.EnsembleRetriever")
    @patch("vectorstore.vectorstore.ContextualCompressionRetriever")
    def test_retrieve_with_session_isolation(self, mock_compression_retriever, mock_ensemble_retriever, mock_bm25):
        self.manager.vectorstore.as_retriever.return_value = MagicMock()
        mock_comp_inst = MagicMock()
        mock_comp_inst.invoke.return_value = [Document(page_content="session doc", metadata={"session_id": "sess_123"})]
        mock_compression_retriever.return_value = mock_comp_inst

        self.manager.documents = [
            Document(page_content="doc A", metadata={"session_id": "sess_123"}),
            Document(page_content="doc B", metadata={"session_id": "other_sess"})
        ]

        res = self.manager.retrieve("query", k=2, session_id="sess_123")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].metadata["session_id"], "sess_123")

if __name__ == "__main__":
    unittest.main()
