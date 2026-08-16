import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from langchain_core.documents import Document
from vectorstore.vectorstore import (
    VectorStoreManager,
    RerankedVectorRetriever,
    compute_chunk_identity,
    compute_corpus_hash
)


class TestVectorStoreManager(unittest.TestCase):
    
    def setUp(self):
        self.mock_embeddings = MagicMock()
        self.mock_astradb_instance = MagicMock()
        self.mock_cohere_instance = MagicMock()

        self.patcher_emb = patch("vectorstore.vectorstore.Config.get_embeddings", return_value=self.mock_embeddings)
        self.patcher_astra = patch("vectorstore.vectorstore.AstraDBVectorStore", return_value=self.mock_astradb_instance)
        self.patcher_cohere = patch("vectorstore.vectorstore.CohereRerank", return_value=self.mock_cohere_instance)

        self.mock_emb_func = self.patcher_emb.start()
        self.mock_astra_cls = self.patcher_astra.start()
        self.mock_cohere_cls = self.patcher_cohere.start()

        self.addCleanup(self.patcher_emb.stop)
        self.addCleanup(self.patcher_astra.stop)
        self.addCleanup(self.patcher_cohere.stop)

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
        self.mock_astradb_instance.astra_env.collection = mock_collection

        docs = [Document(page_content="test page", metadata={"source": "test.txt"})]
        self.manager._add_documents_to_vectorstore(docs, self.manager.vectorstore)

        mock_collection.find.assert_called_once()
        self.mock_astradb_instance.add_documents.assert_called_once_with(docs, ids=[unittest.mock.ANY])

    def test_add_documents_to_vectorstore_duplicates(self):
        mock_collection = MagicMock()
        doc = Document(page_content="test page", metadata={"source": "test.txt"})
        doc_id = compute_chunk_identity(doc)
        mock_collection.find.return_value = [{"_id": doc_id}]
        self.mock_astradb_instance.astra_env.collection = mock_collection

        self.manager._add_documents_to_vectorstore([doc], self.manager.vectorstore)

        mock_collection.find.assert_called_once()
        self.mock_astradb_instance.add_documents.assert_not_called()

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

    def test_create_retriever(self):
        # Mock vector store's as_retriever
        mock_retriever = MagicMock()
        self.mock_astradb_instance.as_retriever.return_value = mock_retriever

        # Run create_retriever with k=5 (oversampled candidate_k = 10)
        ret = self.manager.create_retriever(self.manager.vectorstore, k=5, search_type="mmr")

        # Verify search arguments passed to vectorstore's retriever (candidate_k=10)
        self.mock_astradb_instance.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 10})
        # Verify dedicated compressor was constructed with top_n=5
        self.mock_cohere_cls.assert_called_with(model=unittest.mock.ANY, top_n=5)
        self.assertIsInstance(ret, RerankedVectorRetriever)
        self.assertEqual(ret.base_retriever, mock_retriever)
        self.assertEqual(self.manager._current_k, 5)
        self.assertEqual(self.manager._current_search_type, "mmr")

    def test_rerank_documents_multiple(self):
        docs = [
            Document(page_content="first content"),
            Document(page_content="second content")
        ]
        self.mock_cohere_instance.compress_documents.return_value = [docs[1], docs[0]]
        
        reranked = self.manager.rerank_documents("query", docs)
        
        self.mock_cohere_instance.compress_documents.assert_called_once_with(documents=docs, query="query")
        self.assertEqual(reranked[0], docs[1])

    def test_rerank_documents_single_or_empty(self):
        docs_empty = []
        reranked_empty = self.manager.rerank_documents("query", docs_empty)
        self.assertEqual(reranked_empty, [])

        docs_single = [Document(page_content="only one")]
        reranked_single = self.manager.rerank_documents("query", docs_single)
        self.assertEqual(reranked_single, docs_single)

    def test_retrieve(self):
        # Mock vectorstore as_retriever
        mock_vec_retriever = MagicMock()
        cand_docs = [Document(page_content="cand 1"), Document(page_content="cand 2")]
        mock_vec_retriever.invoke.return_value = cand_docs
        self.mock_astradb_instance.as_retriever.return_value = mock_vec_retriever

        # Mock cohere compress
        reranked_docs = [Document(page_content="cand 2")]
        self.mock_cohere_instance.compress_documents.return_value = reranked_docs

        res = self.manager.retrieve("query", k=3, search_type="mmr")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].page_content, "cand 2")

    def test_add_documents_returns_count(self):
        with patch.object(self.manager, "create_vectorstore") as mock_create:
            docs = [Document(page_content="doc1"), Document(page_content="doc2")]
            count = self.manager.add_documents(docs)
            self.assertEqual(count, 2)
            mock_create.assert_called_once_with(docs)

    def test_retrieve_with_session_isolation(self):
        mock_vec_retriever = MagicMock()
        cand_docs = [Document(page_content="session doc", metadata={"session_id": "sess_123"})]
        mock_vec_retriever.invoke.return_value = cand_docs
        self.mock_astradb_instance.as_retriever.return_value = mock_vec_retriever
        self.mock_cohere_instance.compress_documents.return_value = cand_docs

        res = self.manager.retrieve("query", k=2, session_id="sess_123")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].metadata["session_id"], "sess_123")

        # Verify session_id filter was passed to vectorstore
        self.mock_astradb_instance.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 10, "filter": {"session_id": "sess_123"}}
        )

    def test_compute_corpus_hash(self):
        doc1 = Document(page_content="doc 1 content", metadata={"source": "d1.txt"})
        doc2 = Document(page_content="doc 2 content", metadata={"source": "d2.txt"})

        hash1 = compute_corpus_hash([doc1, doc2])
        hash2 = compute_corpus_hash([doc2, doc1])  # order independent
        self.assertEqual(hash1, hash2)
        self.assertEqual(compute_corpus_hash([]), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


if __name__ == "__main__":
    unittest.main()

