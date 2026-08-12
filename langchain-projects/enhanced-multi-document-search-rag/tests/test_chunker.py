import unittest
from unittest.mock import MagicMock
from pathlib import Path
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from document_ingestion.chunker import Chunker, ChunkStrategy
from langchain_classic.schema import Document

class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = Chunker(chunk_size=100, chunk_overlap=10)

    def test_init(self):
        self.assertEqual(self.chunker.chunk_size, 100)
        self.assertEqual(self.chunker.chunk_overlap, 10)
        self.assertIsNotNone(self.chunker.recursive_text_splitter)
        self.assertIsNotNone(self.chunker.markdown_header_splitter)
        self.assertIsNone(self.chunker.semantic_text_splitter)

    def test_add_metadata_str_source(self):
        docs = [Document(page_content="Hello world")]
        updated_docs = self.chunker.add_metadata(
            docs=docs,
            source="http://example.com",
            loader_name="TestLoader",
            add_chunk=True,
            custom_key="custom_val"
        )
        self.assertEqual(len(updated_docs), 1)
        meta = updated_docs[0].metadata
        self.assertEqual(meta["source"], "http://example.com")
        self.assertEqual(meta["loader"], "TestLoader")
        self.assertEqual(meta["chunk"], 0)
        self.assertEqual(meta["custom_key"], "custom_val")

    def test_add_metadata_path_source(self):
        docs = [Document(page_content="Hello world")]
        path_source = Path("data/doc.pdf")
        updated_docs = self.chunker.add_metadata(
            docs=docs,
            source=path_source,
            loader_name="PyMuPDFLoader",
            add_chunk=False
        )
        self.assertEqual(len(updated_docs), 1)
        meta = updated_docs[0].metadata
        self.assertEqual(meta["source"], str(path_source))
        self.assertEqual(meta["file_name"], "doc.pdf")
        self.assertEqual(meta["file_type"], ".pdf")
        self.assertEqual(meta["loader"], "PyMuPDFLoader")
        self.assertNotIn("chunk", meta)

    def test_split_documents_recursive(self):
        # Test basic recursive splitting
        text = "This is a long string of text that will be split by the recursive character splitter."
        docs = [Document(page_content=text)]
        split_docs = self.chunker.split_documents(docs, ".txt", ChunkStrategy.RECURSIVE)
        self.assertTrue(len(split_docs) >= 1)
        for doc in split_docs:
            self.assertTrue(len(doc.page_content) <= 100)

    def test_split_documents_markdown(self):
        md_text = "# Header 1\nThis is header one content.\n## Header 2\nThis is header two content."
        split_docs = self.chunker.split_documents(md_text, ".md", ChunkStrategy.RECURSIVE)
        self.assertTrue(len(split_docs) >= 1)
        # Check that header info was parsed and preserved by MarkdownHeaderTextSplitter
        headers = [doc.metadata for doc in split_docs]
        has_h1 = any("Header 1" in h for h in headers)
        self.assertTrue(has_h1)

    def test_split_documents_semantic_missing_embeddings(self):
        # If no embeddings, using SEMANTIC should raise ValueError
        docs = [Document(page_content="some text")]
        with self.assertRaises(ValueError):
            self.chunker.split_documents(docs, ".txt", ChunkStrategy.SEMANTIC)

    def test_split_documents_semantic_with_mock_embeddings(self):
        mock_embeddings = MagicMock()
        # Mock embed_documents to return generic vector list of lists
        mock_embeddings.embed_documents.return_value = [[0.1] * 768] * 10
        chunker_with_embed = Chunker(embeddings=mock_embeddings, chunk_size=100, chunk_overlap=10)
        docs = [Document(page_content="This is the first sentence. This is the second sentence.")]
        
        # Test semantic split
        split_docs = chunker_with_embed.split_documents(docs, ".txt", ChunkStrategy.SEMANTIC)
        self.assertTrue(len(split_docs) >= 1)

    def test_invalid_strategy(self):
        docs = [Document(page_content="some text")]
        with self.assertRaises(ValueError):
            self.chunker.split_documents(docs, ".txt", "invalid_strategy")

if __name__ == "__main__":
    unittest.main()
