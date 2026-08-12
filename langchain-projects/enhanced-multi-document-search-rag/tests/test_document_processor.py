import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from document_ingestion.document_processor import DocumentProcessor
from document_ingestion.chunker import ChunkStrategy
from langchain_classic.schema import Document

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)

    @patch("document_ingestion.document_processor.WebBaseLoader")
    def test_load_from_url(self, mock_web_loader):
        # Mock WebBaseLoader instance and its load() method
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="Web content here")]
        mock_web_loader.return_value = mock_instance

        url = "http://example.com"
        docs = self.processor.load_from_url(url, strategy=ChunkStrategy.RECURSIVE)
        
        mock_web_loader.assert_called_once_with(url)
        mock_instance.load.assert_called_once()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0].metadata["source"], url)
        self.assertEqual(docs[0].metadata["loader"], "WebBaseLoader")

    @patch("document_ingestion.document_processor.PyMuPDFLoader")
    def test_load_from_pdf(self, mock_pdf_loader):
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="PDF content here")]
        mock_pdf_loader.return_value = mock_instance

        pdf_path = "dummy.pdf"
        docs = self.processor.load_from_pdf(pdf_path, strategy=ChunkStrategy.RECURSIVE)

        mock_pdf_loader.assert_called_once_with(pdf_path)
        mock_instance.load.assert_called_once()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0].metadata["source"], pdf_path)
        self.assertEqual(docs[0].metadata["loader"], "PyMuPDFLoader")

    @patch("document_ingestion.document_processor.TextLoader")
    def test_load_from_text(self, mock_text_loader):
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="Text file content here")]
        mock_text_loader.return_value = mock_instance

        txt_path = "dummy.txt"
        docs = self.processor.load_from_text(txt_path, strategy=ChunkStrategy.RECURSIVE)

        mock_text_loader.assert_called_once_with(txt_path, encoding="utf-8")
        mock_instance.load.assert_called_once()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0].metadata["source"], txt_path)
        self.assertEqual(docs[0].metadata["loader"], "TextLoader")

    @patch("document_ingestion.document_processor.Docx2txtLoader")
    def test_load_from_docx(self, mock_docx_loader):
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="DOCX content here")]
        mock_docx_loader.return_value = mock_instance

        docx_path = "dummy.docx"
        docs = self.processor.load_from_docx(docx_path, strategy=ChunkStrategy.RECURSIVE)

        mock_docx_loader.assert_called_once_with(docx_path)
        mock_instance.load.assert_called_once()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0].metadata["source"], docx_path)
        self.assertEqual(docs[0].metadata["loader"], "Docx2txtLoader")

    @patch("document_ingestion.document_processor.UnstructuredMarkdownLoader")
    def test_load_from_markdown(self, mock_md_loader):
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="Markdown content here", metadata={"key": "val"})]
        mock_md_loader.return_value = mock_instance

        md_path = "dummy.md"
        docs = self.processor.load_from_markdown(md_path, strategy=ChunkStrategy.RECURSIVE)

        mock_md_loader.assert_called_once_with(md_path, mode="single")
        mock_instance.load.assert_called_once()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0].metadata["source"], md_path)
        self.assertEqual(docs[0].metadata["loader"], "UnstructuredMarkdownLoader")
        self.assertEqual(docs[0].metadata["key"], "val")

    @patch("document_ingestion.document_processor.CSVLoader")
    def test_load_from_csv(self, mock_csv_loader):
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="col1,col2\nval1,val2", metadata={})]
        mock_csv_loader.return_value = mock_instance

        csv_path = "dummy.csv"
        docs = self.processor.load_from_csv(csv_path, strategy=ChunkStrategy.RECURSIVE)

        mock_csv_loader.assert_called_once_with(csv_path)
        mock_instance.load.assert_called_once()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0].metadata["source"], csv_path)
        self.assertEqual(docs[0].metadata["loader"], "CSVLoader")
        self.assertEqual(docs[0].metadata["row"], 0)

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create a mock text file
            txt_file = tmpdir_path / "test.txt"
            txt_file.write_text("Hello text content", encoding="utf-8")
            
            # Create a mock unsupported file
            unsupported_file = tmpdir_path / "test.xyz"
            unsupported_file.write_text("unsupported content")

            with patch.object(self.processor, "load_from_text") as mock_load_text:
                mock_load_text.return_value = [Document(page_content="Hello text content", metadata={"source": str(txt_file)})]
                self.processor.supported_loaders[".txt"] = mock_load_text
                
                docs = self.processor.load_from_directory(tmpdir, strategy=ChunkStrategy.RECURSIVE)
                
                mock_load_text.assert_called_once_with(txt_file, strategy=ChunkStrategy.RECURSIVE)
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0].page_content, "Hello text content")

    def test_load_documents_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load_documents(["non_existent_file.txt"])

if __name__ == "__main__":
    unittest.main()
