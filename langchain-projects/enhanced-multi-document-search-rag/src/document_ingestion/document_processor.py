"""Document processing module for loading and splitting documents"""

from pathlib import Path
from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.schema import Document
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    TextLoader, 
    PyPDFDirectoryLoader,
    Docx2txtLoader
)

class DocumentProcessor:
    """Handles document loading and chunking"""

    def __init__(self, chunk_size: int=500, chunk_overlap: int=50):
        """
        Initialize DocumentProcessor
        Args:
            chunk_size (int): Size of each text chunk
            chunk_overlap (int): Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def load_from_url(self, url: str) -> List[Document]:
        """
        Load documents from a URL
        Args:
            url (str): URL to load documents from
        Returns:
            List[Document]: List of documents
        """
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a PDF file
        Args:
            file_path (Union[str, Path]): Path to the PDF file
        Returns:
            List[Document]: List of documents
        """
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_text(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load document(s) from a text file
        Args:
            file_path (Union[str, Path]): Path to the text file
        Returns:
            List[Document]: List of documents
        """
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """
        Load documents from a directory
        Args:
            directory (Union[str, Path]): Path to the directory
        Returns:
            List[Document]: List of documents
        """
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()    

    def load_from_docx(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a DOCX file
        Args:
            file_path (Union[str, Path]): Path to the DOCX file
        Returns:
            List[Document]: List of documents
        """
        loader = Docx2txtLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files
        Args:
            sources (List[str]): List of sources to load documents from
        Returns:
            List[Document]: List of documents
        """
        
        docs: List[Document] = []

        for src in sources:
            
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_text(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            elif path.suffix.lower() == ".docx":
                docs.extend(self.load_from_docx(path))
            elif not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            else:
                raise ValueError(
                    f"Unsupported source type: {src}."
                    "Use URLs, PDFs, PDF directories, DOCX files, or TXT files."
                )
        
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        Args:
            documents (List[Document]): List of documents to split
        Returns:
            List[Document]: List of split documents
        """
        return self.text_splitter.split_documents(documents)

    def load_and_split(self, sources: List[str]) -> List[Document]:
        """
        Load documents from sources and split them into chunks
        Args:
            sources (List[str]): List of sources to load documents from
        Returns:
            List[Document]: List of split documents
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)