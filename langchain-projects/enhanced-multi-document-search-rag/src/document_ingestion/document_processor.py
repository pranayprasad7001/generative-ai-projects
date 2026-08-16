import ipaddress
import logging
import socket
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Union
from .chunker import Chunker, ChunkStrategy
from collections import defaultdict
from langchain_classic.schema import Document
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyMuPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)

logger = logging.getLogger(__name__)


def validate_safe_url(url: str, allow_local: bool = False) -> str:
    """Validate URL scheme and ensure it does not resolve to private, loopback, or cloud-metadata IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: '{parsed.scheme}'. Only http and https are allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname.")

    if not allow_local:
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    raise ValueError(f"URL target resolves to disallowed private/internal IP: {ip_str}")
        except socket.gaierror as e:
            raise ValueError(f"Could not resolve hostname '{hostname}': {e}")
    return url


class DocumentProcessor:
    """Handles document loading and chunking"""

    def __init__(self, embeddings=None, chunk_size: int=500, chunk_overlap: int=50, allow_local_urls: bool = False):
        """
        Initialize DocumentProcessor
        Args:
            chunk_size (int): Size of each text chunk
            chunk_overlap (int): Overlap between consecutive chunks
            allow_local_urls (bool): Whether to allow private/local URLs (for testing)
        """

        self.chunker = Chunker(
            embeddings=embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.allow_local_urls = allow_local_urls

        self.supported_loaders = {
            ".pdf": self.load_from_pdf,
            ".txt": self.load_from_text,
            ".docx": self.load_from_docx,
            ".md": self.load_from_markdown,
            ".csv": self.load_from_csv,
            ".xlsx": self.load_from_excel,
            ".xls": self.load_from_excel,
        }

    def process_urls(self, urls: List[str], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Process a list of URLs by loading and chunking them.
        Args:
            urls (List[str]): List of URLs to process
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of processed documents
        """
        return self.load_documents(urls, strategy=strategy)

    def load_from_url(self, url: str, strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load documents from a URL with DOM cleaning to remove script, style, nav, and footers.
        Args:
            url (str): URL to load documents from
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of cleaned documents
        """
        validate_safe_url(url, allow_local=self.allow_local_urls)
        docs = WebBaseLoader(url).load()
        cleaned_docs = []
        for doc in docs:
            text = doc.page_content
            try:
                soup = BeautifulSoup(text, "html.parser")
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    element.decompose()
                clean_text = soup.get_text(separator="\n\n")
            except Exception as e:
                logger.debug("HTML cleaning error for %s: %s", url, e)
                clean_text = text

            lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
            normalized_text = "\n".join(lines)
            if normalized_text:
                cleaned_docs.append(Document(page_content=normalized_text, metadata=doc.metadata))

        target_docs = cleaned_docs if cleaned_docs else docs
        split_docs = self.chunker.split_documents(target_docs, "url", strategy=strategy)
        docs_with_metadata = self.chunker.add_metadata(docs=split_docs, source=url, loader_name="WebBaseLoader", add_chunk=True)
        logger.info("Loaded %d clean chunks from %s", len(docs_with_metadata), url)
        return docs_with_metadata

    def load_from_pdf(self, file_path: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load documents from a PDF file
        Args:
            file_path (Union[str, Path]): Path to the PDF file
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        docs = PyMuPDFLoader(str(file_path)).load()
        split_docs = self.chunker.split_documents(docs, ".pdf", strategy=strategy)
        docs_with_metadata = self.chunker.add_metadata(docs=split_docs, source=file_path, loader_name="PyMuPDFLoader", add_chunk=True)
        logger.info("Loaded %d chunks from %s", len(docs_with_metadata), file_path)
        return docs_with_metadata

    def load_from_text(self, file_path: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load document(s) from a text file
        Args:
            file_path (Union[str, Path]): Path to the text file
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        docs = TextLoader(str(file_path), encoding="utf-8").load()
        split_docs = self.chunker.split_documents(docs, ".txt", strategy=strategy)
        docs_with_metadata = self.chunker.add_metadata(docs=split_docs, source=file_path, loader_name="TextLoader", add_chunk=True)
        logger.info("Loaded %d chunks from %s", len(docs_with_metadata), file_path)
        return docs_with_metadata

    def load_from_docx(self, file_path: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load documents from a DOCX file
        Args:
            file_path (Union[str, Path]): Path to the DOCX file
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        docs = Docx2txtLoader(str(file_path)).load()
        split_docs = self.chunker.split_documents(docs, ".docx", strategy=strategy)
        docs_with_metadata = self.chunker.add_metadata(docs=split_docs, source=file_path, loader_name="Docx2txtLoader", add_chunk=True)
        logger.info("Loaded %d chunks from %s", len(docs_with_metadata), file_path)
        return docs_with_metadata

    def load_from_markdown(self, file_path: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load document(s) from a Markdown file
        Args:
            file_path (Union[str, Path]): Path to the Markdown file
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        docs = UnstructuredMarkdownLoader(str(file_path), mode="single").load()
        if not docs:
            logger.warning("No content found in markdown file: %s", file_path)
            return []
        docs_content = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
        base_metadata = docs[0].metadata.copy() if docs else {}
        split_docs = self.chunker.split_documents(docs_content, ".md", strategy=strategy)
        docs_with_metadata = self.chunker.add_metadata(docs=split_docs, source=file_path, loader_name="UnstructuredMarkdownLoader", add_chunk=True, existing_metadata=base_metadata)
        logger.info("Loaded %d markdown chunks from %s", len(docs_with_metadata), file_path)
        return docs_with_metadata

    def load_from_csv(self, file_path: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load document(s) from a CSV file
        Args:
            file_path (Union[str, Path]): Path to the CSV file
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        docs = CSVLoader(str(file_path)).load()
        strat_val = strategy.value if hasattr(strategy, "value") else str(strategy)
        for row, doc in enumerate(docs):
            doc.metadata["row"] = row

        docs_with_metadata = self.chunker.add_metadata(
            docs=docs,
            source=file_path,
            loader_name="CSVLoader",
            add_chunk=True,
            chunk_strategy=strat_val
        )

        logger.info("Loaded %d rows from %s", len(docs), file_path)
        return docs_with_metadata

    def load_from_excel(self, file_path: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load document(s) from an Excel file
        Args:
            file_path (Union[str, Path]): Path to the Excel file
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        docs = UnstructuredExcelLoader(str(file_path), mode="single").load()
        if not docs:
            logger.warning("No content found in Excel file: %s", file_path)
            return []
        split_docs = self.chunker.split_documents(docs, ".xlsx", strategy=strategy)
        docs_with_metadata = self.chunker.add_metadata(docs=split_docs, source=file_path, loader_name="UnstructuredExcelLoader", add_chunk=True)
        logger.info("Loaded %d chunks from %s", len(docs_with_metadata), file_path)
        return docs_with_metadata

    def load_from_directory(self, directory: Union[str, Path], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Recursively load supported documents from a directory.

        Args:
            directory (Union[str, Path]): Directory path.
            strategy (str): Chunking strategy to use

        Returns:
            List[Document]: Loaded documents.
        """

        directory = Path(directory)
        logger.info("Scanning directory for supported files: %s", directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"{directory} is not a directory")

        docs: List[Document] = []

        grouped_files = defaultdict(list)

        for file in sorted(directory.rglob("*")):

            if not file.is_file():
                continue

            if file.name.startswith("~$"):
                logger.debug("Skipping temporary file: %s", file)
                continue
            
            if not file.suffix:
                logger.debug("Skipping file with no extension: %s", file)
                continue

            if file.name.startswith("."):
                logger.debug("Skipping hidden file: %s", file)
                continue
                
            grouped_files[file.suffix.lower()].append(file)

        total_files = sum(len(f) for f in grouped_files.values())
        logger.info("Found %d file(s) across %d different extensions", total_files, len(grouped_files))

        for extension, files in grouped_files.items():

            loader = self.supported_loaders.get(extension)

            if loader is None:
                logger.warning("Skipping unsupported files with extension: %s", extension)
                continue

            logger.info(
                "Loading %d %s file(s)...",
                len(files),
                extension,
            )

            for file in files:
                try:
                    logger.debug("Loading file: %s", file)
                    docs.extend(loader(file, strategy=strategy))
                except Exception as e:
                    logger.error("Failed to load file %s: %s", file, e)

        logger.info("Directory loading complete. Total chunks loaded: %d", len(docs))
        return docs

    def load_documents(self, sources: List[str], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load documents from supported sources including
        URLs, individual files, and directories.
        
        Args:
            sources (List[str]): List of sources to load documents from
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        logger.info("Starting loading process for %d source(s) with strategy '%s'", len(sources), strategy)
        docs: List[Document] = []

        for src in sources:
            logger.info("Processing source: %s", src)

            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src, strategy=strategy))
                continue

            path = Path(src)

            if not path.exists():
                logger.error("Source path does not exist: %s", path)
                raise FileNotFoundError(f"File not found: {path}")

            if path.is_file():
                loader = self.supported_loaders.get(path.suffix.lower())

                if loader is None:
                    logger.error("Unsupported file type: %s", path.suffix)
                    raise ValueError(
                        f"Unsupported file type: {path.suffix or '<no extension>'}"
                    )

                logger.info("Loading single file: %s", path)
                docs.extend(loader(path, strategy=strategy))
                continue

            if path.is_dir():
                docs.extend(self.load_from_directory(path, strategy=strategy))
                continue

            logger.error("Unsupported source format: %s", src)
            raise ValueError(f"Unsupported source: {src}")
        
        logger.info("Document loading pipeline finished. Total chunks created: %d", len(docs))
        return docs