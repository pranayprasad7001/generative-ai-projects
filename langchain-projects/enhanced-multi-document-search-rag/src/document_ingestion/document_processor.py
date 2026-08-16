import ipaddress
import logging
import socket
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from .chunker import Chunker, ChunkStrategy
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


class IngestionResult(BaseModel):
    """Structured report returned by document ingestion pipeline detailing processed and failed sources."""
    documents: List[Document] = Field(default_factory=list, description="All successfully extracted and chunked documents.")
    processed_count: int = Field(default=0, description="Total number of successfully processed files/sources.")
    failed_count: int = Field(default=0, description="Total number of failed files/sources.")
    processed_files: List[str] = Field(default_factory=list, description="Names or paths of successfully processed files.")
    failed_files: List[str] = Field(default_factory=list, description="Names or paths of files that failed ingestion.")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="List of error details with file and error message.")


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
    """
    Handles multi-format document loading and chunking.
    
    Note on document strategies:
    - Unstructured documents (PDF, TXT, DOCX, Markdown, XLSX, Web pages) use configurable text chunking
      (recursive character, semantic embedding-based, or hybrid splitters).
    - Tabular CSV documents use discrete row-level retrieval, where each row is treated as an independent
      structured retrieval unit with preserved column headers, rather than undergoing arbitrary text splitting.
    """

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
        Load document(s) from a CSV file using row-level retrieval.
        
        Unlike unstructured documents that undergo character or semantic splitting,
        each CSV row represents a structured, self-contained record. Each row is indexed
        as a discrete retrieval unit with row index metadata.

        Args:
            file_path (Union[str, Path]): Path to the CSV file
            strategy (ChunkStrategy): Chunking strategy identifier (preserved for schema compatibility)
        Returns:
            List[Document]: List of row-level documents
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
            chunk_strategy="row_level" if strat_val == "recursive" else strat_val
        )

        logger.info("Loaded %d structured rows from %s (row-level retrieval unit)", len(docs), file_path)
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
        detailed = self.load_documents_detailed([str(directory)], strategy=strategy)
        return detailed.documents

    def load_documents_detailed(self, sources: List[str], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> IngestionResult:
        """
        Load documents from supported sources with comprehensive structured reporting:
        tracks successfully processed files, individual failure counts, failed filenames, and exact errors.

        Args:
            sources (List[str]): List of sources (URLs, file paths, directories)
            strategy (ChunkStrategy): Chunking strategy to use

        Returns:
            IngestionResult: Structured report with documents, processed/failed counts, and errors
        """
        result = IngestionResult()
        logger.info("Starting detailed loading process for %d source(s) with strategy '%s'", len(sources), strategy)

        for src in sources:
            logger.info("Processing source: %s", src)

            if src.startswith(("http://", "https://")):
                try:
                    loaded = self.load_from_url(src, strategy=strategy)
                    result.documents.extend(loaded)
                    result.processed_count += 1
                    result.processed_files.append(src)
                except Exception as e:
                    logger.error("Failed to load URL %s: %s", src, e)
                    result.failed_count += 1
                    result.failed_files.append(src)
                    result.errors.append({"file": src, "error": str(e)})
                continue

            path = Path(src)

            if not path.exists():
                logger.error("Source path does not exist: %s", path)
                result.failed_count += 1
                result.failed_files.append(src)
                result.errors.append({"file": src, "error": f"File or directory not found: {path}"})
                continue

            if path.is_file():
                loader = self.supported_loaders.get(path.suffix.lower())
                if loader is None:
                    err_msg = f"Unsupported file type: {path.suffix or '<no extension>'}"
                    logger.error("%s for file: %s", err_msg, path)
                    result.failed_count += 1
                    result.failed_files.append(src)
                    result.errors.append({"file": src, "error": err_msg})
                    continue

                try:
                    loaded = loader(path, strategy=strategy)
                    result.documents.extend(loaded)
                    result.processed_count += 1
                    result.processed_files.append(str(path))
                except Exception as e:
                    logger.error("Failed to load file %s: %s", path, e)
                    result.failed_count += 1
                    result.failed_files.append(str(path))
                    result.errors.append({"file": str(path), "error": str(e)})
                continue

            if path.is_dir():
                for file in sorted(path.rglob("*")):
                    if not file.is_file() or file.name.startswith("~$") or not file.suffix or file.name.startswith("."):
                        continue

                    loader = self.supported_loaders.get(file.suffix.lower())
                    if loader is None:
                        continue

                    try:
                        loaded = loader(file, strategy=strategy)
                        result.documents.extend(loaded)
                        result.processed_count += 1
                        result.processed_files.append(str(file))
                    except Exception as e:
                        logger.error("Failed to load file %s: %s", file, e)
                        result.failed_count += 1
                        result.failed_files.append(str(file))
                        result.errors.append({"file": str(file), "error": str(e)})
                continue

            err_msg = f"Unsupported source format: {src}"
            logger.error(err_msg)
            result.failed_count += 1
            result.failed_files.append(src)
            result.errors.append({"file": src, "error": err_msg})

        logger.info(
            "Detailed ingestion complete: %d chunks, %d processed files, %d failed files.",
            len(result.documents),
            result.processed_count,
            result.failed_count
        )
        return result

    def load_documents(self, sources: List[str], strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Load documents from supported sources including URLs, individual files, and directories.
        Preserves backward compatibility returning List[Document].
        
        Args:
            sources (List[str]): List of sources to load documents from
            strategy (str): Chunking strategy to use
        Returns:
            List[Document]: List of documents
        """
        detailed = self.load_documents_detailed(sources, strategy=strategy)
        if not detailed.documents and detailed.failed_count > 0:
            first_err = detailed.errors[0]["error"] if detailed.errors else "Unknown error loading documents."
            raise ValueError(f"Document loading failed: {first_err}")
        return detailed.documents