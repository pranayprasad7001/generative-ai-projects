"""Document Chunking and Text Splitting Module.

This module provides multi-strategy document splitting:
- Recursive character splitting with configurable chunk size and overlap
- Markdown header-aware splitting preserving document hierarchy
- Experimental semantic chunking based on embedding distance breaks
- Hybrid splitting combining structural headers, semantic chunking, and character windows
- Automatic stamping of deterministic composite chunk identity based on document and chunk metadata
"""

from enum import Enum
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_classic.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Supported document chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class Chunker:
    """Handle text chunking strategies."""

    def __init__(self, embeddings=None, chunk_size=500, chunk_overlap=50):
        """
            Initialize Chunker
        
        Args:
            chunk_size (int): Size of each text chunk
            chunk_overlap (int): Overlap between consecutive chunks
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.debug(
            "Initializing Chunker with chunk_size=%d, chunk_overlap=%d",
            self.chunk_size,
            self.chunk_overlap
        )

        self.recursive_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False
        )

        self.semantic_text_splitter = (
            SemanticChunker(embeddings)
            if embeddings
            else None
        )

    def add_metadata(self, docs: List[Document], source: Union[str, Path], loader_name: str, existing_metadata: dict | None = None, add_chunk: bool = False, **extra_metadata) -> List[Document]:
        """
            Add standardized metadata while preserving existing loader metadata.

        Args:
            docs (List[Document]): List of documents to add metadata to
            source (Union[str, Path]): Source of the documents
            loader_name (str): Name of the loader used
            existing_metadata (dict | None): Existing metadata to preserve
            add_chunk (bool): Whether to add chunk number to metadata
            **extra_metadata: Additional metadata to add

        Returns:
            List[Document]: List of documents with metadata
        """

        if isinstance(source, Path):
            base_metadata = {
                "source": str(source),
                "file_name": source.name,
                "file_type": source.suffix.lower(),
                "loader": loader_name,
            }
        else:
            base_metadata = {
                "source": source,
                "loader": loader_name,
            }

        existing_metadata = existing_metadata or {}
        new_docs = []

        for index, doc in enumerate(docs):
            doc_meta = getattr(doc, "metadata", {}) or {}
            meta = {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunk_strategy": "recursive",
                "doc_version": "v1",
                **base_metadata,
                **existing_metadata,
                **doc_meta,
                **extra_metadata,
            }

            if add_chunk:
                meta["chunk"] = index
                meta["chunk_index"] = index

            new_docs.append(Document(page_content=doc.page_content, metadata=meta))

        logger.info(
            "Added metadata for %d document(s) from source '%s' (loader=%s)",
            len(new_docs),
            source,
            loader_name
        )
        return new_docs

    def split_documents(self, docs: Union[List[Document], str], file_type: str, strategy: ChunkStrategy = ChunkStrategy.RECURSIVE) -> List[Document]:
        """
        Split documents based on loader type and strategy
        
        Args:
            docs (Union[List[Document], str]): List of documents or text string to split
            file_type (str): Name of the loader used
            strategy (str): Chunking strategy to use
        
        Returns:
            List[Document]: List of split documents
        """
        logger.info(
            "Splitting document(s) of type '%s' (input_type=%s, strategy='%s')",
            file_type,
            type(docs).__name__,
            strategy
        )

        # Normalize strategy to ChunkStrategy enum member
        if isinstance(strategy, str):
            try:
                strategy = ChunkStrategy(strategy)
            except ValueError:
                raise ValueError(f"Invalid strategy string: '{strategy}'. Valid options: {[s.value for s in ChunkStrategy]}")
        elif not isinstance(strategy, ChunkStrategy):
            raise ValueError(f"strategy must be a ChunkStrategy or a valid string. Got: {type(strategy).__name__}")

        # Validate embeddings requirement
        semantic_splitter = self.semantic_text_splitter
        if strategy in (ChunkStrategy.SEMANTIC, ChunkStrategy.HYBRID) and semantic_splitter is None:
            raise ValueError(f"Embeddings must be provided to Chunker to use '{strategy.value}' chunking strategy.")

        if file_type == ".md":
            if isinstance(docs, list):
                text = "\n\n".join([doc.page_content for doc in docs])
            elif isinstance(docs, Document):
                text = docs.page_content
            else:
                text = str(docs)
            
            logger.debug("Markdown header splitting started...")
            header_splits = self.markdown_header_splitter.split_text(text)
            logger.debug("Markdown splitting of %d header sections using strategy '%s' started...", len(header_splits), strategy)
            
            if strategy is ChunkStrategy.SEMANTIC:
                if semantic_splitter is None:
                    raise ValueError("Embeddings must be provided for semantic splitting.")
                split_docs = semantic_splitter.split_documents(header_splits)
            elif strategy is ChunkStrategy.HYBRID:
                if semantic_splitter is None:
                    raise ValueError("Embeddings must be provided for hybrid splitting.")
                split_docs = semantic_splitter.split_documents(header_splits)
                split_docs = self.recursive_text_splitter.split_documents(split_docs)
            else: # recursive or fallback
                split_docs = self.recursive_text_splitter.split_documents(header_splits)
        else:
            if isinstance(docs, str):
                docs = [Document(page_content=docs)]
            
            if strategy is ChunkStrategy.SEMANTIC:
                if semantic_splitter is None:
                    raise ValueError("Embeddings must be provided for semantic splitting.")
                logger.debug("Semantic splitting started...")
                split_docs = semantic_splitter.split_documents(docs)
            elif strategy is ChunkStrategy.HYBRID:
                if semantic_splitter is None:
                    raise ValueError("Embeddings must be provided for hybrid splitting.")
                logger.debug("Hybrid splitting started...")
                split_docs = semantic_splitter.split_documents(docs)
                split_docs = self.recursive_text_splitter.split_documents(split_docs)
            else: # recursive
                logger.debug("Recursive splitting started...")
                split_docs = self.recursive_text_splitter.split_documents(docs)
        
        # Stamp chunk_strategy and chunk_index on split chunks
        for i, doc in enumerate(split_docs):
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                doc.metadata.setdefault("chunk_strategy", strategy.value)
                doc.metadata.setdefault("chunk_size", self.chunk_size)
                doc.metadata.setdefault("chunk_overlap", self.chunk_overlap)
                doc.metadata.setdefault("chunk", i)
                doc.metadata.setdefault("chunk_index", i)

        logger.info("Generated %d chunks from document splitting.", len(split_docs))
        return split_docs

