"""Vector Store and Retrieval Management Module using AstraDB and Cohere Rerank.

This module coordinates:
- AstraDB cloud vector embeddings, document storage, and metadata filtering via LiteLLM Gateway
- Deterministic composite chunk identity hashing for zero-duplicate ingestion
- Dynamic candidate oversampling (similarity / MMR)
- Cross-encoder semantic document reranking via Cohere Rerank
"""

import logging
import hashlib
import threading
from typing import Any, List, Optional

from pydantic import ConfigDict
from langchain_cohere import CohereRerank
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from config.llmgateway_config import Config

logger = logging.getLogger(__name__)


def compute_chunk_identity(doc: Document, default_embedding_model: str = Config.EMBEDDING_MODEL) -> str:
    """
    Compute a deterministic composite chunk identity based on document and chunk metadata:
    1. Source document path or URL
    2. Document version (default 'v1')
    3. Chunking strategy (recursive, semantic, hybrid, etc.)
    4. Chunk size and overlap
    5. Embedding model name
    6. Chunk index
    7. Exact chunk text content
    """
    meta = getattr(doc, "metadata", {}) or {}
    source = meta.get("source", "")
    version = meta.get("doc_version") or meta.get("version", "v1")
    chunk_strategy = meta.get("chunk_strategy", "default")
    chunk_size = meta.get("chunk_size", "")
    chunk_overlap = meta.get("chunk_overlap", "")
    chunk_index = meta.get("chunk_index", meta.get("chunk", ""))
    embedding_model = meta.get("embedding_model") or default_embedding_model
    content = doc.page_content or ""

    identity_string = (
        f"src:{source}|ver:{version}|strat:{chunk_strategy}|sz:{chunk_size}|"
        f"ov:{chunk_overlap}|emb:{embedding_model}|chk:{chunk_index}|content:{content}"
    )
    return hashlib.sha256(identity_string.encode("utf-8")).hexdigest()


def compute_corpus_hash(docs: List[Document]) -> str:
    """Compute a deterministic hash for a collection of documents based on chunk identities."""
    if not docs:
        return hashlib.sha256(b"").hexdigest()
    chunk_ids = sorted([compute_chunk_identity(d) for d in docs])
    return hashlib.sha256("|".join(chunk_ids).encode("utf-8")).hexdigest()


class RerankedVectorRetriever(BaseRetriever):
    """Custom retriever that retrieves candidate documents from a base vector retriever
    and applies Cohere cross-encoder reranking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    cohere_reranker: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        if len(docs) == 1:
            return docs
        try:
            return list(self.cohere_reranker.compress_documents(documents=docs, query=query))
        except Exception:
            logger.exception("Reranking failed in retriever. Returning candidate documents.")
            return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if hasattr(self.base_retriever, "ainvoke"):
            docs = await self.base_retriever.ainvoke(query)
        else:
            docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        if len(docs) == 1:
            return docs
        try:
            return list(self.cohere_reranker.compress_documents(documents=docs, query=query))
        except Exception:
            logger.exception("Reranking failed in retriever. Returning candidate documents.")
            return docs


class VectorStoreManager:
    """Manages AstraDB cloud vector store with LiteLLM Gateway embeddings and Cohere Reranking."""

    def __init__(self):
        logger.info("Initializing VectorStoreManager with LiteLLM Gateway embeddings and AstraDB.")
        self._lock = threading.Lock()
        self.embeddings = Config.get_embeddings()
        self.vectorstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name=Config.ASTRA_DB_COLLECTION_NAME,
            token=Config.ASTRA_DB_API_KEY,
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
        )
        self._cached_retriever: Optional[RerankedVectorRetriever] = None
        self._current_k: Optional[int] = None
        self._current_search_type: Optional[str] = None
        self.cohere_reranker = CohereRerank(
            model=Config.COHERE_RERANKER_MODEL,
            top_n=Config.COHERE_RERANKER_TOP_N
        )

    def _add_documents_to_vectorstore(self, split_docs: List[Document], vector_store: AstraDBVectorStore) -> AstraDBVectorStore:
        """
        Embeds documents and prevents duplicates using deterministic composite chunk identity hashing.
        Args:
            split_docs (List[Document]): List of documents to embed
            vector_store (AstraDBVectorStore): AstraDB vector store
        Returns:
            AstraDBVectorStore: AstraDB vector store
        """
        if not split_docs:
            logger.error("No documents to embed")
            raise ValueError("No documents to embed")

        vector_store_collection = vector_store.astra_env.collection

        # Generate unique deterministic IDs based on composite identity (source, version, content, chunking, model)
        doc_id_map = {compute_chunk_identity(doc): doc for doc in split_docs}

        all_ids = list(doc_id_map.keys())
        existing_ids = set()

        # Batch query in chunks of 100 to avoid O(N) sequential network roundtrips
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            chunk_ids = all_ids[i:i + batch_size]
            try:
                cursor = vector_store_collection.find(
                    filter={"_id": {"$in": chunk_ids}},
                    projection={"_id": True}
                )
                if cursor:
                    for record in cursor:
                        rec_id = record.get("_id")
                        if rec_id:
                            existing_ids.add(rec_id)
            except Exception as e:
                logger.warning("Batch deduplication query fallback for chunk: %s", e)
                for doc_id in chunk_ids:
                    if vector_store_collection.find_one(filter={"_id": doc_id}, projection={"_id": True}):
                        existing_ids.add(doc_id)

        new_ids = [doc_id for doc_id in all_ids if doc_id not in existing_ids]
        new_docs = [doc_id_map[doc_id] for doc_id in new_ids]

        if new_docs:
            vector_store.add_documents(new_docs, ids=new_ids)
            logger.info("Processed and inserted %d new chunks into AstraDB vectorstore (skipped %d existing).", len(new_docs), len(existing_ids))
        else:
            logger.info("All %d documents already exist in the AstraDB vectorstore.", len(split_docs))
        return vector_store

    def create_vectorstore(self, documents: List[Document]) -> AstraDBVectorStore:
        """
        Create or update AstraDB vector store from documents with deduplication.
        Args:
            documents (List[Document]): List of documents to add
        Returns:
            AstraDBVectorStore: AstraDB vector store
        """
        if self.vectorstore is not None and len(documents) > 0:
            with self._lock:
                logger.info("Adding %d documents to AstraDB vector store...", len(documents))

                # Invalidate cached retriever so it is rebuilt
                self._cached_retriever = None
                self._current_k = None
                self._current_search_type = None

                self.vectorstore = self._add_documents_to_vectorstore(documents, self.vectorstore)
                logger.info("AstraDB vector store successfully updated.")
        elif self.vectorstore is None:
            logger.error("Cannot add documents; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")
        elif len(documents) == 0:
            logger.error("Cannot add documents; no documents provided.")
            raise ValueError("No documents provided to add to vectorstore.")
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to vector store and return the number of chunks added.
        Args:
            documents (List[Document]): List of documents to add
        Returns:
            int: Number of chunks added
        """
        self.create_vectorstore(documents)
        return len(documents)

    def create_retriever(
        self,
        vectorstore: AstraDBVectorStore,
        k: int = 4,
        search_type: str = "similarity",
        filter: Optional[dict] = None,
        session_id: Optional[str] = None
    ) -> RerankedVectorRetriever:
        """
        Create a retriever from AstraDB vector store with dynamic Cohere reranking and optional session filtering.
        Thread-safe: constructs dedicated retriever and reranker instances without mutating shared global state.
        
        Args:
            vectorstore (AstraDBVectorStore): AstraDB vector store instance
            k (int): Number of final documents to retrieve after reranking
            search_type (str): Type of search to perform ("similarity" or "mmr")
            filter (dict, optional): Metadata filter for vector store
            session_id (str, optional): Target session ID to isolate retrieval
        Returns:
            RerankedVectorRetriever: Custom retriever that performs vector search and Cohere reranking
        """
        filter_dict = dict(filter) if filter else {}
        if session_id:
            filter_dict["session_id"] = session_id

        # Oversample candidates (2x k, min 10) so Cohere reranker has high-recall candidates
        candidate_k = max(k * 2, 10)
        logger.info("Creating AstraDB retriever with target k=%d (candidate_k=%d), search_type='%s', filter=%s", k, candidate_k, search_type, filter_dict)

        if vectorstore is None:
            logger.error("Cannot create retriever; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")

        search_kwargs: dict[str, Any] = {"k": candidate_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        vec_retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        # Dedicated Cohere reranker instance per retriever
        cohere_reranker = CohereRerank(
            model=Config.COHERE_RERANKER_MODEL,
            top_n=k
        )

        reranked_retriever = RerankedVectorRetriever(
            base_retriever=vec_retriever,
            cohere_reranker=cohere_reranker
        )

        # Cache default zero-filter retriever
        if filter is None and session_id is None:
            self._cached_retriever = reranked_retriever
            self._current_k = k
            self._current_search_type = search_type

        return reranked_retriever

    def rerank_documents(self, query: str, documents: List[Document], top_n: Optional[int] = None) -> List[Document]:
        """
        Rerank documents based on query using Cohere Rerank.
        Args:
            query (str): Query to rerank documents for
            documents (List[Document]): List of documents to rerank
            top_n (int, optional): Optional override for number of reranked docs
        Returns:
            List[Document]: Reranked list of documents
        """
        if not query or not query.strip():
            raise ValueError("No query provided.")
        if not documents:
            logger.info("No documents to rerank.")
            return []
        if len(documents) == 1:
            logger.info("Only one document retrieved. Skipping reranking.")
            return documents

        try:
            reranker = self.cohere_reranker
            if top_n is not None and top_n != self.cohere_reranker.top_n:
                reranker = CohereRerank(model=Config.COHERE_RERANKER_MODEL, top_n=top_n)
            reranked = reranker.compress_documents(documents=documents, query=query)
            logger.info("Reranked %d candidate documents into %d documents.", len(documents), len(reranked))
            return list(reranked)
        except Exception:
            logger.exception("Failed to rerank. Returning original documents.")
            return documents

    def get_retriever(
        self,
        k: int = 4,
        search_type: str = "similarity",
        filter: Optional[dict] = None,
        session_id: Optional[str] = None
    ) -> RerankedVectorRetriever:
        """
        Get the reranked vector retriever, initializing or creating a dedicated per-query retriever.
        """
        if self.vectorstore is None:
            logger.error("Cannot get retriever; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")

        if (
            self._cached_retriever is None
            or self._current_k != k
            or self._current_search_type != search_type
            or filter is not None
            or session_id is not None
        ):
            return self.create_retriever(
                self.vectorstore,
                k=k,
                search_type=search_type,
                filter=filter,
                session_id=session_id
            )
        return self._cached_retriever

    def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        filter: Optional[dict] = None,
        session_id: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve and rerank documents from AstraDB vector store in a single optimized pass.
        Args:
            query (str): Query to retrieve documents for
            k (int): Number of documents to retrieve
            search_type (str): Type of search to perform ("similarity" or "mmr")
            filter (dict, optional): Metadata filter for vector store
            session_id (str, optional): Target session ID
        Returns:
            List[Document]: Reranked list of top-k documents
        """
        if not query or not query.strip():
            logger.error("Cannot retrieve; query is empty.")
            raise ValueError("No query provided to retrieve documents for.")

        retriever = self.get_retriever(
            k=k,
            search_type=search_type,
            filter=filter,
            session_id=session_id
        )
        logger.info("Retrieving & reranking documents for query: %s (k=%d, search_type='%s')", repr(query), k, search_type)
        try:
            results = retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving documents for query: %s", query)
            raise e

        logger.info("Retrieved %d relevant documents after reranking.", len(results))
        return results