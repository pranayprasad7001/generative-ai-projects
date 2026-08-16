"""Vector Store and Hybrid Retrieval Management Module.

This module coordinates:
- AstraDB vector embeddings via LiteLLM Gateway
- BM25 lexical retriever hydration and disk caching
- Reciprocal Rank Fusion via LangChain EnsembleRetriever
- Semantic document compression and reranking via Cohere Rerank
- 5-dimension deterministic chunk hashing for zero-duplicate ingestion
"""

import os
import json
import pickle
import logging
import hashlib
import threading
from pathlib import Path
from typing import List, Optional

from langchain_cohere import CohereRerank
from langchain_astradb import AstraDBVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.schema import Document
from config.llmgateway_config import Config

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data")
BM25_CACHE_PATH = CACHE_DIR / "bm25_corpus.json"
BM25_LEGACY_PKL_PATH = CACHE_DIR / "bm25_corpus.pkl"


def compute_chunk_identity(doc: Document, default_embedding_model: str = Config.EMBEDDING_MODEL) -> str:
    """
    Compute a strong, deterministic chunk identity hash based on:
    1. Source document path or URL
    2. Document version (default 'v1')
    3. Exact chunk text content
    4. Chunking configuration (strategy, size, overlap, chunk index)
    5. Embedding model name
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


class VectorStoreManager:
    """Manages AstraDB vector stores with LiteLLM Gateway embeddings and persistent Hybrid Retrieval."""

    def __init__(self):
        logger.info("Initializing VectorStoreManager with LiteLLM Gateway embeddings.")
        self._lock = threading.Lock()
        self.embeddings = Config.get_embeddings()
        self.vectorstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name=Config.ASTRA_DB_COLLECTION_NAME,
            token=Config.ASTRA_DB_API_KEY,
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT
        )
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.compression_retriever: Optional[ContextualCompressionRetriever] = None
        self.retriever = None
        self.documents: List[Document] = []
        self._current_k: Optional[int] = None
        self._current_search_type: Optional[str] = None
        self.cohere_reranker = CohereRerank(
            model=Config.COHERE_RERANKER_MODEL,
            top_n=Config.COHERE_RERANKER_TOP_N
        )

        # Hydrate documents for BM25 from disk cache or AstraDB
        self._hydrate_documents()

    def _save_documents_to_cache(self):
        """Persist document corpus to local JSON disk cache for secure, fast reload."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            serializable_docs = [
                {"page_content": doc.page_content, "metadata": getattr(doc, "metadata", {})}
                for doc in self.documents
            ]
            with open(BM25_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, ensure_ascii=False)
            logger.info("Saved %d documents to BM25 JSON cache at %s", len(self.documents), BM25_CACHE_PATH)
        except Exception as e:
            logger.warning("Could not save BM25 corpus cache: %s", e)

    def _load_documents_from_cache(self) -> bool:
        """Load document corpus from local JSON cache (or legacy pkl)."""
        if BM25_CACHE_PATH.exists():
            try:
                with open(BM25_CACHE_PATH, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, list) and cached_data:
                    self.documents = [
                        Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}))
                        for item in cached_data
                        if isinstance(item, dict)
                    ]
                    logger.info("Loaded %d documents from local BM25 JSON cache.", len(self.documents))
                    return True
            except Exception as e:
                logger.warning("Failed to load BM25 JSON corpus cache: %s", e)

        # Fallback to legacy pkl cache if exists and migrate to JSON
        if BM25_LEGACY_PKL_PATH.exists():
            try:
                with open(BM25_LEGACY_PKL_PATH, "rb") as f:
                    cached_docs = pickle.load(f)
                if isinstance(cached_docs, list) and cached_docs:
                    self.documents = cached_docs
                    logger.info("Loaded %d documents from legacy BM25 pkl cache. Migrating to JSON...", len(self.documents))
                    self._save_documents_to_cache()
                    return True
            except Exception as e:
                logger.warning("Failed to load legacy BM25 pkl cache: %s", e)

        return False

    def _load_documents_from_astradb(self):
        """Fetch indexed document content from AstraDB to hydrate BM25 on cold starts."""
        try:
            if not self.vectorstore or not hasattr(self.vectorstore, "astra_env"):
                return
            collection = getattr(self.vectorstore.astra_env, "collection", None)
            if collection is None:
                return

            logger.info("Hydrating BM25 corpus from AstraDB collection '%s'...", Config.ASTRA_DB_COLLECTION_NAME)
            cursor = collection.find(projection={"content": True, "metadata": True})
            loaded_docs = []
            for record in cursor:
                content = record.get("content")
                if content:
                    metadata = record.get("metadata") or {}
                    loaded_docs.append(Document(page_content=content, metadata=metadata))

            if loaded_docs:
                self.documents = loaded_docs
                logger.info("Successfully hydrated %d documents from AstraDB for BM25.", len(self.documents))
                self._save_documents_to_cache()
            else:
                logger.info("No existing documents found in AstraDB collection.")
        except Exception as e:
            logger.warning("Could not hydrate documents from AstraDB: %s", e)

    def _hydrate_documents(self):
        """Initialize document corpus for BM25 from disk cache or AstraDB."""
        if not self._load_documents_from_cache():
            self._load_documents_from_astradb()

    def _add_documents_to_vectorstore(self, split_docs: List[Document], vector_store: AstraDBVectorStore) -> AstraDBVectorStore:
        """
        Embeds documents and prevents duplicates using deterministic hashing IDs.
        Args:
            split_docs: List[Document]: List of documents to embed
            vector_store: AstraDBVectorStore: AstraDB vector store
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
            logger.info("Processed and inserted %d new chunks into vectorstore (skipped %d existing).", len(new_docs), len(existing_ids))
        else:
            logger.info("All %d documents already exist in the vectorstore.", len(split_docs))
        return vector_store

    def create_vectorstore(self, documents: List[Document]) -> AstraDBVectorStore:
        """
        Create or update AstraDB vector store from documents
        Args:
            documents (List[Document]): List of documents to add
        Returns:
            AstraDBVectorStore: AstraDB vector store
        """
        if self.vectorstore is not None and len(documents) > 0:
            with self._lock:
                logger.info("Adding %d documents to AstraDB vector store...", len(documents))
                self.documents.extend(documents)
                self._save_documents_to_cache()

                # Invalidate cached retriever so it is rebuilt with new corpus
                self.bm25_retriever = None
                self.compression_retriever = None
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
    ) -> ContextualCompressionRetriever:
        """
        Create a hybrid retriever from vector store with dynamic Cohere reranking and optional session filtering.
        Thread-safe: constructs dedicated retriever and reranker instances without mutating shared global state.
        
        Args:
            vectorstore: AstraDBVectorStore 
            k (int): Number of final documents to retrieve after reranking
            search_type (str): Type of search to perform ("similarity" or "mmr")
            filter (dict, optional): Metadata filter for vector store
            session_id (str, optional): Target session ID to isolate retrieval
        Returns:
            ContextualCompressionRetriever: Reranked compression retriever
        """
        filter_dict = dict(filter) if filter else {}
        if session_id:
            filter_dict["session_id"] = session_id

        # Oversample candidates (2x k, min 10) so Cohere reranker has high-recall candidates
        candidate_k = max(k * 2, 10)
        logger.info("Creating hybrid retriever with target k=%d (candidate_k=%d), search_type='%s', filter=%s", k, candidate_k, search_type, filter_dict)

        if vectorstore is None:
            logger.error("Cannot create retriever; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")

        search_kwargs = {"k": candidate_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        vec_retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        # Create a dedicated Cohere reranker instance per retriever to prevent mutating shared state
        cohere_reranker = CohereRerank(
            model=Config.COHERE_RERANKER_MODEL,
            top_n=k
        )

        target_documents = self.documents
        if session_id and self.documents:
            target_documents = [d for d in self.documents if d.metadata.get("session_id") == session_id]

        if target_documents:
            bm25_retriever = BM25Retriever.from_documents(target_documents)
            bm25_retriever.k = candidate_k
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vec_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )
            base_retriever = ensemble_retriever
            logger.info("EnsembleRetriever initialized with AstraDB vector and BM25 lexical search.")
        else:
            logger.warning("No documents loaded in corpus yet; BM25 disabled. Using vector store retriever directly.")
            bm25_retriever = None
            ensemble_retriever = None
            base_retriever = vec_retriever

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=cohere_reranker,
            base_retriever=base_retriever
        )

        # For default zero-argument caching
        if filter is None and session_id is None:
            self.retriever = vec_retriever
            self.bm25_retriever = bm25_retriever
            self.ensemble_retriever = ensemble_retriever
            self.compression_retriever = compression_retriever
            self._current_k = k
            self._current_search_type = search_type

        return compression_retriever

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
            return reranked
        except Exception:
            logger.exception("Failed to rerank. Returning original documents.")
            return documents

    def get_retriever(
        self,
        k: int = 4,
        search_type: str = "similarity",
        filter: Optional[dict] = None,
        session_id: Optional[str] = None
    ) -> ContextualCompressionRetriever:
        """
        Get the compression retriever, initializing or creating a dedicated per-query retriever.
        """
        if self.vectorstore is None:
            logger.error("Cannot get retriever; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")

        if (
            self.compression_retriever is None
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
        return self.compression_retriever

    def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        filter: Optional[dict] = None,
        session_id: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve and rerank documents from vector store and BM25 in a single optimized pass.
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
            # ContextualCompressionRetriever handles both hybrid retrieval and Cohere reranking in 1 pass
            results = retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving documents for query: %s", query)
            raise e

        logger.info("Retrieved %d relevant documents after reranking.", len(results))
        return results