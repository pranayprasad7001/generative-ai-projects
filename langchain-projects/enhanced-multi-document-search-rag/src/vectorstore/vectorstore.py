import logging
import hashlib
from config.llmgateway_config import Config
from typing import List
from langchain_cohere import CohereRerank
from langchain_astradb import AstraDBVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.schema import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages AstraDB vector stores with Google Generative AI embeddings"""
    
    def __init__(self, embedding_model: str = Config.EMBEDDING_MODEL):
        logger.info("Initializing VectorStoreManager with embedding model: %s", embedding_model)
        self.embedding_model = embedding_model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model, google_api_key=Config.GOOGLE_API_KEY)
        self.vectorstore = AstraDBVectorStore(embedding=self.embeddings, collection_name=Config.ASTRA_DB_COLLECTION_NAME, token=Config.ASTRA_DB_API_KEY, api_endpoint=Config.ASTRA_DB_API_ENDPOINT)
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.compression_retriever = None
        self.retriever = None
        self.documents = []
        self._current_k = None
        self._current_search_type = None
        self.cohere_reranker = CohereRerank(model=Config.COHERE_RERANKER_MODEL, top_n=Config.COHERE_RERANKER_TOP_N)

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
        # Generate unique deterministic IDs based on content and source
        ids = []
        docs = []
        for doc in split_docs:
            source = doc.metadata.get("source", "")
            content = doc.page_content
            unique_string = f"{source}_{content}"
            doc_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
            
            existing_doc = vector_store_collection.find_one(
                filter={"_id": doc_id},
                projection={"_id": True}
            )

            if existing_doc:
                logger.info(f"Document with ID {doc_id} already exists. Skipping.")
                continue
            ids.append(doc_id)
            docs.append(doc)

        if docs:
            vector_store.add_documents(docs, ids=ids)
            logger.info(f"Processed {len(docs)} chunks into vectorstore safely!")
        else:
            logger.info("All documents already exist in the vectorstore.")
        return vector_store

    def create_vectorstore(self, documents: List[Document]) -> AstraDBVectorStore:
        """
        Create a AstraDB vector store from documents
        Args:
            documents (List[Document]): List of documents to create vector store from
        Returns:
            AstraDBVectorStore: AstraDB vector store
        """
        if self.vectorstore is not None and len(documents) > 0:
            logger.info("Adding %d documents to AstraDB vector store...", len(documents))
            self.documents.extend(documents)
            self.vectorstore = self._add_documents_to_vectorstore(documents, self.vectorstore)
            logger.info("AstraDB vector store successfully updated.")
        elif self.vectorstore is None:
            logger.error("Cannot add documents; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")
        elif len(documents) == 0:
            logger.error("Cannot add documents; no documents provided.")
            raise ValueError("No documents provided to add to vectorstore.")
        return self.vectorstore

    def create_retriever(self, vectorstore: AstraDBVectorStore, k: int = 4, search_type: str = "similarity") -> ContextualCompressionRetriever :
        """
        Create a retriever from vector store
        Args:
            vectorstore: AstraDBVectorStore 
            k (int): Number of documents to retrieve
            search_type (str): Type of search to perform, can be "similarity" or "mmr"
        Returns:
            Retriever: Retriever
        """
        if self.compression_retriever is None or self._current_k != k or self._current_search_type != search_type:
            if vectorstore is None:
                logger.error("Cannot create retriever; vectorstore is not initialized.")
                raise ValueError("No vectorstore found, please create or load a vectorstore first.")
            logger.info("Creating retriever with k=%d, search_type='%s'", k, search_type)
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k})
            
            # Initialize BM25Retriever using processed documents
            if self.documents:
                self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            else:
                logger.warning("No documents stored in VectorStoreManager to initialize BM25Retriever. Initializing with a placeholder document.")
                placeholder_doc = Document(page_content="placeholder")
                self.bm25_retriever = BM25Retriever.from_documents([placeholder_doc])
                
            self.bm25_retriever.k = k
            self.ensemble_retriever = EnsembleRetriever(retrievers=[self.retriever, self.bm25_retriever], weights=[0.7, 0.3])
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.cohere_reranker,
                base_retriever=self.ensemble_retriever
            )
            self._current_k = k
            self._current_search_type = search_type
            
        return self.compression_retriever
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on query
        Args:
            query (str): Query to rerank documents for
            documents (List[Document]): List of documents to rerank
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
            reranked = self.cohere_reranker.compress_documents(documents=documents, query=query)
            logger.info("Reranked %d candidate documents into %d documents.", len(documents), len(reranked))
            return reranked
        except Exception:
            logger.exception("Failed to rerank. Returning original documents.")
            return documents

    def get_retriever(self, k: int = 4, search_type: str = "similarity") -> ContextualCompressionRetriever:
        """
        Get the ensemble retriever, initializing it if necessary.
        """
        if self.compression_retriever is None or self._current_k != k or self._current_search_type != search_type:
            if self.vectorstore is None:
                logger.error("Cannot get retriever; vectorstore is not initialized.")
                raise ValueError("No vectorstore found, please create or load a vectorstore first.")
            self.create_retriever(self.vectorstore, k=k, search_type=search_type)
        return self.compression_retriever
        
    def retrieve(self, query: str, k: int = 4, search_type: str = "similarity") -> List[Document]:
        """
        Retrieve documents from vector store based on query
        Args:
            query (str): Query to retrieve documents for
            k (int): Number of documents to retrieve
            search_type (str): Type of search to perform, can be "similarity" or "mmr"
        Returns:
            List[Document]: List of documents
        """
        if self.ensemble_retriever is None or self._current_k != k or self._current_search_type != search_type:
            if self.vectorstore is None:
                logger.error("Cannot retrieve; vectorstore is not initialized.")
                raise ValueError("No vectorstore found, please create or load a vectorstore first.")
            self.create_retriever(self.vectorstore, k, search_type)
        if query is None or query == "":
            logger.error("Cannot retrieve; query is None.")
            raise ValueError("No query provided to retrieve documents for.")
        logger.info("Retrieving documents for query: %s (k=%d, search_type='%s')", repr(query), k, search_type)
        try:
            results = self.ensemble_retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving documents for query: %s", query)
            raise e
        logger.info("Retrieved %d relevant documents.", len(results))
        reranked_docs = self.rerank_documents(query, results)
        return reranked_docs