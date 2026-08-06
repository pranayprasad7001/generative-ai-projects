import logging
from pathlib import Path
from typing import List, Union
from langchain_classic.vectorstores import FAISS
from langchain_classic.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages FAISS vector stores with HuggingFace embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Initializing VectorStoreManager with model: %s", model_name)
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = None
        self.retriever = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from documents
        Args:
            documents (List[Document]): List of documents to create vector store from
        Returns:
            FAISS: FAISS vector store
        """
        if self.vectorstore is None:
            logger.info("Creating FAISS vector store from %d documents...", len(documents))
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            logger.info("FAISS vector store successfully created.")
        return self.vectorstore

    def save_vectorstore(self, directory: Union[str, Path]) -> None:
        """
        Save FAISS vector store to disk
        Args:
            directory (Union[str, Path]): Directory to save vector store to
        """
        if self.vectorstore is None:
            logger.error("Attempted to save vector store, but self.vectorstore is None.")
            raise ValueError("No vectorstore to save, please create vector store first.")
        else:
            logger.info("Saving FAISS vector store to directory: %s", directory)
            self.vectorstore.save_local(str(directory))
            logger.info("Vector store successfully saved.")

    def load_vectorstore(self, directory: Union[str, Path]) -> FAISS:
        """
        Load FAISS vector store from disk
        Args:
            directory (Union[str, Path]): Directory to load vector store from
        Returns:
            FAISS: FAISS vector store
        """
        if os.path.exists(directory):
            logger.info("Loading FAISS vector store from directory: %s", directory)
            self.vectorstore = FAISS.load_local(str(directory), self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store successfully loaded.")
            return self.vectorstore
        else:
            logger.error("Vector store directory not found: %s", directory)
            raise FileNotFoundError(f"Directory not found: {directory}")

    def create_retriever(self, vectorstore: FAISS, k: int = 4, search_type: str = "similarity") :
        """
        Create a retriever from vector store
        Args:
            vectorstore (FAISS): FAISS vector store
            k (int): Number of documents to retrieve
            search_type (str): Type of search to perform, can be "similarity" or "mmr"
        Returns:
            Retriever: Retriever
        """
        if self.retriever is None:
            logger.info("Creating retriever with k=%d, search_type='%s'", k, search_type)
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k})
        return self.retriever
        
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
        if self.vectorstore is None:
            logger.error("Cannot retrieve; vectorstore is not initialized.")
            raise ValueError("No vectorstore found, please create or load a vectorstore first.")
        if self.retriever is None:
            self.create_retriever(self.vectorstore, k, search_type)
        
        logger.info("Retrieving documents for query: %s (k=%d, search_type='%s')", repr(query), k, search_type)
        results = self.retriever.invoke(query)
        logger.info("Retrieved %d relevant documents.", len(results))
        return results