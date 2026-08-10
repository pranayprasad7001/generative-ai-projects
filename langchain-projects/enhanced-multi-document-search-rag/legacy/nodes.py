"""Langgraph nodes for RAG workflow"""

from state.rag_state import RAGState

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes with retriever and llm

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents using the retriever

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved documents
        """
        retrieved_docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=retrieved_docs
        )
        
    def generate_response(self, state: RAGState) -> RAGState:
        """
        Generate response using retrieved documents

        Args:
            state: Current RAG state with retrieved documents

        Returns:
            Updated RAG state with generated response
        """
        context = "\n\n".join([
            f"""Context: {doc.page_content}\n"""
            for doc in state.retrieved_docs
        ])

        prompt = f"""Answer the question based on the context
        Context: {context}
        Question: {state.question}
        If the answer is not in the context, say so
        """

        response = self.llm.invoke(prompt)
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
        
        