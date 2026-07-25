import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def initialize_core_system(astradb_api_key, astradb_endpoint, hf_api_key, groq_api_key, llm_model_name, embedding_model_name):
    """Initializes the backend components dynamically without global caching to ensure multi-user isolation."""

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        model_kwargs={"device": "cpu"}
    )

    llm = ChatGroq(model=llm_model_name, api_key=GROQ_API_KEY, verbose=True)

    astra_vector_store = AstraDBVectorStore(
        collection_name="qa_multi_agent",
        embedding=embedding_model,
        api_endpoint=astradb_endpoint,
        token=astradb_api_key,
    )

    print("Loaded all models and vectorstore successfully!")
    return llm, embedding_model, astra_vector_store