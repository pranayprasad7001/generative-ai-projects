import sys
import os

# Add the src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.llmgateway_config import Config
from vectorstore.vectorstore import VectorStoreManager
from graph_builder.adaptive_graph_builder import GraphBuilder

# Initialize LLM and Vector Store
llm = Config.get_llm()
vector_store = VectorStoreManager()
retriever = vector_store.get_retriever()

# Build and compile the graph without custom checkpointer (managed by platform)
graph_builder = GraphBuilder(retriever=retriever, llm=llm)
graph = graph_builder.build_graph(use_checkpointer=False)
