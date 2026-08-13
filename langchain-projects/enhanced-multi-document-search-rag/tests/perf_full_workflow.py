import time
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for console
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set base url to 127.0.0.1
os.environ["LITELLM_BASE_URL"] = "http://127.0.0.1:4000"

from config.llmgateway_config import Config
from vectorstore.vectorstore import VectorStoreManager
from graph_builder.adaptive_graph_builder import GraphBuilder

async def main():
    print("Initializing VectorStoreManager...")
    start = time.time()
    vector_store = VectorStoreManager()
    retriever = vector_store.get_retriever(search_type="similarity")
    print(f"VectorStoreManager initialized in {time.time() - start:.4f} seconds")

    print("Initializing LLM...")
    start = time.time()
    llm = Config.get_llm()
    print(f"LLM initialized in {time.time() - start:.4f} seconds")

    print("Building RAG Graph...")
    start = time.time()
    graph_builder = GraphBuilder(retriever=retriever, llm=llm)
    graph_builder.build_graph()
    print(f"RAG Graph built in {time.time() - start:.4f} seconds")

    question = "what information is there in pdf regarding planets"
    print(f"\nRunning workflow for query: '{question}'")
    
    start_run = time.time()
    result = await graph_builder.run(question)
    elapsed = time.time() - start_run
    
    print(f"\nWorkflow finished in {elapsed:.2f} seconds")
    print("Answer:", result.get("answer"))
    print("Total Cost:", result.get("total_cost"))

if __name__ == "__main__":
    asyncio.run(main())
