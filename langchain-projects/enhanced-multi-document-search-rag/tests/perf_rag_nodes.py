import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for console
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set base url to 127.0.0.1 (overriding the default localhost)
os.environ["LITELLM_BASE_URL"] = "http://127.0.0.1:4000"

from config.llmgateway_config import Config
from nodes.adaptive_node import AdaptiveRAGNodes
from state.adaptive_state import AdaptiveRAGState

# Dummy retriever
class DummyRetriever:
    def invoke(self, q):
        return []

print("Initializing LLMs...")
start = time.time()
llm_generator = Config.get_llm_generator()
llm_checker = Config.get_llm_checker()
print(f"LLMs initialized in {time.time() - start:.4f} seconds")

print("Initializing AdaptiveRAGNodes...")
start = time.time()
nodes = AdaptiveRAGNodes(DummyRetriever(), llm_generator=llm_generator, llm_checker=llm_checker)
print(f"AdaptiveRAGNodes initialized in {time.time() - start:.4f} seconds")

state = AdaptiveRAGState(question="What is the core idea of LLM agents?")

print("\n--- Running query_analyzer node ---")
start = time.time()
try:
    res = nodes.query_analyzer(state)
    print(f"query_analyzer node completed in {time.time() - start:.4f} seconds")
    print("Result analysis:", res.analysis)
    print("Result tool_type:", res.tool_type)
except Exception as e:
    print("Failed:", e)
