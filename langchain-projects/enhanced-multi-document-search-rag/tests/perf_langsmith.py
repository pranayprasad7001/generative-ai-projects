import time
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

base_url = "http://127.0.0.1:4000"
api_key = os.getenv("LITELLM_MASTER_KEY")

def run_completion():
    llm = ChatOpenAI(
        model="gpt-oss-120b-groq",
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=100
    )
    start = time.time()
    try:
        resp = llm.invoke("Hi")
        print(f"Completion took {time.time() - start:.2f} seconds")
    except Exception as e:
        print("Failed:", e)

# Test 1: With current environment (LangSmith enabled)
print("--- Test 1: LangSmith enabled (current env) ---")
print("LANGSMITH_TRACING:", os.getenv("LANGSMITH_TRACING"))
print("LANGSMITH_ENDPOINT:", os.getenv("LANGSMITH_ENDPOINT"))
run_completion()

# Test 2: Disable LangSmith tracing
print("\n--- Test 2: LangSmith disabled ---")
os.environ["LANGSMITH_TRACING"] = "false"
run_completion()
