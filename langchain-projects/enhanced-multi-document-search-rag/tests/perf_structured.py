import time
import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Ensure UTF-8 output encoding for console
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

class ToolUse(BaseModel):
    analysis: str = Field(description="Reasoning process")
    tool_type: str = Field(description="Must be either 'vector_search' or 'external_search'")

base_url = "http://127.0.0.1:4000"
api_key = os.getenv("LITELLM_MASTER_KEY") or "---------------------------"

print("Initializing ChatOpenAI with gpt-oss-120b-groq...")
llm = ChatOpenAI(
    model="gpt-oss-120b-groq",
    api_key=api_key,
    base_url=base_url,
    temperature=0.2,
    max_tokens=7000
)

# Test normal completion
print("\n--- Testing normal completion (no structured output) ---")
start = time.time()
try:
    resp = llm.invoke("What is the core idea of LLM agents?")
    print(f"Normal completion succeeded in {time.time() - start:.2f} seconds")
    print("Content:", resp.content[:200])
except Exception as e:
    print("Normal completion failed:", e)

# Test structured output
print("\n--- Testing structured output ---")
start = time.time()
try:
    structured_llm = llm.with_structured_output(ToolUse)
    resp = structured_llm.invoke("What is the core idea of LLM agents?")
    print(f"Structured output succeeded in {time.time() - start:.2f} seconds")
    print("Structured response:", resp)
except Exception as e:
    print("Structured output failed:", e)
