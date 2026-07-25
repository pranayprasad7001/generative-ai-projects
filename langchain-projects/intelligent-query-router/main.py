import os
import uuid
from dotenv import load_dotenv
from model_init import initialize_core_system
from query_router import route_to_datasource
from preprocess_docs import load_docs_from_urls, embed_documents, vectorstore_to_retriever
from tool_call import tools_init
from graph import build_graph

# Load environment variables
load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

astradb_api_key = os.getenv("ASTRA_DB_API_KEY")
astradb_endpoint = os.getenv("ASTRA_DB_ENDPOINT")
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
llm_model_name = "openai/gpt-oss-120b"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Core Systems & Tools
llm, embedding_model, astra_vector_store = initialize_core_system(
    astradb_api_key, astradb_endpoint, hf_api_key, groq_api_key, llm_model_name, embedding_model_name
)

wiki, arxiv = tools_init()

# Ingest Documents & Prepare Retriever
urls = [
    'https://lilianweng.github.io/posts/2026-07-04-harness/',
    'https://lilianweng.github.io/posts/2025-05-01-thinking/',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/'
]

splitted_docs = load_docs_from_urls(urls)
vector_store = embed_documents(splitted_docs, astra_vector_store)
retriever = vectorstore_to_retriever(vector_store)

# Initialize Router
question_router = route_to_datasource(llm)

# Pass llm directly to build_graph
app = build_graph(
    retriever=retriever,
    question_router=question_router,
    wiki_tool=wiki,
    arxiv_tool=arxiv,
    llm=llm
)

# Run an example query
if __name__ == "__main__":
    # Create a unique thread_id to track conversation memory for this session
    session_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("==================================================")
    print("Welcome to Agent Chat! Type 'exit', 'quit', or 'q' to end.")
    print("==================================================")

    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            # Exit check
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting application... Goodbye!")
                break
                
            if not user_input:
                continue

            # Pass user question and thread configuration
            inputs = {"question": user_input}
            
            # Execute graph turn (streaming output prints directly inside generate node)
            app.invoke(inputs, config=session_config)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break