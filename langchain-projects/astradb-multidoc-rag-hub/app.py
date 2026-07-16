import os
import shutil
import uuid
import atexit
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.indexes import SQLRecordManager, index
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.messages import trim_messages
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv

# Load environment variables as default fallbacks if they exist locally
load_dotenv()

os.environ["LANGSMITH_TRACING_V2"] = st.secrets["LANGSMITH_TRACING_V2"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# --------- END POINTS & MODEL INITIALIZER (SESSION SCOPED) --------
def initialize_core_system(astradb_api_key, astradb_endpoint, hf_api_key, llm_model_name):
    """Initializes the backend components dynamically without global caching to ensure multi-user isolation."""
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model = HuggingFaceEndpointEmbeddings(
        repo_id=embedding_model_name, 
        huggingfacehub_api_token=hf_api_key, 
        task="feature-extraction"
    )

    llm_model = HuggingFaceEndpoint(
        repo_id=llm_model_name, 
        huggingfacehub_api_token=hf_api_key, 
        task="text-generation"
    )

    llm = ChatHuggingFace(llm=llm_model, verbose=True)
    
    astra_vector_store = AstraDBVectorStore(
        collection_name="qa_mini",
        embedding=embedding_model,
        api_endpoint=astradb_endpoint,
        token=astradb_api_key,
    )

    return llm, astra_vector_store

# --------- DOCUMENT LOADER & INDEXER --------
def process_and_index_multiple_pdfs(uploaded_files, astra_vector_store, session_id):
    """Save uploaded files locally in a session-isolated directory, split them, and sync using full cleanup."""
    all_chunks = []
    
    # Isolate directories by session_id to prevent cross-user data clashing
    temp_dir = f"./temp_docs_{session_id}"
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Inject structural metadata isolation tag into every extracted document chunk
        for chunk in chunks:
            chunk.metadata["session_id"] = session_id
            
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"message": "No text extracted from documents."}

    # Isolate the SQLite cache database ledger to handle concurrent deletion tracking accurately
    record_manager = SQLRecordManager(
        namespace=f"astradb/qa_mini/{session_id}", 
        db_url=f"sqlite:///record_manager_cache_{session_id}.sql",
    )
    record_manager.create_schema()

    # Sync to Astra DB using full cleanup mode to handle runtime file changes safely
    result = index(
        all_chunks,
        record_manager,
        astra_vector_store,
        cleanup="full",       #
        source_id_key="source",   
    )
    return result

# ----------- CLEANUP STORAGE METHOD -----------
def cleanup_session_storage(session_id):
    """Deletes temporary disk directories and internal SQLite ledger files associated with this session identifier."""
    temp_dir = f"./temp_docs_{session_id}"
    sqlite_file = f"record_manager_cache_{session_id}.sql"
    
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
    if os.path.exists(sqlite_file):
        try:
            os.remove(sqlite_file)
        except Exception:
            pass

# ----------- SESSION HISTORY -----------
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_history_store[session_id]

# ---------- HISTORY TRIMMER ----------
def history_trimmer_func():
    """Trim the conversation history using the fast approximate heuristic shortcut."""
    return trim_messages(
        max_tokens=500,
        strategy="last",
        token_counter="approximate",
        include_system=False,
        allow_partial=False,
        start_on="human",
    )

# --------- CONTEXTUALIZE QUERY ---------
def contextualize_query():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    return ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

# -------- QUESTION QUERY --------
def question_query():
    system_prompt = """
    You are a helpful AI Assistant. Use the following context parsed from multiple documents: {context} to answer the question.
    If the answer is not in the context, say "I don't know".
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

# -------- RAG PIPELINE --------
def build_rag_chain(astra_vector_store, llm, session_id):
    # Restrict retrieval boundaries strictly to chunks containing the corresponding unique session_id tag
    retriever = astra_vector_store.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"session_id": session_id}
        }
    )
    
    contextualize_q_prompt = contextualize_query()
    question_q_prompt = question_query()
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, question_q_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    
    trimmer = history_trimmer_func()

    trim_history_step = RunnablePassthrough.assign(
        chat_history=lambda x: trimmer.invoke(x["chat_history"])
    )

    trimmed_retrieval_chain = trim_history_step | retrieval_chain   

    return RunnableWithMessageHistory(
        trimmed_retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# ---------- STREAMLIT USER INTERFACE ----------
st.set_page_config(page_title="AstraDB Multi-Doc RAG Hub", page_icon="🗃️", layout="centered")

st.title("🗃️ AstraDB Multi-Doc RAG Hub")
st.caption("Configure your workspace credentials, load custom Hugging Face LLMs, and index files securely.")

# Initialize a runtime session token identifier for sandboxed memory allocation boundaries
if "unique_session_id" not in st.session_state:
    st.session_state.unique_session_id = str(uuid.uuid4())

current_session_id = st.session_state.unique_session_id

# Register fallback server hooks to clean up memory structures if the process stops completely
atexit.register(cleanup_session_storage, session_id=current_session_id)

# --- SIDEBAR CONFIGURATION PANEL ---
with st.sidebar:
    st.header("⚙️ Workspace Configuration")
    
    input_astra_token = st.text_input(
        "AstraDB Application Token", 
        value=os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""), 
        type="password",
        help="Starts with AstraCS:..."
    )
    input_astra_endpoint = st.text_input(
        "AstraDB API Endpoint", 
        value=os.getenv("ASTRA_DB_ENDPOINT", ""),
        help="The full database URL endpoint"
    )
    input_hf_token = st.text_input(
        "HuggingFace API Token", 
        value=os.getenv("HF_TOKEN", ""), 
        type="password",
        help="Your hf_... user access token"
    )
    
    input_model_name = st.text_input(
        "HuggingFace Model Repo ID", 
        value="deepseek-ai/DeepSeek-V4-Flash",
        help="Examples: meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it"
    )
    
    st.markdown("---")
    st.header("📂 Document Control Panel")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    # Process custom dynamic upload files inside our isolated runtime context
    if uploaded_files:
        if st.button("🔄 Sync & Index Uploaded PDFs"):
            if input_astra_token and input_astra_endpoint and input_hf_token:
                # Re-verify initialization state configuration constraints on demand
                if (
                    "astra_vector_store" not in st.session_state 
                    or st.session_state.get("current_hf_token") != input_hf_token
                    or st.session_state.get("current_astra_token") != input_astra_token
                ):
                    llm, astra_vector_store = initialize_core_system(
                        input_astra_token, input_astra_endpoint, input_hf_token, input_model_name
                    )
                    st.session_state.llm = llm
                    st.session_state.astra_vector_store = astra_vector_store
                    st.session_state.current_hf_token = input_hf_token
                    st.session_state.current_astra_token = input_astra_token
                    st.session_state.current_model = input_model_name
                
                with st.spinner("Processing documents for your isolated session..."):
                    sync_stats = process_and_index_multiple_pdfs(
                        uploaded_files, st.session_state.astra_vector_store, current_session_id
                    )
                    st.success(f"Sync Finished! Stats: {sync_stats}")
            else:
                st.error("Please provide all credential tokens in the configuration inputs before indexing.")
    else:
        st.info("Drop one or multiple PDF documents above to begin indexing.")
        
    st.markdown("---")
    # Triggers explicit file wipe along with standard pipeline state resetting
    if st.button("🗑️ Clear Chat History"):
        cleanup_session_storage(current_session_id)
        st.session_state.visible_messages = []
        st.session_state.chat_history_store = {}
        st.rerun()

# --- BACKEND INITIALIZATION & APPLICATION STREAM RUNTIME ---
if input_astra_token and input_astra_endpoint and input_hf_token:
    
    # Check if we need to initialize or change active settings strictly for this isolated session context
    if (
        "llm" not in st.session_state 
        or st.session_state.get("current_model") != input_model_name
        or st.session_state.get("current_hf_token") != input_hf_token
        or st.session_state.get("current_astra_token") != input_astra_token
    ):
        llm, astra_vector_store = initialize_core_system(
            input_astra_token, input_astra_endpoint, input_hf_token, input_model_name
        )
        st.session_state.llm = llm
        st.session_state.astra_vector_store = astra_vector_store
        st.session_state.current_model = input_model_name
        st.session_state.current_hf_token = input_hf_token
        st.session_state.current_astra_token = input_astra_token

    # Reference isolated resources out of active session state context blocks
    active_llm = st.session_state.llm
    active_store = st.session_state.astra_vector_store
    
    # Establish processing pipeline mappings passing the active context identifier
    conversational_rag_chain = build_rag_chain(active_store, active_llm, current_session_id)

    # UI visible chat stack state array checks
    if "visible_messages" not in st.session_state:
        st.session_state.visible_messages = []

    for msg in st.session_state.visible_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
        
        st.session_state.visible_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching isolated document chunks..."):
                config = {"configurable": {"session_id": f"session_{current_session_id}"}}
                response = conversational_rag_chain.invoke(
                    {"input": user_prompt},
                    config=config
                )
                answer = response["answer"]
                st.markdown(answer)
                
        st.session_state.visible_messages.append({"role": "assistant", "content": answer})

else:
    st.info("👋 Welcome! Please enter your AstraDB credentials and Hugging Face Token in the sidebar panel to launch your secure chat environment.")