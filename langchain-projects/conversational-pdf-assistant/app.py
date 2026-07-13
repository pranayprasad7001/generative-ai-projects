# IMPORTANT STREAMLIT CLOUD SQLITE PATCH 
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------------------

import os
import shutil
import tempfile
import streamlit as st
from operator import itemgetter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING_V2"] = st.secrets["LANGSMITH_TRACING_V2"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# ------------------ Resource & Document Handlers ------------------

@st.cache_resource
def get_models(api_key, model, temperature, max_tokens):
    """Cache the heavy static resources: LLM and Embeddings."""
    llm = ChatGroq(api_key=api_key, model_name=model, temperature=temperature, max_tokens=max_tokens) 
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return llm, embeddings

def load_resources(api_key, model, temperature, max_tokens, documents):
    """Load resources and build a disk-backed vector store to save RAM on Streamlit Cloud."""
    llm, embeddings = get_models(api_key, model, temperature, max_tokens)
    
    splitted_documents = preprocess_document(documents)
    
    # FIX: Offload Chroma DB to a temporary disk directory to prevent RAM OOM crashes
    persist_dir = tempfile.mkdtemp()
    chroma_db = Chroma.from_documents(
        documents=splitted_documents, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # Save the directory path to session state so we can clean it up later if needed
    st.session_state.chroma_dir = persist_dir 
    
    retriever = chroma_db.as_retriever()
    return llm, retriever

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get Session History or Create New One."""
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def preprocess_document(file_paths):
    """Preprocess a list of uploaded PDF document paths.""" 
    all_documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        all_documents.extend(loader.load())
        
    # keep chunks small for 512 token limit of the embedding model
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    splitted_documents = text_splitter.split_documents(all_documents)
    return splitted_documents

def load_uploaded_documents(uploaded_files):
    """Safely load uploaded PDF documents using temporary files."""
    file_paths = []
    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        file_paths.append(temp_file.name)
    return file_paths

# LangChain Construct Builders

def contextualize_retriever(llm, retriever):
    """Create a retriever that can answer questions with chat history."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, rephrase the follow up question "
        "to be a standalone question. Do Not Use The Chat History To Answer The Question, "
        "Only Use It To Provide Context To Rephrase The Question if Needed Otherwise Return It as It Is."
    )
                
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)    

def create_qa_chain(llm):
    """Create a question answering chain."""
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise. \n\n"
        "<context>\n{context}\n</context>"
    )
                
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)

# Streamlit App UI & Execution Flow 

st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDFs and Chat with their Content")

# Initialize session states for pipeline readiness
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "store" not in st.session_state:
    st.session_state.store = {}

# Sidebar Configuration
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
groq_model = st.sidebar.selectbox("Select a model:", [
    "llama-3.3-70b-versatile", 
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b", 
    "openai/gpt-oss-20b"
])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=0, max_value=4096, value=1200)
session_id = st.sidebar.text_input("Session ID", value="default_session")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)  

# Document Processing Block
if st.button("Upload & Initialize"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key")
        st.stop()
        
    if not uploaded_files:
        st.error("Please upload a PDF file")
        st.stop()

    file_paths = []
    with st.spinner("Loading resources and building RAG core..."):
        try:
            # Clear previous chains and old Chroma disk instances to avoid storage bloat
            if "rag_chain" in st.session_state:
                del st.session_state.rag_chain
            if "chroma_dir" in st.session_state and os.path.exists(st.session_state.chroma_dir):
                shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)

            file_paths = load_uploaded_documents(uploaded_files)   
            
            llm, retriever = load_resources(groq_api_key, groq_model, temperature, max_tokens, file_paths)
            
            contextualize_q_chain = contextualize_retriever(llm, retriever)
            question_answer_chain = create_qa_chain(llm)
            base_rag_chain = create_retrieval_chain(contextualize_q_chain, question_answer_chain)

            trimmer = trim_messages(
                max_tokens=700,
                strategy="last",
                token_counter=llm,
                include_system=True,
                allow_partial=False,
                start_on="human"
            )

            rag_chain_with_trimming = (
                RunnablePassthrough.assign(chat_history=itemgetter("chat_history") | trimmer)
                | base_rag_chain
            )

            rag_chain_with_history = RunnableWithMessageHistory(
                rag_chain_with_trimming,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            st.session_state.rag_chain = rag_chain_with_history
            st.session_state.pipeline_ready = True
            
            st.success("Resources loaded successfully! You can now start chatting.")
            
        except Exception as e:
            st.error(f"An error occurred during initialization: {str(e)}")
            st.session_state.pipeline_ready = False
            
        finally:
            # temp file cleanup
            for path in file_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

# Chat Interface Block
if st.session_state.pipeline_ready:
    st.divider()
    user_input = st.text_input("Your Question: ", type="default")

    if user_input:
        session_history = get_session_history(session_id)
        with st.spinner("Analyzing context..."):
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input}, 
                    config={"configurable": {"session_id": session_id}}
                )
                
                st.info(f"**Assistant:** {response['answer']}") 
                
                with st.expander("View Chat History"):
                    for msg in session_history.messages:
                        role = "User" if msg.type == "human" else "Assistant"
                        st.write(f"**{role}:** {msg.content}")
            except Exception as e:
                st.error(f"Failed to process query: {str(e)}")