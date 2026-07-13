import os
import shutil
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Page Configuration
st.set_page_config(page_title="Research Paper Assistant", page_icon="📄")
st.markdown("## 📄 Interactive Research Paper Assistant")

# Sidebar: Configuration & Document Management
with st.sidebar:
    st.header("Configuration")
    
    default_groq_key = os.getenv("GROQ_API_KEY", "")
    user_groq_api_key = st.text_input(
        "Enter your Groq API Key", 
        value=default_groq_key, 
        type="password",
        help="Get a free key from console.groq.com."
    )
    
    st.divider()
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload PDF Research Papers", type="pdf", accept_multiple_files=True)

# Initialize Models
@st.cache_resource
def get_models(groq_key):
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)
    # CPU-friendly local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

if user_groq_api_key:
    llm, embeddings = get_models(user_groq_api_key)
else:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to start.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# System Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert research assistant. Your primary objective is to answer the user's question using the provided context.

    Follow these strict instructions:
    1. Analyze the provided <context> to see if it contains the answer to the user's question.
    2. If the answer is found in the <context>, provide a detailed response based ONLY on that information.
    3. If the answer is NOT found in the <context>, you must structure your response exactly like this:
       - First, explicitly state: "The provided document does not contain information to answer this query."
       - Second, provide an answer based on your general knowledge.
       - Third, conclude with this exact warning: "**⚠️ WARNING:** The above information is drawn from general model knowledge, not the uploaded research paper. Please verify these claims with authoritative sources."

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Document Embedding Logic (Memory-Safe Batching)
if uploaded_files:
    with st.sidebar:
        if st.button("Process & Embed Documents"):
            with st.spinner("Processing documents..."):
                docs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name
                    
                    loader = PyPDFLoader(temp_path)
                    docs.extend(loader.load())
                    os.remove(temp_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)

                # Clean up any previous session's vector store before creating a new one
                if "chroma_dir" in st.session_state and os.path.exists(st.session_state.chroma_dir):
                    shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)

                # SECURITY/ISOLATION FIX: use a fresh temp directory per session instead of a
                # shared "./chroma_db" path. Streamlit Community Cloud runs one shared process
                # for every visitor, so a fixed shared path would let one user's uploaded papers
                # (and the answers derived from them) leak into another user's session.
                persist_directory = tempfile.mkdtemp()
                st.session_state.chroma_dir = persist_directory

                # Batch processing to prevent Out-Of-Memory (OOM) crashes on 1GB cloud containers
                batch_size = 100
                vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
                
                progress_bar = st.progress(0, text="Embedding batches into local vector store...")
                
                for i in range(0, len(final_documents), batch_size):
                    batch = final_documents[i : i + batch_size]
                    vectorstore.add_documents(batch)
                    
                    current_progress = min((i + batch_size) / len(final_documents), 1.0)
                    progress_bar.progress(current_progress, text=f"Embedded {min(i + batch_size, len(final_documents))}/{len(final_documents)} chunks...")

                st.success(f"Successfully embedded all {len(final_documents)} chunks!")

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about the uploaded papers..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    if "chroma_dir" in st.session_state and os.path.exists(st.session_state.chroma_dir):
        vectorstore = Chroma(persist_directory=st.session_state.chroma_dir, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating response..."):
                start_time = time.time()
                response = retrieval_chain.invoke({'input': user_input})
                latency = time.time() - start_time
                
                answer = response['answer']
                st.markdown(answer)
                st.caption(f"Response generated in {latency:.2f} seconds.")
                
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response['context']):
                        source = doc.metadata.get('source', 'Unknown Source')
                        st.markdown(f"**Chunk {i+1}** *(Source: {os.path.basename(source)})*")
                        st.write(doc.page_content)
                        st.divider()
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Vector database is empty. Please upload and process a document first.")