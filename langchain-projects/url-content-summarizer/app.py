import os
import validators 
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api._errors import RequestBlocked, IpBlocked, TranscriptsDisabled, NoTranscriptFound

os.environ["LANGSMITH_TRACING_V2"] = st.secrets["LANGSMITH_TRACING_V2"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Model Initialization
def model_initialization(api_key, model, temp):
    llm = ChatGroq(groq_api_key=api_key, model_name=model, temperature=temp)
    return llm

# Streamlit App Configuration
st.set_page_config(page_title="URL Content Summarizer", layout='wide', page_icon="📑")
st.title("📑 YouTube & Webpage Summarizer")
st.subheader('Paste a link below to generate a concise summary')

# Get the Groq API Key and URL (YT or Website) to Summarize
st.sidebar.title("API Credentials")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type='password')
temperature = st.sidebar.slider("Temperature: ", min_value=0.1, max_value=1.0, value=0.2, step=0.1)

# Select the model to use for summarizing
model_choice = st.sidebar.selectbox(
    "Choose a Model", 
    ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"], 
    index=2
)

generic_url = st.text_input("Enter the URL to Summarize", label_visibility="collapsed")

if st.button("Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get Started!")
    elif not validators.url(generic_url):
        st.error("Please enter a Valid URL!")
    else:
        try:
            llm = model_initialization(groq_api_key, model_choice, temperature)
            with st.spinner("Fetching and Summarizing the Content..."):
                
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                        docs = loader.load()
                    except RequestBlocked:
                        st.error("YouTube is blocking transcript requests from this app's cloud server — a known limitation of free-tier hosting (Streamlit Cloud's IPs are blocked by YouTube). Try a webpage URL instead, or run this app locally to summarize YouTube videos.")
                        st.stop()
                    except (TranscriptsDisabled, NoTranscriptFound):
                        st.error("This video doesn't have a transcript available.")
                        st.stop()
                        
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        continue_on_failure=False, 
                        ssl_verify=True, 
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
                    )
                
                docs = loader.load()
                full_text = "".join([doc.page_content for doc in docs])
                estimated_tokens = len(full_text) / 4
                
                st.write(f"Estimated Tokens: {int(estimated_tokens)}")

                if estimated_tokens > 6000:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=300)
                    chunks = text_splitter.split_documents(docs)
                    
                    map_prompt_text = """
                    Write a concise summary of the following chunk of text:
                    Content: {text}
                    """
                    map_prompt_template = PromptTemplate(template=map_prompt_text, input_variables=["text"])

                    combine_prompt = """
                    Provide a Final Summary of the following contents in 400 words:
                    Content: {text}
                    """
                    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
                    
                    chain = load_summarize_chain(
                        llm=llm, 
                        chain_type="map_reduce", 
                        map_prompt=map_prompt_template, 
                        combine_prompt=combine_prompt_template
                    )
                else:     
                    chunks = docs
                    prompt_template = """
                    Provide a Summary of the following content in 300 words:
                    Content: {text}
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                    chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                
                result = chain.invoke({"input_documents": chunks})
                output_summary = result["output_text"]
                
                st.subheader("Summary:")
                st.markdown(output_summary)
            
        except Exception as e:
            error_msg = str(e)
            if "SSL" in error_msg or "CERTIFICATE_VERIFY_FAILED" in error_msg:
                st.error("This site's SSL certificate couldn't be verified. Try a different URL.")
            else:
                st.error(f"Error: {e}")