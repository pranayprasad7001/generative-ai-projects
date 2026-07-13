import re
import streamlit as st 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


from dotenv import load_dotenv
import os

# Load environment variables for LangSmith tracing
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, selected_model, temperature, max_tokens):
    # Initialize the Groq model using ChatGroq
    llm = ChatGroq(
        api_key=api_key, 
        model_name=selected_model, 
        temperature=temperature, 
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Invoke the chain
    answer = chain.invoke({'question': question})
    return answer

# Title of the App
st.title("Q&A Chatbot with Groq ⚡")

# Sidebar for Configuration
st.sidebar.title("Settings")

# Secure Input for Groq API Key
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Select the Groq Model (populated with current popular Groq-hosted models)
selected_model = st.sidebar.selectbox(
    "Select Open Source model", 
    [
        "qwen/qwen3-32b",
        "qwen/qwen3.6-27b",
        "llama-3.3-70b-versatile", 
        "llama-3.1-8b-instant", 
        "gemma2-9b-it",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
)

# Adjust Response Parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=4000, value=1500)

# Main Interface for User Input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        try:
            with st.spinner("Groq is generating..."):
                response = generate_response(user_input, api_key, selected_model, temperature, max_tokens)
                
                # Check if it's a reasoning model outputting a <think> tag
                if "<think>" in response:
                    # Scenario A: The model finished its thought properly
                    if "</think>" in response:
                        think_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
                        thoughts = think_match.group(1).strip()
                        final_answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                        
                        with st.expander("🧠 View AI Reasoning"):
                            st.write(thoughts)
                        
                        st.write(final_answer)
                    
                    # Scenario B: The model ran out of tokens mid-thought!
                    else:
                        thoughts = response.replace("<think>", "").strip()
                        
                        with st.expander("🧠 View AI Reasoning (Incomplete)"):
                            st.write(thoughts)
                            
                        st.warning("⚠️ The AI ran out of tokens before it could finish thinking. Please increase the **Max Tokens** slider in the sidebar.")
                        
                else:
                    # Standard display for non-reasoning models (like Llama 3.3)
                    st.write(response)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter your Groq API Key in the sidebar to continue.")