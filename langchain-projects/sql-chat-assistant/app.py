import os
import sqlite3
import streamlit as st 
from pathlib import Path
from urllib.parse import quote_plus
from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# Setting Up the Page Configs
st.set_page_config(page_title="SQL Agent", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# Warning Message
INJECTION_WARNING = """‼️ SQL AGENT CAN BE VULNERABLE TO PROMPT INJECTION. USE A DB ROLE WITH LIMITED PRIVILEGES ‼️"""
st.warning(INJECTION_WARNING)

# LocalDB or MySQL DB
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Radio Options for DB selection
radio_opt = ["Use SQLite 3 DB: Student.db", "Connect to your SQL Database"]

# Radio button selection
selected_opt = st.sidebar.radio(label="Choose the DB with which you want to chat", options=radio_opt) 

# If MySQL DB selected, get the DB info else use local db 
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_pass = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database", value="test")
else:
    db_uri = LOCALDB

# Groq API Key
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Temperature
temp = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# DB URI validation
if not db_uri:
    st.info("Please Provide DB Info and URI")
    st.stop()
    
# Groq API Key validation
if not groq_api_key:
    st.info("Please Provide Groq API Key")
    st.stop()

# Model selection
model_selected = st.sidebar.selectbox("Choose Model", ["qwen/qwen3-32b", "qwen/qwen3.6-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "openai/gpt-oss-20b"])  

# Model Initialization
try:
    llm = ChatGroq(model=model_selected, api_key=groq_api_key, temperature=temp, streaming=True)
except Exception as e:
    st.error(f"Failed to initialize the LLM: {str(e)}")
    st.stop()

# DB Configuration 
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_pass=None, mysql_db=None):
    try:
        if db_uri == LOCALDB:
            db_file_path = (Path(__file__).parent / "student.db").absolute()
            creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        elif db_uri == MYSQL:
            if not (mysql_host and mysql_user and mysql_pass and mysql_db):
                st.error("Please Provide All MySQL DB Info")
                st.stop()
            
            # CRITICAL: Encode the password to safely handle special characters like @, #, /
            encoded_pass = quote_plus(mysql_pass)
            engine_uri = f"mysql+mysqlconnector://{mysql_user}:{encoded_pass}@{mysql_host}/{mysql_db}"
            return SQLDatabase(create_engine(engine_uri))
    except Exception as e:
        st.error(f"Failed to connect to the database: {str(e)}")
        st.stop()

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_pass, mysql_db)
else:
    db = configure_db(db_uri)

# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True # Added to help ReAct agents recover if the LLM output format is slightly off
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am SQL Agent. How can I help you with your database queries today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask a question about your database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback_handler = StreamlitCallbackHandler(st.container())
        
        try:
            response = agent.invoke(
                {"input": user_query}, 
                {"callbacks": [streamlit_callback_handler]}
            )
            
            # Extract the string output from the resulting dictionary
            final_output = response.get("output", "I could not generate an answer.")
            
            st.session_state.messages.append({"role": "assistant", "content": final_output})
            st.write(final_output)
            
        except Exception as e:
            st.error(f"An error occurred while running the agent: {str(e)}")