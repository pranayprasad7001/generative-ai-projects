import os
import streamlit as st 
from langchain_groq import ChatGroq
from groq import RateLimitError, APIError
from langchain_core.tools import tool
from langchain_exa import ExaSearchResults, ExaFindSimilarResults
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

os.environ["LANGSMITH_TRACING_V2"] = st.secrets["LANGSMITH_TRACING_V2"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

def make_exa_tools(exa_api_key: str):
    """Build Exa-backed tools bound to one request's own API key.

    Deliberately NOT module-level globals: Streamlit Community Cloud runs one
    shared Python process for every visitor, so a global would get overwritten
    by whichever session's request ran most recently - meaning one user's
    search could silently execute with a different user's Exa key. Building
    fresh closures per-request keeps each session's key isolated.
    """
    exa_search = ExaSearchResults(exa_api_key=exa_api_key)
    exa_find_similar = ExaFindSimilarResults(exa_api_key=exa_api_key)

    @tool
    def exa_search_tool(query: str):
        """Search the web for information on a general topic."""
        try:
            return exa_search.invoke({
                "query": query,
                "num_results": 2,
                "highlights": True,
                "text_contents_options": {"max_characters": 1500}
            })
        except Exception as e:
            return f"Exa search failed ({type(e).__name__}): {e}"

    @tool
    def exa_find_similar_tool(url: str):
        """Find webpages similar to a given URL."""
        try:
            return exa_find_similar.invoke({
                "url": url,
                "num_results": 2,
                "highlights": True,
                "text_contents_options": {"max_characters": 1500}
            })
        except Exception as e:
            return f"Exa find-similar failed ({type(e).__name__}): {e}"

    return exa_search_tool, exa_find_similar_tool

# ----- Streamlit App -----

st.set_page_config(page_title='Agentic Search Engine', 
                   page_icon='🧠',
                   layout='wide',
                   initial_sidebar_state='expanded')

# Keep layout="wide" (so the sidebar behaves normally) but cap and center the
# main content column, so the title and chat sit in the middle of the page
# instead of stretching edge-to-edge.
st.markdown("""
<style>
.block-container {
    max-width: 800px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

st.title("Agentic Search Engine 🧠")
st.caption("An AI agent that can search the web and answer your questions")

st.sidebar.title("Settings")
api_key_groq = st.sidebar.text_input(label='Enter Your Groq API Key', type='password')
api_key_exa = st.sidebar.text_input(label='Enter Your Exa API Key', type='password')

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Hello! I'm your AI Assistant who can search the web. How can I help you today?",
    }]

# Render existing chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Check for API keys
if not api_key_groq or not api_key_exa:
    st.warning("Please enter your Groq API Key and Exa API Key in the sidebar to continue.")
    st.stop()
else:
    st.session_state.api_key_groq = api_key_groq
    st.session_state.api_key_exa = api_key_exa
    
    if prompt := st.chat_input("Ask a question..."):
        # Append user message to UI state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Initialize LLM
        llm = ChatGroq(
            api_key=st.session_state.api_key_groq,
            model_name="openai/gpt-oss-20b",
            temperature=0,
            reasoning_format="parsed",
            streaming=False,
            max_tokens=2048
        )
        
        exa_search_tool, exa_find_similar_tool = make_exa_tools(st.session_state.api_key_exa)
        tools = [exa_search_tool, exa_find_similar_tool]

        system_prompt = """You are a precise web search assistant. 

        CRITICAL LIMITS:
        1. You are allowed a MAXIMUM of 2 tool calls per user query. Do not perform repetitive or redundant searches.
        2. If the first search provides relevant information, STOP searching immediately and write your final answer.
        3. If you cannot find a perfect answer after 2 attempts, synthesize the best possible response using the information you gathered. Do not keep searching.
        4. Your internal reasoning is never shown to the user. After you finish reasoning, you MUST always write out the actual final answer as plain text in your response - never leave it blank just because you already worked it out while thinking."""

        # Setup Prompt Template for the Agent
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create a modern tool-calling agent
        agent = create_tool_calling_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            early_stopping_method="force",
            handle_parsing_errors=True,
        )

        # Convert Streamlit dictionary messages to LangChain message objects
        chat_history = []
        for msg in st.session_state.messages[:-1]: 
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # Run the agent and render the output
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            MAX_RETRIES = 2
            output = None
            last_error = None

            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": chat_history},
                        config={"callbacks": [st_cb]}
                    )
                    output = response["output"]
                    if output and output.strip():
                        break
                    output = None
                    last_error = "empty response"
                    continue
                except RateLimitError:
                    output = (
                        "Groq's rate limit was hit (the free tier is quite tight - "
                        "30 requests/min and limited tokens/min for this model). "
                        "Please wait a few seconds and try again."
                    )
                    break
                except APIError as e:
                    last_error = e
                    continue
                except Exception as e:
                    output = f"An error occurred: {type(e).__name__}: {e}"
                    break

            if output is None:
                output = (
                    f"The request failed after {MAX_RETRIES + 1} attempts "
                    f"({type(last_error).__name__ if isinstance(last_error, Exception) else last_error}). "
                    "Please try rephrasing your question, or try again in a moment."
                )
                
            # Append agent response to UI state
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.write(output)