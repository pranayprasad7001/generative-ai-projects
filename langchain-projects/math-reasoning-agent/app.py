import os
import time
import streamlit as st 
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import Tool, AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI   

os.environ["LANGSMITH_TRACING_V2"] = st.secrets["LANGSMITH_TRACING_V2"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Streamlit Page Configuration
st.set_page_config(
    page_title="Math & Reasoning Agent", 
    page_icon="🔢", 
    layout="wide"
)

st.title("Math & Reasoning Agent")
st.caption("A ReAct agent using Google Gemini, Wikipedia search, and a calculator tool.")

st.sidebar.title(" ⚙️ Settings ")

def initialize_model(google_models, google_api_key):
    return ChatGoogleGenerativeAI(model=google_models, api_key=google_api_key)

google_models = st.sidebar.selectbox("Select a Model", ["gemini-3.5-flash", "gemini-3.1-flash-lite"], index=1)
google_api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

if not google_api_key:
    st.error("Please enter your Google API Key to start.")
    st.stop()
else:
    model = initialize_model(google_models, google_api_key)

# Tools Setup
wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="wikipedia",
    func=wiki_wrapper.run,
    description="A tool for searching the internet to find information on various topics."
)

math_chain = LLMMathChain.from_llm(llm=model, verbose=True)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only the mathematical expression input needs to be provided."
)

# Reasoning tool setup using standard classic LLMChain
reasoning_prompt = PromptTemplate.from_template(
    "Logically arrive at the solution, provide a detailed explanation point-wise.\nQuestion: {question}\nAnswer:"
)
llm_math_chain = LLMChain(llm=model, prompt=reasoning_prompt)
 
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=llm_math_chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

tools = [calculator, wiki_tool, reasoning_tool]

# Construct Modern Classic ReAct Prompt & Agent
react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_prompt_template)

# Build the ReAct agent graph and wrap it in the executor loop
agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
assistant_agent = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# Chat History UI Management
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am a Math Chatbot. Ask me any math-related question."}
    ]

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state["messages"] = []
    st.session_state["messages"].append( {"role": "assistant", "content": "Hello! I am a Math Chatbot. Ask me any math-related question."})
    st.rerun()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# Interaction Logic
if question:= st.chat_input("Enter your Question:"):

    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
            
        # Resilient invoke loop with a retry mechanism for transient API/network errors
        max_retries = 3
        response_output = None
            
        for attempt in range(max_retries):
            try:
                response = assistant_agent.invoke({"input": question})
                response_output = response["output"]
                break  # Success! Exit loop
            except Exception as e:
                error_str = str(e)
                # Handle typical structural errors or rate limits gracefully
                if "503" in error_str or "RateLimit" in error_str or "QuotaExceeded" in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Short delay before retrying
                        continue
                    else:
                        st.error("🚨 The upstream Google Gemini API is experiencing high traffic or rate limits. Please check your quota or try again shortly.")
                else:
                    st.error(f"An unexpected runtime error occurred: {e}")
                    break
            
        # Display results if the agent successfully parsed a response
        if response_output:
                st.session_state.messages.append({"role": "assistant", "content": response_output})
                st.chat_message("assistant").write(response_output)

                intermediate_steps = response.get("intermediate_steps", [])
                if intermediate_steps:
                    with st.expander("🔎 Show reasoning steps"):
                        for i, (action, observation) in enumerate(intermediate_steps, start=1):
                            st.markdown(f"**Step {i}: {action.tool}**")
                            st.markdown(f"- Input: `{action.tool_input}`")
                            st.markdown(f"- Observation: {observation}")
                            if action.log:
                                st.caption(action.log.strip())