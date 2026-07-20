from litellm import max_tokens
from crewai import Agent, LLM
from tools import tavily_tool
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = "NA"
os.environ['OPENAI_BASE_URL'] = "NA"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize the LLM
llm_1 = LLM(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.environ["NVIDIA_API_BASE"],
    temperature=0.2,
    max_tokens=4096
)

# Initialize the LLM
llm_2 = LLM(
    model="qwen/qwen3-next-80b-a3b-instruct",
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.environ["NVIDIA_API_BASE"],
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    stream=False
)


hf_embedder = dict(
    provider="huggingface",
    config=dict(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# Global News Researcher Agent
news_researcher = Agent(
    role='Lead Global Intelligence & News Analyst',
    goal=(
        "Scour the web using Tavily to find, verify, and compile the most recent 2026 trending news "
        "for the following 5 categories: Breaking News, India, World, AI & Tech, and Science. "
        "Strictly limit the output to a maximum of 10 highly relevant news items per category."
    ),
    verbose=True,
    memory=False,
    backstory=(
        "You are a seasoned international journalist and data intelligence analyst. You specialize "
        "in filtering through web noise to extract highly credible, impactful, and real-time news updates. "
        "You understand regional Indian contexts as well as major global, scientific, and technological shifts. "
        "Your job is to curate facts, ensure cross-verification, and present a concise digest free of fluff."
        "CRITICAL INSTRUCTION FOR TOOL USAGE: "
        "When using a tool, you MUST provide the Action Input as a raw, valid JSON dictionary. "
        "Do NOT wrap the dictionary in markdown code blocks (e.g., avoid ```json). "
        "Do NOT include strings or arrays. "
        "Example of Correct Action Input: {'search_query': 'Latest AI tech news 2026'}"
    ),
    tools=[tavily_tool],
    allow_delegation=False,
    llm=llm_1,
    max_rpm=20,
    max_tokens=4096,
    embedder=hf_embedder
)

# Senior News Editor / Writer Agent
news_writer = Agent(
    role="Chief News Editor & Briefing Author",
    goal=(
        "Transform raw verified news data into a comprehensive, highly readable daily briefing "
        "categorized perfectly with up to 10 bulleted stories per section."
    ),
    verbose=True,
    memory=False,
    backstory=(
        "You are an elite editorial chief known for publishing objective, engaging, and fast-paced "
        "news briefings. You have an exceptional eye for structure, ensuring readers can grasp "
        "complex geopolitical shifts, tech breakthroughs, and breaking headlines at a single glance. "
        "You maintain a strictly informative, neutral, and authoritative journalistic tone."
    ),
    tools=[tavily_tool], 
    allow_delegation=False,
    llm=llm_2,
    max_rpm=20,
    max_tokens=4096,
    embedder=hf_embedder
)