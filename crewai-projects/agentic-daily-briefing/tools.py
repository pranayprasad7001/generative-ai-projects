import os
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tavily_tool = TavilySearchTool(max_results=10, search_depth="advanced")