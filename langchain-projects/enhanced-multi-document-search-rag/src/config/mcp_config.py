import shutil
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.llmgateway_config import Config

logger = logging.getLogger(__name__)


class MCPToolManager:

    def __init__(self):
        server_configs = {}

        # 1. Tavily MCP (requires npx)
        if shutil.which("npx") and Config.TAVILY_API_KEY:
            server_configs["tavily"] = {
                "command": "npx",
                "args": [
                    "-y",
                    "mcp-remote",
                    f"https://mcp.tavily.com/mcp/?tavilyApiKey={Config.TAVILY_API_KEY}",
                ],
                "transport": "stdio",
            }
        else:
            logger.info("npx not found or TAVILY_API_KEY missing; skipping Tavily MCP subprocess.")

        # 2. Wikipedia & Arxiv MCP (requires uvx)
        if shutil.which("uvx"):
            server_configs["wikipedia"] = {
                "command": "uvx",
                "args": [
                    "--from",
                    "wikipedia-mcp-server@latest",
                    "wikipedia-mcp",
                ],
                "transport": "stdio",
            }
            server_configs["arxiv"] = {
                "command": "uvx",
                "args": [
                    "arxiv-mcp-server",
                ],
                "transport": "stdio",
            }
        else:
            logger.info("uvx not found in PATH; skipping Wikipedia & ArXiv MCP subprocesses.")

        self.server_configs = server_configs
        self.client = MultiServerMCPClient(server_configs) if server_configs else None

    async def get_tools(self) -> list:
        """Retrieve MCP tools or fall back to native LangChain tools."""
        tools = []
        if self.client and self.server_configs:
            try:
                tools = await self.client.get_tools()
                logger.info("Successfully loaded %d MCP tools.", len(tools))
            except Exception as e:
                logger.warning("Failed to initialize MCP tools from servers: %s. Using native fallbacks.", e)
                tools = []

        # If MCP returned no tools, provide native LangChain tool fallbacks
        if not tools:
            tools = self._get_native_fallback_tools()

        return tools

    def _get_native_fallback_tools(self) -> list:
        """Initialize native LangChain search tools as fallback."""
        fallback_tools = []
        try:
            try:
                from langchain_tavily import TavilySearch
                if Config.TAVILY_API_KEY:
                    fallback_tools.append(TavilySearch(max_results=5, tavily_api_key=Config.TAVILY_API_KEY))
                    logger.info("Initialized native TavilySearch fallback tool.")
            except ImportError:
                from langchain_community.tools.tavily_search import TavilySearchResults
                if Config.TAVILY_API_KEY:
                    fallback_tools.append(TavilySearchResults(max_results=5, tavily_api_key=Config.TAVILY_API_KEY))
                    logger.info("Initialized native TavilySearchResults fallback tool.")
        except Exception as e:
            logger.debug("Native Tavily fallback tool unavailable: %s", e)

        try:
            from langchain_community.utilities import WikipediaAPIWrapper
            from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
            fallback_tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
            logger.info("Initialized native WikipediaQueryRun fallback tool.")
        except Exception as e:
            logger.debug("Native Wikipedia fallback tool unavailable: %s", e)

        return fallback_tools