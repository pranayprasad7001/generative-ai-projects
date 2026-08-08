from langchain_mcp_adapters.client import MultiServerMCPClient
from config.config import Config


class MCPToolManager:

    def __init__(self):
        self.client = MultiServerMCPClient(
            {
                "tavily": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "mcp-remote",
                        f"https://mcp.tavily.com/mcp/?tavilyApiKey={Config.TAVILY_API_KEY}",
                    ],
                    "transport": "stdio",
                },
                "wikipedia": {
                    "command": "uvx",
                    "args": [
                        "--from",
                        "wikipedia-mcp-server@latest",
                        "wikipedia-mcp",
                    ],
                    "transport": "stdio",
                },
                "arxiv": {
                    "command": "uvx",
                    "args": [
                        "arxiv-mcp-server",
                    ],
                    "transport": "stdio",
                },
            }
        )

    async def get_tools(self):
        return await self.client.get_tools()