import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config.mcp_config import MCPToolManager


class TestMCPToolManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for MCPToolManager and graceful tool fallbacks."""

    def test_init_without_npx_uvx(self):
        with patch("shutil.which", return_value=None):
            manager = MCPToolManager()
            self.assertEqual(manager.server_configs, {})
            self.assertIsNone(manager.client)

    async def test_get_tools_fallback_when_client_empty(self):
        with patch("shutil.which", return_value=None):
            manager = MCPToolManager()
            tools = await manager.get_tools()
            self.assertIsInstance(tools, list)

    async def test_get_tools_handles_mcp_server_exception(self):
        with patch("shutil.which", return_value="/usr/bin/npx"):
            manager = MCPToolManager()
            manager.client = AsyncMock()
            manager.client.get_tools.side_effect = ConnectionRefusedError("MCP Server Offline")
            
            tools = await manager.get_tools()
            self.assertIsInstance(tools, list)


if __name__ == "__main__":
    unittest.main()
