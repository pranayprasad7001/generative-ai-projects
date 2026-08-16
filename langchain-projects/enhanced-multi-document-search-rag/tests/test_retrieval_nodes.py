import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from state.adaptive_state import AdaptiveRAGState
from nodes.retrieval_nodes import RetrievalNodes
from nodes.schema import ToolUse, QuestionRewrite
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage, AIMessage


class TestRetrievalNodes(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_llm_checker = MagicMock()
        self.mock_guardrails = MagicMock()

        self.nodes = RetrievalNodes(
            retriever=self.mock_retriever,
            llm_checker=self.mock_llm_checker,
            guardrails=self.mock_guardrails
        )

    async def test_query_analyzer(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=ToolUse(tool_type="hybrid_retrieval", analysis="Search docs"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?")
        updated_state = await self.nodes.query_analyzer(state)
        self.assertEqual(updated_state.tool_type, "hybrid_retrieval")
        self.assertEqual(updated_state.query_analysis, "Search docs")
        self.assertEqual(updated_state.analysis, "Search docs")

    async def test_hybrid_retrieval(self):
        docs = [Document(page_content="RAG is Retrieval-Augmented Generation")]
        self.mock_retriever.invoke.return_value = docs

        state = AdaptiveRAGState(question="What is RAG?")
        updated_state = await self.nodes.hybrid_retrieval(state)
        self.assertEqual(updated_state.retrieved_docs, docs)

    async def test_hybrid_retrieval_with_config_override(self):
        docs = [Document(page_content="Config retriever doc")]
        custom_retriever = MagicMock()
        custom_retriever.invoke.return_value = docs

        state = AdaptiveRAGState(question="What is RAG?")
        config = {"configurable": {"retriever": custom_retriever}}
        updated_state = await self.nodes.hybrid_retrieval(state, config=config)
        self.assertEqual(updated_state.retrieved_docs, docs)

    async def test_query_rewriter(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=QuestionRewrite(rewritten_question="What is retrieval augmented generation?", reasoning="Expanded acronym"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?")
        updated_state = await self.nodes.query_rewriter(state)
        self.assertEqual(updated_state.question, "What is retrieval augmented generation?")
        self.assertEqual(updated_state.rewrite_explanation, "Expanded acronym")
        self.assertEqual(updated_state.rewrite_reasoning, "Expanded acronym")
        self.assertEqual(updated_state.rewrite_count, 1)


    async def test_external_search_structured_citations(self):
        mock_agent = MagicMock()
        mock_tool_msg = ToolMessage(
            content=json.dumps([
                {
                    "title": "LangChain Documentation",
                    "url": "https://python.langchain.com/docs",
                    "source": "Tavily"
                }
            ]),
            name="tavily_search",
            tool_call_id="call_123"
        )
        mock_ai_msg = AIMessage(content="Here is info from LangChain docs.")
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_tool_msg, mock_ai_msg]})

        self.nodes.external_search_agent = mock_agent

        state = AdaptiveRAGState(question="Tell me about LangChain")
        updated_state = await self.nodes.external_search(state)
        
        self.assertEqual(len(updated_state.external_citations), 1)
        citation = updated_state.external_citations[0]
        self.assertEqual(citation["title"], "LangChain Documentation")
        self.assertEqual(citation["url"], "https://python.langchain.com/docs")
        self.assertEqual(citation["source"], "Tavily")
        self.assertEqual(citation["tool"], "tavily_search")
        self.assertTrue(citation["retrieval_timestamp"].endswith("+00:00") or "T" in citation["retrieval_timestamp"])


if __name__ == "__main__":
    unittest.main()
