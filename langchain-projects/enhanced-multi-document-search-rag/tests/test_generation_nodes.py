import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from state.adaptive_state import AdaptiveRAGState
from nodes.generation_nodes import GenerationNodes
from langchain_classic.schema import Document
from langchain_core.messages import AIMessage


class TestGenerationNodes(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm_generator = MagicMock()
        self.nodes = GenerationNodes(llm_generator=self.mock_llm_generator)

    async def test_answer_generator_standard(self):
        mock_invoker = MagicMock()
        mock_invoker.ainvoke = AsyncMock(return_value=AIMessage(content="Generated answer."))
        self.mock_llm_generator.bind.return_value = mock_invoker

        state = AdaptiveRAGState(question="What is RAG?", retrieved_docs=[Document(page_content="Context text")])
        updated_state = await self.nodes.answer_generator(state)
        self.assertEqual(updated_state.answer, "Generated answer.")
        self.assertEqual(updated_state.generate_count, 1)

    async def test_answer_generator_self_correction_regeneration(self):
        mock_invoker = MagicMock()
        mock_invoker.ainvoke = AsyncMock(return_value=AIMessage(content="Corrected grounded answer."))
        self.mock_llm_generator.bind.return_value = mock_invoker

        state = AdaptiveRAGState(
            question="What is RAG?",
            answer="Bad hallucinated draft",
            generate_count=1,
            hallucination_grade="no",
            analysis="Hallucinated claims found",
            retrieved_docs=[Document(page_content="Context text")]
        )
        updated_state = await self.nodes.answer_generator(state)
        self.assertEqual(updated_state.answer, "Corrected grounded answer.")
        self.assertEqual(updated_state.generate_count, 2)


if __name__ == "__main__":
    unittest.main()
