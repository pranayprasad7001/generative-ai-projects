import unittest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from state.adaptive_state import AdaptiveRAGState
from nodes.evaluation_nodes import EvaluationNodes
from nodes.schema import RetrievalGrade, HallucinationGrade, AnswerRelevanceGrade
from langchain_classic.schema import Document


class TestEvaluationNodes(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm_checker = MagicMock()
        self.nodes = EvaluationNodes(llm_checker=self.mock_llm_checker)

    async def test_documents_grader_rich_pass(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=RetrievalGrade(score=0.88, decision="pass", reasoning="Highly relevant"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?", retrieved_docs=[Document(page_content="RAG docs")])
        updated_state = await self.nodes.documents_grader(state)
        self.assertEqual(updated_state.retrieval_grade, "yes")
        self.assertEqual(updated_state.retrieval_score, 0.88)
        self.assertEqual(updated_state.retrieval_reasoning, "Highly relevant")
        self.assertEqual(updated_state.analysis, "Highly relevant")

    async def test_documents_grader_rich_fail(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=RetrievalGrade(score=0.35, decision="rewrite", reasoning="Irrelevant docs"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?", retrieved_docs=[Document(page_content="Irrelevant")])
        updated_state = await self.nodes.documents_grader(state)
        self.assertEqual(updated_state.retrieval_grade, "no")
        self.assertEqual(updated_state.retrieval_score, 0.35)
        self.assertEqual(updated_state.retrieval_reasoning, "Irrelevant docs")

    async def test_hallucination_detector_rich_grounded(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=HallucinationGrade(score=0.95, decision="pass", reasoning="Fully grounded"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?", answer="RAG is...", retrieved_docs=[Document(page_content="RAG docs")])
        updated_state = await self.nodes.hallucination_detector(state)
        self.assertEqual(updated_state.hallucination_grade, "yes")
        self.assertEqual(updated_state.hallucination_score, 0.95)
        self.assertEqual(updated_state.grounding_reasoning, "Fully grounded")

    async def test_hallucination_detector_rich_hallucinated(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=HallucinationGrade(score=0.40, decision="retry", reasoning="Hallucinated stats"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?", answer="Fake claims", retrieved_docs=[Document(page_content="RAG docs")])
        updated_state = await self.nodes.hallucination_detector(state)
        self.assertEqual(updated_state.hallucination_grade, "no")
        self.assertEqual(updated_state.hallucination_score, 0.40)
        self.assertEqual(updated_state.grounding_reasoning, "Hallucinated stats")

    async def test_answer_relevance_grader_rich(self):
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=AnswerRelevanceGrade(score=0.92, decision="pass", reasoning="Directly relevant"))
        self.mock_llm_checker.with_structured_output.return_value = mock_structured

        state = AdaptiveRAGState(question="What is RAG?", answer="RAG is...")
        updated_state = await self.nodes.answer_relevance_grader(state)
        self.assertEqual(updated_state.answer_relevance_grade, "yes")
        self.assertEqual(updated_state.answer_relevance_score, 0.92)
        self.assertEqual(updated_state.relevance_reasoning, "Directly relevant")


if __name__ == "__main__":
    unittest.main()
