import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from state.adaptive_state import AdaptiveRAGState
from nodes.schema import ToolUse, RetrievalGrade, QuestionRewrite, HallucinationGrade, AnswerRelevanceGrade
from langchain_classic.schema import Document
from langchain_core.messages import AIMessage, ToolMessage

class TestAdaptiveRAGNodes(unittest.IsolatedAsyncioTestCase):

    @patch("nodes.adaptive_node.Guardrails")
    def setUp(self, mock_guardrails_class):
        self.mock_guardrails = MagicMock()
        mock_guardrails_class.return_value = self.mock_guardrails
        
        self.mock_input_agent = AsyncMock()
        self.mock_output_agent = AsyncMock()
        self.mock_guardrails.get_input_guardrail_agent.return_value = self.mock_input_agent
        self.mock_guardrails.get_output_guardrail_agent.return_value = self.mock_output_agent

        self.mock_retriever = MagicMock()
        self.mock_llm = MagicMock()
        
        # Instantiate AdaptiveRAGNodes
        from nodes.adaptive_node import AdaptiveRAGNodes
        self.nodes = AdaptiveRAGNodes(self.mock_retriever, self.mock_llm)

    async def test_input_query_security_check_pass(self):
        # Setup mock return from input guardrail agent
        mock_msg = AIMessage(content="PASSED")
        self.mock_input_agent.ainvoke.return_value = {"messages": [mock_msg]}

        state = AdaptiveRAGState(question="Safe question?")
        updated_state = await self.nodes.input_query_security_check(state)
        
        self.assertFalse(updated_state.query_blocked)
        self.mock_input_agent.ainvoke.assert_called_once()

    async def test_input_query_security_check_blocked(self):
        mock_msg = AIMessage(content="BLOCKED: contains exploit keywords")
        self.mock_input_agent.ainvoke.return_value = {"messages": [mock_msg]}

        state = AdaptiveRAGState(question="How to hack something?")
        updated_state = await self.nodes.input_query_security_check(state)
        
        self.assertTrue(updated_state.query_blocked)
        self.assertIn("cannot process", updated_state.answer)

    def test_input_query_security_router(self):
        state_blocked = AdaptiveRAGState(question="query", query_blocked=True)
        self.assertEqual(self.nodes.input_query_security_router(state_blocked), "end")

        state_passed = AdaptiveRAGState(question="query", query_blocked=False)
        self.assertEqual(self.nodes.input_query_security_router(state_passed), "query_analyzer")

    async def test_output_answer_security_check(self):
        mock_msg = AIMessage(content="Safe rewritten answer")
        self.mock_output_agent.ainvoke.return_value = {"messages": [mock_msg]}

        state = AdaptiveRAGState(question="query", answer="original answer")
        updated_state = await self.nodes.output_answer_security_check(state)
        
        self.assertEqual(updated_state.answer, "Safe rewritten answer")
        self.mock_output_agent.ainvoke.assert_called_once()

    def test_query_analyzer(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = ToolUse(tool_type="vector_search", analysis="Needs DB search")
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="What is LangGraph?")
        updated_state = self.nodes.query_analyzer(state)

        self.assertEqual(updated_state.tool_type, "vector_search")
        self.assertEqual(updated_state.analysis, "Needs DB search")
        self.mock_llm.with_structured_output.assert_called_once_with(ToolUse)

    def test_vector_search(self):
        docs = [Document(page_content="LangGraph is cool")]
        self.mock_retriever.invoke.return_value = docs

        state = AdaptiveRAGState(question="What is LangGraph?")
        updated_state = self.nodes.vector_search(state)

        self.assertEqual(updated_state.retrieved_docs, docs)
        self.mock_retriever.invoke.assert_called_once_with("What is LangGraph?")

    def test_documents_grader(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = RetrievalGrade(grade="yes", reasoning="Highly relevant doc")
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="What is AI?", retrieved_docs=[Document(page_content="AI is artificial intelligence")])
        updated_state = self.nodes.documents_grader(state)

        self.assertEqual(updated_state.retrieval_grade, "yes")
        self.assertEqual(updated_state.analysis, "Highly relevant doc")
        self.mock_llm.with_structured_output.assert_called_once_with(RetrievalGrade)

    def test_query_rewriter(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = QuestionRewrite(rewritten_question="What is artificial intelligence?", reasoning="More standard phrasing")
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="What is AI?", original_question="What is AI?")
        updated_state = self.nodes.query_rewriter(state)

        self.assertEqual(updated_state.question, "What is artificial intelligence?")
        self.assertEqual(updated_state.rewrite_count, 1)
        self.mock_llm.with_structured_output.assert_called_once_with(QuestionRewrite)

    def test_answer_generator(self):
        mock_invoker = MagicMock()
        mock_invoker.invoke.return_value = AIMessage(content="AI refers to intelligent machines.")
        self.mock_llm.bind.return_value = mock_invoker

        state = AdaptiveRAGState(question="What is AI?", retrieved_docs=[Document(page_content="AI refers to intelligent machines.")])
        updated_state = self.nodes.answer_generator(state)

        self.assertEqual(updated_state.answer, "AI refers to intelligent machines.")
        self.assertEqual(updated_state.generate_count, 1)

    def test_hallucination_detector(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = HallucinationGrade(grade="yes", reasoning="Fully supported")
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="AI?", retrieved_docs=[Document(page_content="AI is good")], answer="AI is good")
        updated_state = self.nodes.hallucination_detector(state)

        self.assertEqual(updated_state.hallucination_grade, "yes")
        self.mock_llm.with_structured_output.assert_called_once_with(HallucinationGrade)

    def test_answer_relevance_grader(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = AnswerRelevanceGrade(grade="yes", reasoning="Addresses query")
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="AI?", answer="AI stands for Artificial Intelligence")
        updated_state = self.nodes.answer_relevance_grader(state)

        self.assertEqual(updated_state.answer_relevance_grade, "yes")
        self.mock_llm.with_structured_output.assert_called_once_with(AnswerRelevanceGrade)

    def test_query_router(self):
        state_vector = AdaptiveRAGState(question="query", tool_type="vector_search")
        self.assertEqual(self.nodes.query_router(state_vector), "vector_search")

        state_external = AdaptiveRAGState(question="query", tool_type="external_search")
        self.assertEqual(self.nodes.query_router(state_external), "external_search")

    def test_grader_router(self):
        state_yes = AdaptiveRAGState(question="query", retrieval_grade="yes")
        self.assertEqual(self.nodes.grader_router(state_yes), "answer_generator")

        state_no = AdaptiveRAGState(question="query", retrieval_grade="no", rewrite_count=0)
        self.assertEqual(self.nodes.grader_router(state_no), "query_rewriter")

        state_max_rewrites = AdaptiveRAGState(question="query", retrieval_grade="no", rewrite_count=5)
        self.assertEqual(self.nodes.grader_router(state_max_rewrites), "external_search")

if __name__ == "__main__":
    unittest.main()
