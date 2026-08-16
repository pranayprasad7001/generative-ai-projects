import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from state.adaptive_state import AdaptiveRAGState
from nodes.schema import ToolUse, RetrievalGrade, QuestionRewrite, HallucinationGrade, AnswerRelevanceGrade
from nodes.adaptive_node import AdaptiveRAGNodes
from langchain_core.documents import Document
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
        
        self.nodes = AdaptiveRAGNodes(self.mock_retriever, self.mock_llm)

    async def test_input_query_security_check_pass(self):
        # Setup mock return from input guardrail agent
        mock_msg = AIMessage(content="PASSED")
        self.mock_input_agent.ainvoke.return_value = {"messages": [mock_msg]}

        state = AdaptiveRAGState(question="Safe question?")
        updated_state = await self.nodes.input_query_security_check(state)
        
        self.assertFalse(updated_state.query_blocked)
        self.mock_input_agent.ainvoke.assert_called_once()

    async def test_input_query_security_check_non_violating_phrasing(self):
        # Setup mock return where message contains words like 'violation' in negative context
        mock_msg = AIMessage(content="The query does not violate safety policies. SAFE.")
        self.mock_input_agent.ainvoke.return_value = {"messages": [mock_msg]}

        state = AdaptiveRAGState(question="How do ORMs prevent SQL injection?")
        updated_state = await self.nodes.input_query_security_check(state)
        
        self.assertFalse(updated_state.query_blocked)

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

    async def test_output_answer_security_check_fail_closed(self):
        self.mock_output_agent.ainvoke.side_effect = Exception("Guardrail service unavailable")
        state = AdaptiveRAGState(question="query", answer="original answer")
        updated_state = await self.nodes.output_answer_security_check(state)
        self.assertIn("could not be verified", updated_state.answer)

    async def test_query_analyzer(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=ToolUse(tool_type="hybrid_retrieval", analysis="Needs DB search"))
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="What is LangGraph?")
        updated_state = await self.nodes.query_analyzer(state)

        self.assertEqual(updated_state.tool_type, "hybrid_retrieval")
        self.assertEqual(updated_state.analysis, "Needs DB search")
        self.mock_llm.with_structured_output.assert_called_once_with(ToolUse)

    async def test_hybrid_retrieval(self):
        docs = [Document(page_content="LangGraph is cool")]
        self.mock_retriever.invoke.return_value = docs

        state = AdaptiveRAGState(question="What is LangGraph?")
        updated_state = await self.nodes.hybrid_retrieval(state)

        self.assertEqual(updated_state.retrieved_docs, docs)
        self.mock_retriever.invoke.assert_called_once_with("What is LangGraph?")

    async def test_hybrid_retrieval_with_config_retriever(self):
        docs = [Document(page_content="Dynamic Retriever Result")]
        custom_retriever = MagicMock()
        custom_retriever.invoke.return_value = docs

        state = AdaptiveRAGState(question="What is LangGraph?")
        config = {"configurable": {"retriever": custom_retriever}}
        updated_state = await self.nodes.hybrid_retrieval(state, config=config)

        self.assertEqual(updated_state.retrieved_docs, docs)
        custom_retriever.invoke.assert_called_once_with("What is LangGraph?")

    async def test_documents_grader(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=RetrievalGrade(grade="yes", reasoning="Highly relevant doc"))
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="What is AI?", retrieved_docs=[Document(page_content="AI is artificial intelligence")])
        updated_state = await self.nodes.documents_grader(state)

        self.assertEqual(updated_state.retrieval_grade, "yes")
        self.assertEqual(updated_state.analysis, "Highly relevant doc")
        self.mock_llm.with_structured_output.assert_called_once_with(RetrievalGrade)

    async def test_query_rewriter(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=QuestionRewrite(rewritten_question="What is artificial intelligence?", reasoning="More standard phrasing"))
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="What is AI?", original_question="What is AI?")
        updated_state = await self.nodes.query_rewriter(state)

        self.assertEqual(updated_state.question, "What is artificial intelligence?")
        self.assertEqual(updated_state.rewrite_count, 1)
        self.mock_llm.with_structured_output.assert_called_once_with(QuestionRewrite)

    async def test_answer_generator(self):
        mock_invoker = MagicMock()
        mock_invoker.ainvoke = AsyncMock(return_value=AIMessage(content="AI refers to intelligent machines."))
        self.mock_llm.bind.return_value = mock_invoker

        state = AdaptiveRAGState(question="What is AI?", retrieved_docs=[Document(page_content="AI refers to intelligent machines.")])
        updated_state = await self.nodes.answer_generator(state)

        self.assertEqual(updated_state.answer, "AI refers to intelligent machines.")
        self.assertEqual(updated_state.generate_count, 1)

    async def test_hallucination_detector(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=HallucinationGrade(grade="yes", reasoning="Fully supported"))
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="AI?", retrieved_docs=[Document(page_content="AI is good")], answer="AI is good")
        updated_state = await self.nodes.hallucination_detector(state)

        self.assertEqual(updated_state.hallucination_grade, "yes")
        self.mock_llm.with_structured_output.assert_called_once_with(HallucinationGrade)

    async def test_answer_relevance_grader(self):
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=AnswerRelevanceGrade(grade="yes", reasoning="Addresses query"))
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(question="AI?", answer="AI stands for Artificial Intelligence")
        updated_state = await self.nodes.answer_relevance_grader(state)

        self.assertEqual(updated_state.answer_relevance_grade, "yes")
        self.mock_llm.with_structured_output.assert_called_once_with(AnswerRelevanceGrade)

    async def test_dual_model_routing(self):
        """Verify that llm_checker is used for evaluation and llm_generator for answer generation."""
        mock_generator = MagicMock()
        mock_checker = MagicMock()
        dual_nodes = AdaptiveRAGNodes(self.mock_retriever, llm_generator=mock_generator, llm_checker=mock_checker)

        # 1. Check query_analyzer calls mock_checker
        mock_structured_checker = MagicMock()
        mock_structured_checker.ainvoke = AsyncMock(return_value=ToolUse(tool_type="vector_search", analysis="test"))
        mock_checker.with_structured_output.return_value = mock_structured_checker

        state = AdaptiveRAGState(question="Test question")
        await dual_nodes.query_analyzer(state)
        mock_checker.with_structured_output.assert_called_with(ToolUse)
        mock_generator.with_structured_output.assert_not_called()

        # 2. Check answer_generator calls mock_generator
        mock_invoker = MagicMock()
        mock_invoker.ainvoke = AsyncMock(return_value=AIMessage(content="Generated answer"))
        mock_generator.bind.return_value = mock_invoker

        state_gen = AdaptiveRAGState(question="Test", retrieved_docs=[Document(page_content="Content")])
        await dual_nodes.answer_generator(state_gen)
        mock_generator.bind.assert_called_once()

    def test_query_router(self):
        state_vector = AdaptiveRAGState(question="query", tool_type="hybrid_retrieval")
        self.assertEqual(self.nodes.query_router(state_vector), "hybrid_retrieval")

        state_external = AdaptiveRAGState(question="query", tool_type="external_search")
        self.assertEqual(self.nodes.query_router(state_external), "external_search")

    def test_grader_router(self):
        state_yes = AdaptiveRAGState(question="query", retrieval_grade="yes")
        self.assertEqual(self.nodes.grader_router(state_yes), "answer_generator")

        state_no = AdaptiveRAGState(question="query", retrieval_grade="no", rewrite_count=0)
        self.assertEqual(self.nodes.grader_router(state_no), "query_rewriter")

        state_max_rewrites = AdaptiveRAGState(question="query", retrieval_grade="no", rewrite_count=5)
        self.assertEqual(self.nodes.grader_router(state_max_rewrites), "external_search")

    def test_hallucination_router(self):
        state_yes = AdaptiveRAGState(question="query", hallucination_grade="yes")
        self.assertEqual(self.nodes.hallucination_router(state_yes), "answer_relevance_grader")

        state_no = AdaptiveRAGState(question="query", hallucination_grade="no", generate_count=1)
        self.assertEqual(self.nodes.hallucination_router(state_no), "answer_generator")

        state_max_gens = AdaptiveRAGState(question="query", hallucination_grade="no", generate_count=5)
        self.assertEqual(self.nodes.hallucination_router(state_max_gens), "external_search")

    def test_answer_relevance_router(self):
        state_yes = AdaptiveRAGState(question="query", answer_relevance_grade="yes")
        self.assertEqual(self.nodes.answer_relevance_router(state_yes), "output_answer_security_check")

        state_no = AdaptiveRAGState(question="query", answer_relevance_grade="no", rewrite_count=0)
        self.assertEqual(self.nodes.answer_relevance_router(state_no), "query_rewriter")

        state_external = AdaptiveRAGState(question="query", answer_relevance_grade="no", tool_type="external_search")
        self.assertEqual(self.nodes.answer_relevance_router(state_external), "output_answer_security_check")

    async def test_answer_generator_critique_regeneration(self):
        """Test that answer_generator uses critique feedback and bypasses cache when self-correcting after a hallucination."""
        mock_invoker = MagicMock()
        mock_invoker.ainvoke = AsyncMock(return_value=AIMessage(content="Corrected grounded answer."))
        self.mock_llm.bind.return_value = mock_invoker

        state = AdaptiveRAGState(
            question="What is AI?",
            retrieved_docs=[Document(page_content="AI is machine intelligence.")],
            answer="AI is sentient robots taking over.",
            generate_count=1,
            hallucination_grade="no",
            analysis="The claim that robots are taking over is not supported by context."
        )
        updated_state = await self.nodes.answer_generator(state)

        self.assertEqual(updated_state.answer, "Corrected grounded answer.")
        self.assertEqual(updated_state.generate_count, 2)
        self.mock_llm.bind.assert_called_once_with(extra_body={"cache": {"use-cache": False}})

    async def test_external_search_preserves_retrieved_docs(self):
        """Test that external_search does NOT wipe retrieved_docs."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                ToolMessage(content="https://en.wikipedia.org/wiki/AI", tool_call_id="call_1"),
                AIMessage(content="External summary of AI.")
            ]
        }
        self.nodes.external_search_agent = mock_agent

        initial_docs = [Document(page_content="Local context doc.")]
        state = AdaptiveRAGState(question="What is AI?", retrieved_docs=list(initial_docs))
        updated_state = await self.nodes.external_search(state)

        self.assertEqual(updated_state.retrieved_docs, initial_docs)
        self.assertEqual(updated_state.external_results, "External summary of AI.")
        citation_urls = [c["url"] if isinstance(c, dict) else c for c in updated_state.external_citations]
        self.assertIn("https://en.wikipedia.org/wiki/AI", citation_urls)

    async def test_hallucination_detector_with_external_results(self):
        """Test that hallucination_detector includes external_results in context."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=HallucinationGrade(grade="yes", reasoning="External info verified"))
        self.mock_llm.with_structured_output.return_value = mock_structured_llm

        state = AdaptiveRAGState(
            question="What is AI?",
            retrieved_docs=[Document(page_content="Local doc")],
            external_results="External web info on AI",
            answer="AI is great"
        )
        updated_state = await self.nodes.hallucination_detector(state)

        self.assertEqual(updated_state.hallucination_grade, "yes")
        self.mock_llm.with_structured_output.assert_called_once_with(HallucinationGrade)

if __name__ == "__main__":
    unittest.main()
