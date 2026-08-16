import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from state.adaptive_state import AdaptiveRAGState
from config.llmgateway_config import Config
from nodes.routing import (
    input_query_security_router,
    query_router,
    grader_router,
    hallucination_router,
    answer_relevance_router,
)


class TestRouting(unittest.TestCase):

    def test_input_query_security_router(self):
        state_blocked = AdaptiveRAGState(question="hack", query_blocked=True)
        self.assertEqual(input_query_security_router(state_blocked), "end")

        state_passed = AdaptiveRAGState(question="valid query", query_blocked=False)
        self.assertEqual(input_query_security_router(state_passed), "query_analyzer")

    def test_query_router(self):
        state_hybrid = AdaptiveRAGState(question="q", tool_type="hybrid_retrieval")
        self.assertEqual(query_router(state_hybrid), "hybrid_retrieval")

        state_vec = AdaptiveRAGState(question="q", tool_type="vector_search")
        self.assertEqual(query_router(state_vec), "hybrid_retrieval")

        state_ext = AdaptiveRAGState(question="q", tool_type="external_search")
        self.assertEqual(query_router(state_ext), "external_search")

        state_inv = AdaptiveRAGState(question="q", tool_type=None)
        with self.assertRaises(ValueError):
            query_router(state_inv)

    def test_grader_router(self):
        state_yes = AdaptiveRAGState(question="q", retrieval_grade="yes")
        self.assertEqual(grader_router(state_yes), "answer_generator")

        state_score_pass = AdaptiveRAGState(question="q", retrieval_score=0.85)
        self.assertEqual(grader_router(state_score_pass), "answer_generator")

        state_no_retry = AdaptiveRAGState(question="q", retrieval_grade="no", retrieval_score=0.4, rewrite_count=0)
        self.assertEqual(grader_router(state_no_retry), "query_rewriter")

        state_no_max = AdaptiveRAGState(question="q", retrieval_grade="no", rewrite_count=Config.MAX_REWRITES)
        self.assertEqual(grader_router(state_no_max), "external_search")

    def test_hallucination_router(self):
        state_grounded = AdaptiveRAGState(question="q", hallucination_grade="yes")
        self.assertEqual(hallucination_router(state_grounded), "answer_relevance_grader")

        state_score_grounded = AdaptiveRAGState(question="q", hallucination_score=0.9)
        self.assertEqual(hallucination_router(state_score_grounded), "answer_relevance_grader")

        state_hallucinated = AdaptiveRAGState(question="q", hallucination_grade="no", hallucination_score=0.3, generate_count=1)
        self.assertEqual(hallucination_router(state_hallucinated), "answer_generator")

        state_hallucinated_max = AdaptiveRAGState(question="q", hallucination_grade="no", generate_count=Config.MAX_GENERATIONS)
        self.assertEqual(hallucination_router(state_hallucinated_max), "external_search")

    def test_answer_relevance_router(self):
        state_rel = AdaptiveRAGState(question="q", answer_relevance_grade="yes")
        self.assertEqual(answer_relevance_router(state_rel), "output_answer_security_check")

        state_score_rel = AdaptiveRAGState(question="q", answer_relevance_score=0.8)
        self.assertEqual(answer_relevance_router(state_score_rel), "output_answer_security_check")

        state_irrel = AdaptiveRAGState(question="q", answer_relevance_grade="no", answer_relevance_score=0.2, rewrite_count=0)
        self.assertEqual(answer_relevance_router(state_irrel), "query_rewriter")

        state_irrel_max = AdaptiveRAGState(question="q", answer_relevance_grade="no", rewrite_count=Config.MAX_REWRITES)
        self.assertEqual(answer_relevance_router(state_irrel_max), "external_search")


    def test_grader_router_score_authoritative(self):
        # Even if grade is 'yes', a low score (< 0.7) must fail
        state_failing_score = AdaptiveRAGState(question="q", retrieval_grade="yes", retrieval_score=0.3, rewrite_count=0)
        self.assertEqual(grader_router(state_failing_score), "query_rewriter")

    def test_hallucination_router_score_authoritative(self):
        # Even if grade is 'yes', a low score (< 0.7) must trigger regeneration
        state_failing_score = AdaptiveRAGState(question="q", hallucination_grade="yes", hallucination_score=0.2, generate_count=1)
        self.assertEqual(hallucination_router(state_failing_score), "answer_generator")

    def test_answer_relevance_router_score_authoritative(self):
        # Even if grade is 'yes', a low score (< 0.7) must trigger rewrite
        state_failing_score = AdaptiveRAGState(question="q", answer_relevance_grade="yes", answer_relevance_score=0.2, rewrite_count=0)
        self.assertEqual(answer_relevance_router(state_failing_score), "query_rewriter")

    def test_hallucination_router_max_generations_external_search_refusal(self):
        state = AdaptiveRAGState(
            question="q",
            hallucination_grade="no",
            hallucination_score=0.2,
            generate_count=Config.MAX_GENERATIONS,
            external_results="Extracted search snippet",
            answer="Hallucinated claim"
        )
        route = hallucination_router(state)
        self.assertEqual(route, "output_answer_security_check")
        self.assertIn("unable to provide a verified answer", state.answer)


if __name__ == "__main__":
    unittest.main()
