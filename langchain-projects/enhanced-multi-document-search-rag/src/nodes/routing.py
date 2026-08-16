"""Routing logic and conditional edge functions for Adaptive RAG."""

import logging
from state.adaptive_state import AdaptiveRAGState
from config.llmgateway_config import Config

logger = logging.getLogger(__name__)


def input_query_security_router(state: AdaptiveRAGState) -> str:
    """Route based on input security check result."""
    if state.query_blocked:
        logger.warning("Routing to END due to security violation.")
        return "end"
    logger.info("Routing to query_analyzer after successful security check.")
    return "query_analyzer"


def query_router(state: AdaptiveRAGState) -> str:
    """
    Routes the query to the appropriate retrieval node based on query analysis.
    """
    logger.info("Routing query to: %s", state.tool_type)
    if state.tool_type in ("hybrid_retrieval", "vector_search"):
        return "hybrid_retrieval"

    if state.tool_type == "external_search":
        return "external_search"

    raise ValueError(
        f"Invalid tool_type: {state.tool_type}"
    )


def grader_router(state: AdaptiveRAGState) -> str:
    """Route to the next node based on retrieval grading."""
    if state.retrieval_score is not None:
        is_passed = state.retrieval_score >= Config.RETRIEVAL_GRADE_PASS_THRESHOLD
    else:
        is_passed = state.retrieval_grade in ("yes", "pass")

    logger.info(
        "Routing from retrieval grader. Passed: %s, Grade: %s, Score: %s, Rewrite count: %d",
        is_passed, state.retrieval_grade, state.retrieval_score, state.rewrite_count
    )
    if is_passed:
        return "answer_generator"
    if state.rewrite_count >= Config.MAX_REWRITES:
        return "external_search"
    return "query_rewriter"


def hallucination_router(state: AdaptiveRAGState) -> str:
    """Route based on hallucination detection, score threshold, and retry count."""
    if state.hallucination_score is not None:
        is_grounded = state.hallucination_score >= Config.HALLUCINATION_GRADE_PASS_THRESHOLD
    else:
        is_grounded = state.hallucination_grade in ("yes", "pass")

    logger.info(
        "Routing from hallucination detector. Grounded: %s, Grade: %s, Score: %s, Generate count: %d",
        is_grounded, state.hallucination_grade, state.hallucination_score, state.generate_count
    )
    if is_grounded:
        return "answer_relevance_grader"
    if state.generate_count >= Config.MAX_GENERATIONS:
        if state.tool_type == "external_search" or state.external_results:
            logger.warning("Answer remains ungrounded after MAX_GENERATIONS with external search. Refusing to return hallucinated answer.")
            state.answer = "I am unable to provide a verified answer based on the available documents and external sources. Please refine or rephrase your question."
            return "output_answer_security_check"
        return "external_search"
    return "answer_generator"


def answer_relevance_router(state: AdaptiveRAGState) -> str:
    """Route to next node based on answer relevance grader output and score threshold."""
    if state.answer_relevance_score is not None:
        is_relevant = state.answer_relevance_score >= Config.ANSWER_RELEVANCE_PASS_THRESHOLD
    else:
        is_relevant = state.answer_relevance_grade in ("yes", "pass")

    logger.info(
        "Routing from relevance grader. Relevant: %s, Grade: %s, Score: %s, Rewrite count: %d",
        is_relevant, state.answer_relevance_grade, state.answer_relevance_score, state.rewrite_count
    )
    if is_relevant:
        return "output_answer_security_check"
    if state.rewrite_count >= Config.MAX_REWRITES or state.tool_type == "external_search" or state.external_results:
        if state.tool_type == "external_search" or state.external_results:
            return "output_answer_security_check"
        return "external_search"
    return "query_rewriter"


def output_answer_security_router(state: AdaptiveRAGState) -> str:
    """
    Route after output security check.
    If the output guardrail modified/rewrote the answer, run hallucination and
    relevance validation again (with a loop guard) to ensure the rewritten answer
    remains grounded and relevant.
    """
    if getattr(state, "output_modified", False) and not state.query_blocked and state.guardrail_recheck_count <= 1:
        logger.info(
            "Routing from output guardrail back to hallucination_detector (guardrail recheck count: %d).",
            state.guardrail_recheck_count
        )
        # Reset output_modified flag for the recheck pass
        state.output_modified = False
        return "hallucination_detector"

    logger.info("Routing from output security check to end.")
    return "end"

