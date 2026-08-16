"""Evaluation and grading nodes for Adaptive RAG with continuous scoring and configurable thresholds."""

import logging
import time
from state.adaptive_state import AdaptiveRAGState
from langchain_core.prompts import ChatPromptTemplate
from nodes.schema import RetrievalGrade, HallucinationGrade, AnswerRelevanceGrade
from config.llmgateway_config import Config
from prompts.rag_prompts import (
    RETRIEVAL_GRADER_SYSTEM_PROMPT,
    HALLUCINATION_DETECTOR_SYSTEM_PROMPT,
    ANSWER_RELEVANCE_GRADER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class EvaluationNodes:
    """Nodes responsible for retrieval document grading, hallucination detection, and answer relevance grading."""

    def __init__(self, llm_checker):
        self.llm_checker = llm_checker

    async def documents_grader(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Grade the relevance and sufficiency of retrieved documents with continuous scoring."""
        t0 = time.perf_counter()
        logger.info("Grading %d retrieved documents.", len(state.retrieved_docs))
        logger.debug("Grading documents for query: %s", state.question)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RETRIEVAL_GRADER_SYSTEM_PROMPT),
            (
                "human",
            """
            User Question:
            {question}

            Retrieved Documents:
            {documents}
            """
            ),
        ])

        documents = "\n\n".join(
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_docs, start=1)
        )

        response = await self.llm_checker.with_structured_output(
            RetrievalGrade
        ).ainvoke(
            prompt.format_messages(
                question=state.question,
                documents=documents,
            )
        )

        score = getattr(response, "score", 1.0 if response.grade == "yes" else 0.0)
        passed = score >= Config.RETRIEVAL_GRADE_PASS_THRESHOLD if score is not None else (response.grade == "yes")
        
        state.retrieval_score = score
        state.retrieval_grade = "yes" if passed else "no"
        state.retrieval_evaluation = response.reasoning
        state.analysis = response.reasoning
        elapsed = time.perf_counter() - t0
        state.grader_latency = round(state.grader_latency + elapsed, 4)
        state.latency_breakdown["grader_documents"] = round(elapsed, 4)
        logger.info(
            "Retrieval grader output in %.3fs - Score: %s, Decision: %s, Grade: %s (Threshold: %.2f)",
            elapsed,
            score,
            getattr(response, "decision", state.retrieval_grade),
            state.retrieval_grade,
            Config.RETRIEVAL_GRADE_PASS_THRESHOLD
        )
        logger.debug("Retrieval grader evaluation: %s", state.analysis)
        return state

    async def hallucination_detector(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Detect hallucinations in the generated answer with continuous groundedness scoring."""
        t0 = time.perf_counter()
        logger.debug("Running hallucination check for generated answer.")

        context_sections = []
        if state.retrieved_docs:
            doc_str = "\n\n".join(
                f"[Document {i}]\n{doc.page_content}"
                for i, doc in enumerate(state.retrieved_docs, start=1)
            )
            context_sections.append(f"Retrieved Documents:\n{doc_str}")
        if state.external_results:
            context_sections.append(f"External Search Results:\n{state.external_results}")

        documents_content: str = "\n\n".join(context_sections) if context_sections else "No relevant context provided."

        prompt = ChatPromptTemplate.from_messages([
            ("system", HALLUCINATION_DETECTOR_SYSTEM_PROMPT),
            (
                "human",
                """
                User Question:
                {question}

                Retrieved Documents:
                {documents}

                Generated Answer:
                {answer}
                """
            ),
        ])
        
        response = await self.llm_checker.with_structured_output(HallucinationGrade).ainvoke(
            prompt.format_messages(question=state.question, documents=documents_content, answer=state.answer)
        )

        score = getattr(response, "score", 1.0 if response.grade == "yes" else 0.0)
        passed = score >= Config.HALLUCINATION_GRADE_PASS_THRESHOLD if score is not None else (response.grade == "yes")

        state.hallucination_score = score
        state.hallucination_grade = "yes" if passed else "no"
        state.grounding_evaluation = response.reasoning
        state.analysis = response.reasoning
        elapsed = time.perf_counter() - t0
        state.grader_latency = round(state.grader_latency + elapsed, 4)
        state.latency_breakdown["grader_hallucination"] = round(elapsed, 4)
        logger.info(
            "Hallucination check output in %.3fs - Score: %s, Decision: %s, Grade: %s (Threshold: %.2f)",
            elapsed,
            score,
            getattr(response, "decision", state.hallucination_grade),
            state.hallucination_grade,
            Config.HALLUCINATION_GRADE_PASS_THRESHOLD
        )
        logger.debug("Hallucination check evaluation: %s", state.analysis)
        return state

    async def answer_relevance_grader(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Grades answer relevance with continuous scoring and configurable threshold."""
        t0 = time.perf_counter()
        logger.debug("Running answer relevance check.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_RELEVANCE_GRADER_SYSTEM_PROMPT),
            (
                "human",
                """
                User Question:
                {question}

                Generated Answer:
                {answer}
                """
            ),
        ])

        response = await self.llm_checker.with_structured_output(AnswerRelevanceGrade).ainvoke(
            prompt.format_messages(question=state.question, answer=state.answer)
        )

        score = getattr(response, "score", 1.0 if response.grade == "yes" else 0.0)
        passed = score >= Config.ANSWER_RELEVANCE_PASS_THRESHOLD if score is not None else (response.grade == "yes")

        state.answer_relevance_score = score
        state.answer_relevance_grade = "yes" if passed else "no"
        state.relevance_evaluation = response.reasoning
        state.analysis = response.reasoning
        elapsed = time.perf_counter() - t0
        state.grader_latency = round(state.grader_latency + elapsed, 4)
        state.latency_breakdown["grader_relevance"] = round(elapsed, 4)
        logger.info(
            "Answer relevance output in %.3fs - Score: %s, Decision: %s, Grade: %s (Threshold: %.2f)",
            elapsed,
            score,
            getattr(response, "decision", state.answer_relevance_grade),
            state.answer_relevance_grade,
            Config.ANSWER_RELEVANCE_PASS_THRESHOLD
        )
        logger.debug("Answer relevance reasoning: %s", state.analysis)
        return state
