import logging
import time
from state.adaptive_state import AdaptiveRAGState
from langchain_core.prompts import ChatPromptTemplate
from prompts.rag_prompts import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_REGENERATION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class GenerationNodes:
    """Nodes responsible for drafting and self-correcting grounded answer generation."""

    def __init__(self, llm_generator):
        self.llm_generator = llm_generator

    async def answer_generator(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Generate or regenerate answer based on retrieved documents and self-correction critique."""
        t0 = time.perf_counter()
        logger.info("Generating answer. Generate count: %d", state.generate_count + 1)

        context_sections = []
        if state.retrieved_docs:
            doc_str = "\n\n".join(
                f"[Document {i}]\n{doc.page_content}"
                for i, doc in enumerate(state.retrieved_docs, start=1)
            )
            context_sections.append(f"Retrieved Documents:\n{doc_str}")
        if state.external_results:
            context_sections.append(f"External Search Results:\n{state.external_results}")

        documents_content: str = "\n\n".join(context_sections) if context_sections else "No relevant documents found."

        # Self-correction regeneration branch if previously flagged for hallucination
        critique = state.grounding_evaluation or state.analysis
        if state.generate_count > 0 and state.hallucination_grade in ("no", "retry", "fail") and critique:
            logger.info("Executing critique-aware self-correction answer regeneration.")
            prompt = ChatPromptTemplate.from_messages([
                ("system", GENERATOR_REGENERATION_SYSTEM_PROMPT),
                (
                    "human",
                    """
                    User Question:
                    {question}

                    Retrieved Context:
                    {context}

                    Previous Draft Answer:
                    {previous_answer}

                    Hallucination Critique & Feedback:
                    {critique}
                    """
                ),
            ])
            response = await self.llm_generator.bind(
                extra_body={"cache": {"use-cache": False}}
            ).ainvoke(
                prompt.format_messages(
                    question=state.question,
                    context=documents_content,
                    previous_answer=state.answer,
                    critique=critique
                )
            )
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", GENERATOR_SYSTEM_PROMPT),
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

            response = await self.llm_generator.bind(
                extra_body={"cache": {"use-cache": True, "ttl": 1800}}
            ).ainvoke(prompt.format_messages(question=state.question, documents=documents_content))

        state.answer = response.content
        state.generate_count += 1
        elapsed = time.perf_counter() - t0
        state.generation_latency = round(state.generation_latency + elapsed, 4)
        state.latency_breakdown["generation"] = round(state.latency_breakdown.get("generation", 0.0) + elapsed, 4)
        logger.info("Answer generated successfully in %.3fs. Total generations: %d", elapsed, state.generate_count)
        return state
