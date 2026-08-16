"""Security guardrail nodes for Adaptive RAG."""

import logging
import time
from state.adaptive_state import AdaptiveRAGState
from langchain_core.messages import HumanMessage
from nodes.guardrails import Guardrails

logger = logging.getLogger(__name__)


class SecurityNodes:
    """Nodes responsible for input query validation and output answer sanitization."""

    def __init__(self, llm_checker, llm_generator=None, guardrails: Guardrails | None = None):
        self.llm_checker = llm_checker
        self.llm_generator = llm_generator if llm_generator is not None else llm_checker
        self.guardrails = guardrails or Guardrails(llm_checker=self.llm_checker, llm_generator=self.llm_generator)
        self.input_guardrail_agent = self.guardrails.get_input_guardrail_agent()
        self.output_guardrail_agent = self.guardrails.get_output_guardrail_agent()

    async def input_query_security_check(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Validate the user's query before entering the RAG workflow."""
        t0 = time.perf_counter()
        logger.info("Running input security check.")
        logger.debug("Input query content: %s", state.question)

        try:
            response = await self.input_guardrail_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=state.question)
                    ]
                }
            )

            messages = response.get("messages", [])

            if not messages:
                logger.warning("Input security check failed to return messages. Failing closed.")
                state.query_blocked = True
                state.answer = "⚠️ This request could not be processed due to security filtering."
                elapsed = time.perf_counter() - t0
                state.latency_breakdown["security_input"] = round(elapsed, 4)
                return state

            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            content_clean = content.strip().upper()

            if "BLOCKED" in content_clean or content_clean.startswith("UNSAFE"):
                logger.warning("Input query blocked by security guardrail.")
                state.query_blocked = True
                state.answer = "I cannot process this request. Please rephrase your question."
            elif "SAFE" in content_clean or "PASSED" in content_clean:
                logger.info("Input query passed security check.")
                state.query_blocked = False
            else:
                denial_patterns = ["REQUEST IS UNSAFE", "DENIED", "ACCESS DENIED", "POLICY VIOLATION DETECTED"]
                if any(pattern in content_clean for pattern in denial_patterns):
                    logger.warning("Input query denied by safety guardrail agent.")
                    state.query_blocked = True
                    state.answer = "I cannot process this request. Please rephrase your question."
                else:
                    logger.info("Input query treated as SAFE by guardrail agent.")
                    state.query_blocked = False

        except Exception as e:
            logger.error("Exception during input security guardrail check: %s. Failing closed.", e, exc_info=True)
            state.query_blocked = True
            state.answer = "⚠️ This request could not be processed due to security filtering."

        elapsed = time.perf_counter() - t0
        state.latency_breakdown["security_input"] = round(elapsed, 4)
        return state

    async def output_answer_security_check(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Validate and sanitize the final generated answer with fail-closed safety."""
        t0 = time.perf_counter()
        logger.info("Running output security check.")
        logger.debug("Raw generated answer: %s", state.answer)

        if not state.answer:
            logger.warning("Empty answer provided to output security check.")
            return state

        try:
            response = await self.output_guardrail_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content=f"Review the following answer for safety and policy compliance:\n\n{state.answer}")
                    ]
                }
            )

            messages = response.get("messages", [])

            if not messages:
                logger.warning("Output security check failed to return messages. Failing closed for safety.")
                state.answer = "⚠️ The generated answer could not be verified by security guardrails. Please rephrase your question."
                return state

            last_message = messages[-1]
            sanitized_answer = last_message.content if hasattr(last_message, "content") else str(last_message)

            if sanitized_answer and sanitized_answer.strip():
                state.answer = sanitized_answer.strip()
                logger.info("Output answer validated/sanitized successfully.")
            else:
                logger.warning("Output security check returned empty content. Failing closed for safety.")
                state.answer = "⚠️ The generated answer could not be verified by security guardrails. Please rephrase your question."

        except Exception as e:
            logger.error("Exception during output security guardrail check: %s. Failing closed.", e, exc_info=True)
            state.answer = "⚠️ The generated answer could not be verified by security guardrails. Please rephrase your question."

        elapsed = time.perf_counter() - t0
        state.latency_breakdown["security_output"] = round(elapsed, 4)
        return state
