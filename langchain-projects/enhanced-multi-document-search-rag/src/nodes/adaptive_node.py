"""Composite / Facade Nodes for Adaptive RAG Workflow.

This module unifies the specialized node classes:
- SecurityNodes (nodes/security_nodes.py)
- RetrievalNodes (nodes/retrieval_nodes.py)
- GenerationNodes (nodes/generation_nodes.py)
- EvaluationNodes (nodes/evaluation_nodes.py)
- Routing functions (nodes/routing.py)
"""

import logging
from typing import List, Optional
from state.adaptive_state import AdaptiveRAGState
from langchain_core.runnables import RunnableConfig
from nodes.guardrails import Guardrails
from nodes.security_nodes import SecurityNodes
from nodes.retrieval_nodes import RetrievalNodes
from nodes.generation_nodes import GenerationNodes
from nodes.evaluation_nodes import EvaluationNodes
from nodes.routing import (
    input_query_security_router,
    query_router,
    grader_router,
    hallucination_router,
    answer_relevance_router,
)

logger = logging.getLogger(__name__)


class AdaptiveRAGNodes:
    """Facade composing all modular node components for the Adaptive RAG graph."""

    def __init__(self, retriever, llm_generator=None, llm_checker=None, llm=None):
        """
        Initialize RAG nodes with retriever, llm_generator, and llm_checker.

        Args:
            retriever: Document retriever instance
            llm_generator: Language model instance for answer generation
            llm_checker: Language model instance for structured checks/routing/grading
            llm: Fallback language model instance (for backward compatibility)
        """
        logger.info("Initializing AdaptiveRAGNodes with retriever, LLM Generator, and LLM Checker.")
        self._retriever = retriever
        
        # Handle backward compatibility / flexible arguments
        if llm_generator is None and llm is not None:
            self.llm_generator = llm
            self.llm_checker = llm_checker if llm_checker is not None else llm
        elif llm_generator is not None and llm_checker is None:
            self.llm_generator = llm_generator
            self.llm_checker = llm_generator
        else:
            self.llm_generator = llm_generator
            self.llm_checker = llm_checker

        self.llm = self.llm_generator

        # Shared guardrails & agents
        self.guardrails = Guardrails(llm_checker=self.llm_checker, llm_generator=self.llm_generator)

        # Initialize modular node handlers
        self.security_nodes = SecurityNodes(
            llm_checker=self.llm_checker,
            llm_generator=self.llm_generator,
            guardrails=self.guardrails
        )
        self.retrieval_nodes = RetrievalNodes(
            retriever=self._retriever,
            llm_checker=self.llm_checker,
            llm_generator=self.llm_generator,
            guardrails=self.guardrails
        )
        self.generation_nodes = GenerationNodes(
            llm_generator=self.llm_generator
        )
        self.evaluation_nodes = EvaluationNodes(
            llm_checker=self.llm_checker
        )

    @property
    def retriever(self):
        return self.retrieval_nodes.retriever

    @retriever.setter
    def retriever(self, value):
        self._retriever = value
        self.retrieval_nodes.retriever = value

    @property
    def input_guardrail_agent(self):
        return self.security_nodes.input_guardrail_agent

    @input_guardrail_agent.setter
    def input_guardrail_agent(self, value):
        self.security_nodes.input_guardrail_agent = value

    @property
    def output_guardrail_agent(self):
        return self.security_nodes.output_guardrail_agent

    @output_guardrail_agent.setter
    def output_guardrail_agent(self, value):
        self.security_nodes.output_guardrail_agent = value

    @property
    def external_search_agent(self):
        return self.retrieval_nodes.external_search_agent

    @external_search_agent.setter
    def external_search_agent(self, value):
        self.retrieval_nodes.external_search_agent = value

    # -------------------------------------------------------------------------
    # Security Node Delegations
    # -------------------------------------------------------------------------
    async def input_query_security_check(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.security_nodes.input_query_security_check(state)

    async def output_answer_security_check(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.security_nodes.output_answer_security_check(state)

    # -------------------------------------------------------------------------
    # Retrieval & Query Analysis Node Delegations
    # -------------------------------------------------------------------------
    async def query_analyzer(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.retrieval_nodes.query_analyzer(state)

    async def hybrid_retrieval(self, state: AdaptiveRAGState, config: RunnableConfig | None = None) -> AdaptiveRAGState:
        return await self.retrieval_nodes.hybrid_retrieval(state, config=config)

    # Backward compatibility alias
    vector_search = hybrid_retrieval

    async def external_search(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.retrieval_nodes.external_search(state)

    async def query_rewriter(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.retrieval_nodes.query_rewriter(state)

    # -------------------------------------------------------------------------
    # Generation Node Delegations
    # -------------------------------------------------------------------------
    async def answer_generator(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.generation_nodes.answer_generator(state)

    # -------------------------------------------------------------------------
    # Evaluation & Grading Node Delegations
    # -------------------------------------------------------------------------
    async def documents_grader(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.evaluation_nodes.documents_grader(state)

    async def hallucination_detector(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.evaluation_nodes.hallucination_detector(state)

    async def answer_relevance_grader(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        return await self.evaluation_nodes.answer_relevance_grader(state)

    # -------------------------------------------------------------------------
    # Router Functions
    # -------------------------------------------------------------------------
    input_query_security_router = staticmethod(input_query_security_router)
    query_router = staticmethod(query_router)
    grader_router = staticmethod(grader_router)
    hallucination_router = staticmethod(hallucination_router)
    answer_relevance_router = staticmethod(answer_relevance_router)