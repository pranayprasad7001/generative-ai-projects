import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from prompts.rag_prompts import QUERY_SECURITY_SYSTEM_PROMPT, OUTPUT_ANSWER_SECURITY_SYSTEM_PROMPT
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    PIIMiddleware,
    hook_config,
)

logger = logging.getLogger(__name__)

class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail that blocks requests containing banned keywords.

    Runs before the security agent processes the request, so blocked
    requests do not incur an LLM call.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [
            keyword.lower()
            for keyword in banned_keywords
        ]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self,
        state: AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:

        if not state["messages"]:
            return None

        first_message = state["messages"][0]

        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                logger.warning(
                    f"Security middleware blocked keyword: '{keyword}'"
                )

                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": (
                                "BLOCKED: The request contains inappropriate "
                                "or potentially unsafe content. "
                                "Please rephrase your request."
                            ),
                        }
                    ],
                    "jump_to": "end",
                }

        return None

class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Deterministic post-processing safety guardrail.

    Runs after the output guardrail agent's LLM call.
    Never makes another LLM call.
    """

    def __init__(self, blocked_patterns: list[str]):
        super().__init__()

        self.blocked_patterns = [
            pattern.lower()
            for pattern in blocked_patterns
        ]

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self,
        state: AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:

        if not state["messages"]:
            return None

        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            return None

        content = last_message.content.lower()

        for pattern in self.blocked_patterns:

            if pattern in content:

                logger.warning(
                    f"Output middleware blocked pattern: '{pattern}'"
                )

                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": (
                                "I'm unable to provide that response "
                                "because it contains sensitive information."
                            ),
                        }
                    ],
                    "jump_to": "end",
                }

        return None

class Guardrails:

    BANNED_KEYWORDS = [
        "hack",
        "exploit",
        "malware",
        "jailbreak",
        "bypass",
        "ddos",
        "ransomware",
        "phishing",
        "trojan",
        "keylogger",
        "rootkit",
        "sql injection",
        "xss",
        "botnet",
        "bomb",
        "explosive",
        "terrorist",
        "illegal drugs",
        "ignore previous instructions",
        "system prompt",
        "override safety",
        "reveal prompt",
        "reveal instructions",
    ]

    BLOCKED_PATTERNS = [
        "api_key",
        "password",
        "secret",
        "access_token",
        "private_key",
        "bearer token",
        "client_secret",
        "ssh key",
        "connection string",
        "db_password",
        "database password",
    ]

    def __init__(self, llm):
        self.llm = llm
        self.input_guardrail_agent = None
        self.output_guardrail_agent = None

    def _get_common_pii_middleware(self) -> list:
        """Helper to return fresh instances of common PII middleware."""
        return [
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "credit_card",
                strategy="mask",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",
                strategy="block",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "phone_number",
                detector=r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                strategy="mask",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "ip_address",
                detector=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                strategy="mask",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "ssn",
                detector=r"\b\d{3}-\d{2}-\d{4}\b",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            ),
        ]

    def _build_input_guardrail_agent(self):
        """Build input guardrail agent"""
        logger.debug("Building input guardrail agent with PII and keyword filtering.")
        self.input_guardrail_agent = create_agent(
            model=self.llm,
            system_prompt=QUERY_SECURITY_SYSTEM_PROMPT,
            middleware=self._get_common_pii_middleware() + [
                ContentFilterMiddleware(
                    banned_keywords=self.BANNED_KEYWORDS
                ),
            ],
        )

    def _build_output_guardrail_agent(self):
        """Build output guardrail agent"""
        logger.debug("Building output guardrail agent with PII and pattern filtering.")
        self.output_guardrail_agent = create_agent(
            model=self.llm,
            system_prompt=OUTPUT_ANSWER_SECURITY_SYSTEM_PROMPT,
            middleware=self._get_common_pii_middleware() + [
                SafetyGuardrailMiddleware(
                    blocked_patterns=self.BLOCKED_PATTERNS
                ),
            ],
        )
    
    def get_input_guardrail_agent(self):
        """Get input guardrail agent"""
        if self.input_guardrail_agent is None:
            self._build_input_guardrail_agent()
        return self.input_guardrail_agent

    def get_output_guardrail_agent(self):
        """Get output guardrail agent"""
        if self.output_guardrail_agent is None:
            self._build_output_guardrail_agent()
        return self.output_guardrail_agent