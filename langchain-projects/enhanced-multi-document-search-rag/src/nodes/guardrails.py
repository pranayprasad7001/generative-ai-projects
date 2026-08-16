import re
import logging
from contextvars import ContextVar
from typing import Any, Callable, Awaitable
from config.mcp_config import MCPToolManager
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.types import Command

from prompts.rag_prompts import (
    QUERY_SECURITY_SYSTEM_PROMPT, 
    OUTPUT_ANSWER_SECURITY_SYSTEM_PROMPT, 
    EXTERNAL_SEARCH_SYSTEM_PROMPT
)

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    PIIMiddleware,
    ToolCallRequest,
    hook_config,
)


logger = logging.getLogger(__name__)

# Request-scoped tool calls counter context variable
_tool_calls_counter_ctx: ContextVar[int] = ContextVar("tool_calls_counter_ctx", default=0)

PII_PATTERNS = {
    "api_key": re.compile(r"\b(?:sk-(?:proj-)?[a-zA-Z0-9_-]{20,100}|AIza[0-9A-Za-z-_]{35}|gsk_[a-zA-Z0-9]{48,64}|sk-ant-[a-zA-Z0-9_-]{20,100}|ghp_[a-zA-Z0-9]{36})\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "phone_number": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
}


def sanitize_pii(text: str) -> str:
    """
    Deterministically sanitize/redact PII (emails, phone numbers, credit cards, SSNs, API keys, IPs)
    from text to ensure sensitive data is not propagated through state.question or downstream nodes.
    """
    if not text or not isinstance(text, str):
        return text
    sanitized = text
    sanitized = PII_PATTERNS["api_key"].sub("[REDACTED_API_KEY]", sanitized)
    sanitized = PII_PATTERNS["email"].sub("[REDACTED_EMAIL]", sanitized)
    sanitized = PII_PATTERNS["ssn"].sub("[REDACTED_SSN]", sanitized)
    sanitized = PII_PATTERNS["credit_card"].sub("[REDACTED_CREDIT_CARD]", sanitized)
    sanitized = PII_PATTERNS["phone_number"].sub("[REDACTED_PHONE]", sanitized)
    sanitized = PII_PATTERNS["ip_address"].sub("[REDACTED_IP]", sanitized)
    return sanitized


class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail that blocks requests containing banned keywords.

    Runs before the security agent processes the request, so blocked
    requests do not incur an LLM call.
    """

    def __init__(self, banned_keywords: list[str], max_tool_calls: int = 3):
        super().__init__()
        self.banned_keywords = [
            keyword.lower()
            for keyword in banned_keywords
        ]
        self.max_tool_calls = max_tool_calls

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self,
        state: AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        # Reset request-scoped tool calls counter for this agent execution
        _tool_calls_counter_ctx.set(0)

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

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept and validate tool arguments and output in sync context."""
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args", {})
        
        current_calls = _tool_calls_counter_ctx.get() + 1
        _tool_calls_counter_ctx.set(current_calls)
        if current_calls > self.max_tool_calls:
            logger.warning(
                f"Maximum tool limit ({self.max_tool_calls}) reached. Forcing external search agent to finalize answer."
            )
            return ToolMessage(
                content=f"Maximum tool limit ({self.max_tool_calls}) reached. Do not call any further tools. Synthesize your final response immediately.",
                tool_call_id=request.tool_call["id"],
            )

        for val in args.values():
            if isinstance(val, str):
                for keyword in self.banned_keywords:
                    if keyword in val.lower():
                        logger.warning(
                            f"Security middleware blocked tool call to '{tool_name}' containing banned keyword: '{keyword}'"
                        )
                        return ToolMessage(
                            content="Error: Query contains blocked terms.",
                            tool_call_id=request.tool_call["id"],
                        )

        try:
            result = handler(request)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {str(e)}")
            return ToolMessage(
                content="Error: The external tool failed to retrieve information.",
                tool_call_id=request.tool_call["id"],
            )

        if isinstance(result, ToolMessage) and result.content:
            content_str = result.content if isinstance(result.content, str) else str(result.content)
            content = content_str.lower()
            for keyword in self.banned_keywords:
                if keyword in content:
                    logger.warning(
                        f"Redacted banned content containing keyword '{keyword}' from tool '{tool_name}' output."
                    )
                    return ToolMessage(
                        content="[Banned or unsafe content redacted from search results]",
                        id=result.id,
                        name=result.name,
                        tool_call_id=result.tool_call_id,
                    )

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept and validate tool arguments and output in async context."""
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args", {})
        
        current_calls = _tool_calls_counter_ctx.get() + 1
        _tool_calls_counter_ctx.set(current_calls)
        if current_calls > self.max_tool_calls:
            logger.warning(
                f"Maximum tool limit ({self.max_tool_calls}) reached. Forcing external search agent to finalize answer."
            )
            return ToolMessage(
                content=f"Maximum tool limit ({self.max_tool_calls}) reached. Do not call any further tools. Synthesize your final response immediately.",
                tool_call_id=request.tool_call["id"],
            )

        for val in args.values():
            if isinstance(val, str):
                for keyword in self.banned_keywords:
                    if keyword in val.lower():
                        logger.warning(
                            f"Security middleware blocked tool call to '{tool_name}' containing banned keyword: '{keyword}'"
                        )
                        return ToolMessage(
                            content="Error: Query contains blocked terms.",
                            tool_call_id=request.tool_call["id"],
                        )

        try:
            result = await handler(request)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {str(e)}")
            return ToolMessage(
                content="Error: The external tool failed to retrieve information.",
                tool_call_id=request.tool_call["id"],
            )

        if isinstance(result, ToolMessage) and result.content:
            content_str = result.content if isinstance(result.content, str) else str(result.content)
            content = content_str.lower()
            for keyword in self.banned_keywords:
                if keyword in content:
                    logger.warning(
                        f"Redacted banned content containing keyword '{keyword}' from tool '{tool_name}' output."
                    )
                    return ToolMessage(
                        content="[Banned or unsafe content redacted from search results]",
                        id=result.id,
                        name=result.name,
                        tool_call_id=result.tool_call_id,
                    )

        return result

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
        "ignore all previous instructions",
        "ignore previous instructions",
        "disregard all previous instructions",
        "disregard previous instructions",
        "forget all previous instructions",
        "override system instructions",
        "override safety guidelines",
        "reveal system prompt",
        "reveal your instructions",
        "dan mode enabled",
        "jailbreak prompt",
        "unrestricted ai mode",
        "synthesize biological weapons",
        "manufacture explosives at home",
    ]

    BLOCKED_PATTERNS = [
        "password=",
        "password:",
        "api_key=",
        "api_key:",
        "client_secret=",
        "client_secret:",
        "private_key=",
        "private_key:",
        "db_password=",
        "db_password:",
        "bearer eyj",
        "-----begin rsa private key-----",
        "-----begin openssh private key-----",
        "-----begin private key-----",
    ]

    def __init__(self, llm_checker, llm_generator=None):
        self.llm_checker = llm_checker
        self.llm_generator = llm_generator if llm_generator is not None else llm_checker
        # Backward compatibility alias
        self.llm = self.llm_checker
        self.input_guardrail_agent = None
        self.output_guardrail_agent = None
        self.combined_guardrail_agent = None
        self.mcp_manager = MCPToolManager()

    def _get_common_pii_middleware(self) -> list:
        """Helper to return fresh instances of common PII middleware."""
        return [
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            ),
            PIIMiddleware(
                "credit_card",
                strategy="mask",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            ),
            PIIMiddleware(
                "api_key",
                detector=r"\b(?:sk-(?:proj-)?[a-zA-Z0-9_-]{20,100}|AIza[0-9A-Za-z-_]{35}|gsk_[a-zA-Z0-9]{48,64}|sk-ant-[a-zA-Z0-9_-]{20,100}|ghp_[a-zA-Z0-9]{36})\b",
                strategy="block",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            ),
            PIIMiddleware(
                "phone_number",
                detector=r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                strategy="mask",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            ),
            PIIMiddleware(
                "ip_address",
                detector=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                strategy="mask",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            ),
            PIIMiddleware(
                "ssn",
                detector=r"\b\d{3}-\d{2}-\d{4}\b",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=True
            ),
        ]

    def _build_input_guardrail_agent(self):
        """Build input guardrail agent"""
        logger.debug("Building input guardrail agent with PII and keyword filtering.")
        self.input_guardrail_agent = create_agent(
            model=self.llm_checker,
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
            model=self.llm_checker,
            system_prompt=OUTPUT_ANSWER_SECURITY_SYSTEM_PROMPT,
            middleware=self._get_common_pii_middleware() + [
                SafetyGuardrailMiddleware(
                    blocked_patterns=self.BLOCKED_PATTERNS
                ),
            ],
        )

    async def _build_combined_guardrail_agent(self):
        """Build combined guardrail agent"""
        logger.debug("Building combined guardrail agent with PII and keyword filtering.")
        tools = await self.mcp_manager.get_tools()
        cached_llm = self.llm_generator.bind(
            extra_body={"cache": {"use-cache": True, "ttl": 1800}}
        )
        self.combined_guardrail_agent = create_agent(
            model=cached_llm,
            tools=tools,
            system_prompt=EXTERNAL_SEARCH_SYSTEM_PROMPT,
            middleware=self._get_common_pii_middleware() + [
                ContentFilterMiddleware(
                    banned_keywords=self.BANNED_KEYWORDS
                ),
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

    async def get_combined_guardrail_agent(self):
        """Get combined guardrail agent."""
        if self.combined_guardrail_agent is None:
            await self._build_combined_guardrail_agent()
        return self.combined_guardrail_agent