import logging
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

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept and validate tool arguments and output in sync context."""
        tool_name = request.tool_call.get("name")
        args = request.tool_call.get("args", {})

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
                        f"Redacted banned content from tool '{tool_name}' output."
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
                        f"Redacted banned content from tool '{tool_name}' output."
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
                detector=r"sk-[a-zA-Z0-9]{32}",
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

    async def _build_combined_guardrail_agent(self):
        """Build combined guardrail agent"""
        logger.debug("Building combined guardrail agent with PII and keyword filtering.")
        tools = await self.mcp_manager.get_tools()
        self.combined_guardrail_agent = create_agent(
            model=self.llm,
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