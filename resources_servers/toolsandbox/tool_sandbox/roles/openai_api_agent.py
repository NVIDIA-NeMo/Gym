# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Agent role for any OpenAI-compatible chat completion endpoint.

The agent is configured at construction time via an :class:`OpenAIRoleConfig`
holding base_url, model, api_key and optional sampling parameters
(temperature, top_p, max_tokens, enable_thinking). Sampling fields default to
``None`` and are omitted from the request when unset so the server's own
defaults apply. ``enable_thinking`` toggles vLLM's
``chat_template_kwargs.enable_thinking`` via ``extra_body`` for reasoning
models that honour the kwarg (Qwen3, Nemotron, Gemma-thinking, …).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from logging import getLogger
from typing import Any, Iterable, List, Literal, Optional, Union, cast

from openai import (
    NOT_GIVEN,
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotGiven,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    sanitize_tool_call_id,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole

LOGGER = getLogger(__name__)

# Transient OpenAI-SDK errors we retry. Permanent client errors
# (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError,
# UnprocessableEntityError, ConflictError) are intentionally NOT retried — they
# won't fix themselves and we shouldn't waste budget hammering the endpoint.
_RETRIABLE_OPENAI_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

openai_retry = retry(
    reraise=True,
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=1, max=8) + wait_random(0, 0.5),
    retry=retry_if_exception_type(_RETRIABLE_OPENAI_ERRORS),
    before_sleep=before_sleep_log(LOGGER, logging.WARNING),
)


@dataclass(frozen=True)
class OpenAIRoleConfig:
    """Configuration for an OpenAI-compatible role (agent or user simulator).

    ``api_key`` may be an empty string for endpoints that accept anonymous
    access; the SDK still requires a non-empty value, so a placeholder is
    substituted when building the client.

    Sampling fields default to ``None``; ``None`` means "do not send the
    parameter so the server-side default is used".
    """

    base_url: str
    model: str
    api_key: str = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_thinking: Optional[bool] = None


def _is_openai_reasoning_model(model: str) -> bool:
    """Heuristic: does the model accept top-level ``reasoning_effort``?

    Matches OpenAI reasoning families (o-series, gpt-5+) with or without an
    ``openai/`` provider prefix. Other model ids fall back to the vLLM-style
    ``chat_template_kwargs.enable_thinking`` mechanism.
    """
    m = model.lower().rsplit("/", maxsplit=1)[-1]
    if m.startswith(("o1", "o3", "o4", "o5")):
        return True
    if m.startswith("gpt-5") or m.startswith("gpt-6"):
        return True
    return False


def _sampling_kwargs(config: OpenAIRoleConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    is_openai = _is_openai_reasoning_model(config.model)
    if config.max_tokens is not None:
        if is_openai:
            kwargs["max_completion_tokens"] = config.max_tokens
        else:
            kwargs["max_tokens"] = config.max_tokens
    if config.enable_thinking is not None:
        if is_openai:
            # OpenAI/GPT-style: top-level reasoning_effort. OpenAI rejects
            # unknown body params, so do NOT send chat_template_kwargs here.
            kwargs["reasoning_effort"] = "high" if config.enable_thinking else "low"
        else:
            # vLLM/Qwen-style toggle via extra_body.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": config.enable_thinking}
            }
    return kwargs


class OpenAIAPIAgent(BaseRole):
    """Tool-calling agent against any OpenAI-compatible chat endpoint."""

    role_type: RoleType = RoleType.AGENT

    def __init__(self, config: OpenAIRoleConfig) -> None:
        self._config = config
        self.model_name = config.model
        self.openai_client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-used",
        )
        self._sampling = _sampling_kwargs(config)

    async def teardown(self) -> None:
        await self.openai_client.close()

    async def respond(self, ending_index: Optional[int] = None) -> None:
        """Read the message log and respond with one or more agent messages.

        Either replies to the user in natural language or emits one or more
        tool calls to the execution environment.
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return

        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven]
        if messages[-1].sender in (RoleType.USER, RoleType.EXECUTION_ENVIRONMENT):
            openai_tools = cast(
                Iterable[ChatCompletionToolParam],
                convert_to_openai_tools(available_tools),
            )
        else:
            openai_tools = NOT_GIVEN

        openai_messages, _ = to_openai_messages(messages)
        LOGGER.debug("Agent model input (last msg): %s", openai_messages[-1])
        response = await self._model_inference(openai_messages, openai_tools)
        openai_response_message = response.choices[0].message

        response_messages: List[Message] = []
        if not openai_response_message.tool_calls:
            assert openai_response_message.content is not None
            response_messages.append(
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            )
        else:
            assert openai_tools is not NOT_GIVEN
            current_context = get_current_context()
            for tool_call in openai_response_message.tool_calls:
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                sanitized_id = sanitize_tool_call_id(tool_call.id)
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=sanitized_id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @openai_retry
    async def _model_inference(
        self,
        openai_messages: list[
            dict[Literal["role", "content", "tool_call_id", "name", "tool_calls"], Any]
        ],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Call the configured model and return the raw chat completion."""
        with all_logging_disabled(logging.INFO):
            return await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,
                **self._sampling,
            )
