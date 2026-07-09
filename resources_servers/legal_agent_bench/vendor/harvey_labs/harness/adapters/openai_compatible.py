# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""OpenAI-compatible Chat Completions adapter.

This is intended for self-hosted endpoints such as vLLM or other servers that
implement /v1/chat/completions with tool calls. It is separate from
OpenAIAdapter, which uses OpenAI's Responses API.
"""

from __future__ import annotations

import time
from typing import Any

import openai

from harness.adapters.base import ModelAdapter, ModelResponse, ToolCall


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible Chat Completions endpoints."""

    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
        reasoning_effort: str | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        timeout_seconds: float | None = None,
        omit_temperature: bool | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        top_p: float | None = None,
    ):
        super().__init__(model, temperature, reasoning_effort)
        if not base_url:
            raise ValueError("OpenAI-compatible adapter requires a base_url")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.timeout_seconds = timeout_seconds
        self.omit_temperature = self._should_omit_temperature(model) if omit_temperature is None else omit_temperature
        self.chat_template_kwargs = chat_template_kwargs
        self.top_p = top_p
        api_key = api_key or "EMPTY"
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": 0,
        }
        if timeout_seconds is not None:
            client_kwargs["timeout"] = timeout_seconds
        self.client = openai.OpenAI(**client_kwargs)

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [self._translate_tool(t) for t in tools],
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
        }
        if not self.omit_temperature:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": self.chat_template_kwargs,
            }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = self._create_with_retries(kwargs)
        choice = response.choices[0]
        msg = choice.message

        content = getattr(msg, "content", None) or ""
        tool_calls = []
        message_tool_calls = []

        for tc in getattr(msg, "tool_calls", None) or []:
            function = getattr(tc, "function", None)
            name = getattr(function, "name", "") if function else ""
            arguments = getattr(function, "arguments", "{}") if function else "{}"
            tool_call_id = getattr(tc, "id", None) or f"call_{len(tool_calls)}"
            tool_calls.append(ToolCall(id=tool_call_id, name=name, arguments=arguments or "{}"))
            message_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": name,
                        "arguments": arguments or "{}",
                    },
                }
            )

        message = {
            "role": "assistant",
            "content": content,
        }
        if message_tool_calls:
            message["tool_calls"] = message_tool_calls

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return ModelResponse(
            message=message,
            tool_calls=tool_calls,
            text=content,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
        )

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        return [
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }
            for tool_call_id, result in results
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    def _create_with_retries(self, kwargs: dict[str, Any]):
        """Call chat completions with explicit transient-error retries."""
        attempt = 0
        while True:
            try:
                return self._create_once(kwargs)
            except Exception as exc:
                if not self._is_retryable_error(exc) or attempt >= self.max_retries:
                    raise
                attempt += 1
                delay = self._retry_delay(attempt)
                total_attempts = self.max_retries + 1
                print(
                    "OpenAI-compatible agent request failed with "
                    f"{self._error_label(exc)}; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{total_attempts})"
                )
                time.sleep(delay)

    def _create_once(self, kwargs: dict[str, Any]):
        return self.client.chat.completions.create(**kwargs)

    def _retry_delay(self, attempt: int) -> float:
        return min(60.0, self.retry_backoff_seconds * (2 ** (attempt - 1)))

    def _is_retryable_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in self.RETRYABLE_STATUS_CODES:
            return True
        return exc.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "TimeoutError",
        }

    def _error_label(self, exc: Exception) -> str:
        status_code = getattr(exc, "status_code", None)
        if status_code:
            return f"HTTP {status_code}"
        return exc.__class__.__name__

    def _should_omit_temperature(self, model: str) -> bool:
        """NVIDIA-hosted Bedrock Claude models reject temperature."""
        model_lower = model.lower()
        return (
            model_lower.startswith("aws/anthropic/")
            or model_lower.startswith("nvidia/aws/anthropic/")
            or "anthropic/bedrock-claude" in model_lower
        )

    def _translate_tool(self, tool: dict) -> dict:
        """Translate canonical tool definition to Chat Completions format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
