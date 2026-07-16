# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Translate the LAB harness protocol to Gym's Chat Completions API."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientSession, ClientTimeout

from harness.adapters.base import ModelAdapter, ModelResponse, ToolCall


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for the local Gym model server selected by the Harbor agent."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
        reasoning_effort: str | None = None,
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
        self.timeout_seconds = timeout_seconds
        self.omit_temperature = self._should_omit_temperature(model) if omit_temperature is None else omit_temperature
        self.chat_template_kwargs = chat_template_kwargs
        self.top_p = top_p

    async def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
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

        response = await self._create_chat_completion(kwargs)
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("Gym model server returned no chat completion choices")
        msg = choices[0].get("message") or {}

        content = msg.get("content") or ""
        tool_calls = []
        message_tool_calls = []

        for tc in msg.get("tool_calls") or []:
            function = tc.get("function") or {}
            name = function.get("name", "")
            arguments = function.get("arguments", "{}")
            tool_call_id = tc.get("id") or f"call_{len(tool_calls)}"
            tool_calls.append(ToolCall(id=tool_call_id, name=name, arguments=arguments or "{}"))
            message_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": tc.get("type", "function"),
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

        usage = response.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return ModelResponse(
            message=message,
            tool_calls=tool_calls,
            text=content,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
        )

    async def _create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the selected Gym model server without loading config in the Ray worker."""
        timeout = ClientTimeout(total=self.timeout_seconds)
        headers = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}"}
        async with ClientSession(timeout=timeout, headers=headers) as client:
            async with client.post(self._chat_completions_endpoint(), json=payload) as response:
                response.raise_for_status()
                return await response.json()

    def _chat_completions_endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

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
