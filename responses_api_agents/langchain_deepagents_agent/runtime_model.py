# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responses transport used inside the sandbox, with one aiohttp pool per invocation."""

import asyncio
from typing import Any

import aiohttp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from responses_api_agents.langchain_deepagents_agent.runtime_messages import from_response, to_gym_input
from responses_api_agents.langchain_deepagents_agent.runtime_observer import DeepAgentsObserver


class ModelRequestError(RuntimeError):
    """An upstream model request failed after the bounded retry policy."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        super().__init__(f"Model Server HTTP {status}: {body}")


class SandboxResponsesModel(BaseChatModel):
    """Preserve DeepAgents' native loop while routing every call through Gym capture."""

    http: Any = Field(exclude=True)
    observer: DeepAgentsObserver = Field(exclude=True)
    url: str
    request_defaults: dict[str, Any] = Field(default_factory=dict)
    assistant_message_header: str

    @property
    def _llm_type(self) -> str:
        return "gym-responses-api"

    def _generate(
        self, messages: list, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        raise NotImplementedError("Sandbox DeepAgents uses async model calls")

    def bind_tools(self, tools: list, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        formatted = [{"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools]
        return self.bind(
            tools=formatted, **({"tool_choice": tool_choice} if tool_choice is not None else {}), **kwargs
        )

    async def _agenerate(
        self, messages: list, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        if stop:
            raise ValueError("Responses transport does not support stop sequences")
        request = {**self.request_defaults, "input": to_gym_input(messages), "stream": False}
        for name in ("tools", "tool_choice"):
            if name in kwargs:
                request[name] = kwargs[name]
        owner, turn = self.observer.begin_model(run_id=run_manager.run_id, request=request)
        headers = {"x-session-id": owner, self.assistant_message_header: str(run_manager.run_id)}
        for attempt in range(3):
            async with self.http.post(self.url, json=request, headers=headers) as response:
                if response.status < 400:
                    body = await response.json()
                    break
                error = ModelRequestError(response.status, await response.text())
                if response.status not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise error
                delay = response.headers.get("Retry-After", "")
                await asyncio.sleep(min(float(delay), 30) if delay.replace(".", "", 1).isdigit() else 2**attempt)
        self.observer.end_model(owner=owner, turn=turn, response=body)
        return ChatResult(generations=[ChatGeneration(message=from_response(body))])


def model_http_session() -> aiohttp.ClientSession:
    """Use one shared transport for all policy, child-agent and summary requests."""
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), cookie_jar=aiohttp.CookieJar(unsafe=True))
