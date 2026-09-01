# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-tree agent-server for wrapping any LangChain `deepagents` graph as a native Gym `responses_api_agents` server.

Generic base for wrapping any LangChain `deepagents` graph as a native Gym `responses_api_agents` server
(model calls go through Gym's own `model_server`, on-policy, instead of the agent bringing its own model).
Concrete instances subclass `DeepAgentsAgent` and implement `build_agent(model)` — see
`reasoning_search_agent.py` for the one used by this repo's reasoning_gym/tavily_search comparison.

Model calls cannot go through `langchain_openai.ChatOpenAI`: it runs on the `openai` SDK, which runs on
`httpx`, and CLAUDE.md requires all async HTTP inside a Gym server process to go through Gym's own
aiohttp-backed `server_client` instead (httpx/httpcore's connection pooling degrades badly at high
concurrency). `GymResponsesChatModel` below reimplements the minimum LangChain model-calling surface
deepagents needs (`bind_tools()` + `_agenerate()`) on top of `server_client.post()`.

The compiled deepagents graph is built once, in `__init__`, and stored as `self.agent` — not per-request,
and named to match how `deepagents`' own docs/examples name the object `create_deep_agent()` returns
(`agent = create_deep_agent(...)`), rather than `langgraph_agent`'s `self.graph`. `GymResponsesChatModel`
is a singleton bound to the owning agent, and gets its per-request `rollout_id`/cookies from a
`ContextVar` set once per request in `responses()`, rather than being rebuilt per request. This is safe
under concurrent in-flight requests: `ContextVar` values are asyncio-task-local.
"""

import json
import time
import uuid
from abc import abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from fastapi import Body, Request, Response
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status, rollout_path_prefix
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


_request_context: ContextVar[dict] = ContextVar("langchain_deepagents_agent_request_context")


class DeepAgentsAgentConfig(SimpleAgentConfig):
    """No tool-specific fields — those belong on a concrete subclass's config (see reasoning_search_agent.py).

    `max_steps` (inherited from SimpleAgentConfig) is unused: deepagents runs its own internal tool loop
    and answers in one call, the same as remote_agent's `max_steps: 1` in
    examples/langchain_deepagent/configs/config_reasoning_gym.yaml.
    """


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _output_text(content_items: list) -> str:
    """Extract text from a NeMoGymResponseOutputMessage.content list — real Pydantic objects, not dicts
    (unlike the input-side content lists _text() handles), so it needs its own extraction, not _text()."""
    return "".join(part.text for part in content_items if isinstance(part, NeMoGymResponseOutputText))


def to_langchain(input_items: list) -> list:
    """Gym Responses API input items -> LangChain messages. Runs once per responses() call, on the 
    first request that comes in. This is different from to_gym_input()/to_langchain_ai_message(), 
    which run once per internal model call inside GymResponsesChatModel._agenerate()."""
    # "developer" is an OpenAI Responses API role with no LangChain equivalent; map it to SystemMessage.
    roles = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "developer": SystemMessage}
    # Only "message" items are kept — function_call/function_call_output items in input_items are
    # silently dropped. Currently safe because deepagents' tools run internally (params["tools"] is
    # ignored, see below) so no function_call items ever appear in input_items today. Would need
    # handling if this agent is ever wired to a resources-server-mediated tool loop or fed
    # pre-seeded multi-turn tool-call history.
    return [
        roles.get(item.role, HumanMessage)(content=_text(item.content))
        for item in input_items
        if getattr(item, "type", None) == "message"
    ]


def to_responses(new_messages: list, model_name: str) -> dict:
    final_text = next((m.content for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content), "")
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "created_at": time.time(),
        "model": model_name,
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": _text(final_text), "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }


def to_gym_input(messages: list) -> list[dict]:
    """LangChain messages -> Gym Responses API input items, for one internal deepagents model call."""
    items: list[dict] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _text(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            if message.content:
                items.append({"type": "message", "role": "assistant", "content": _text(message.content)})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    }
                )
        elif isinstance(message, SystemMessage):
            items.append({"type": "message", "role": "system", "content": _text(message.content)})
        else:
            items.append({"type": "message", "role": "user", "content": _text(message.content)})
    return items


def to_langchain_ai_message(gym_response: NeMoGymResponse) -> AIMessage:
    """Gym Responses API output -> a single LangChain AIMessage, for one internal deepagents model call."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in gym_response.output:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            text_parts.append(_output_text(item.content))
        elif item_type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                # Malformed args from the model server. Pass through as {} rather than crashing the
                # request — deepagents' underlying ToolNode (handle_tool_errors=True by default) will
                # fail the tool invocation against the real schema and feed the model an error
                # ToolMessage, same graceful-degradation outcome as simple_agent's explicit guard.
                args = {}
            tool_calls.append(tool_call(name=item.name, args=args, id=item.call_id))
    return AIMessage(
        content="".join(text_parts),
        tool_calls=tool_calls,
        id=gym_response.id,
        response_metadata={"id": gym_response.id},
    )


class GymResponsesChatModel(BaseChatModel):
    """LangChain-compatible chat model backed by Gym's own aiohttp-based server_client — see module
    docstring for why ChatOpenAI/langchain_openai can't be used here. A singleton per DeepAgentsAgent
    instance; per-request rollout_id/cookies come from `_request_context`, not constructor fields."""

    agent: Any = Field(default=None, exclude=True)  # the owning DeepAgentsAgent instance

    @property
    def _llm_type(self) -> str:
        return "gym-responses-api"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("GymResponsesChatModel is async-only; use ainvoke/astream_events.")

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # Gym's FunctionToolParam (nemo_gym/openai_utils.py, a passthrough of the real OpenAI Responses
        # API type) requires "strict" as a present-but-nullable field — omitting it entirely fails
        # request validation ("Field required"), confirmed by actually running this against a live
        # model server. convert_to_openai_tool()'s "function" dict doesn't include it by default.
        formatted = [
            {"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools
        ]
        bind_kwargs: dict[str, Any] = {"tools": formatted}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        ctx = _request_context.get()
        url_path = f"{rollout_path_prefix(ctx['rollout_id'])}/v1/responses"
        request_body: dict[str, Any] = {"input": to_gym_input(messages)}
        if "tools" in kwargs:
            request_body["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            request_body["tool_choice"] = kwargs["tool_choice"]
        resp = await self.agent.server_client.post(
            server_name=self.agent.config.model_server.name,
            url_path=url_path,
            json=request_body,
            cookies=ctx["cookies"],
        )
        await raise_for_status(resp)
        gym_response = NeMoGymResponse.model_validate(await get_response_json(resp))
        return ChatResult(generations=[ChatGeneration(message=to_langchain_ai_message(gym_response))])


class DeepAgentsAgent(SimpleAgent):
    """Inherits SimpleAgent.run() unchanged (seed_session -> self-POST /v1/responses -> verify). Overrides
    responses() with everything true of any deepagents-based agent wired into Gym.

    Does not build a TrajectoryRecord (per-tool-call/model-call observability) — see README.md's "Known
    limitation" section. That's deliberately deferred, not an oversight."""

    config: DeepAgentsAgentConfig
    agent: Any = Field(default=None, exclude=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = self.build_agent(GymResponsesChatModel(agent=self))

    @abstractmethod
    def build_agent(self, model: BaseChatModel) -> Any:
        """Return a compiled deepagents graph, e.g. create_deep_agent(model=model, tools=[...], ...).
        Called once, from __init__ — not per-request."""
        raise NotImplementedError

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # Different servers have different session-middleware keys (get_session_middleware_key() is
        # f"{class_name}___{server_name}"), so a cookie from the resources server's /seed_session does not
        # automatically ride through to /verify unless re-set explicitly here.
        for key, value in request.cookies.items():
            response.set_cookie(key, value)

        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None

        if isinstance(body.input, str):
            input_items = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        else:
            input_items = body.input
        input_messages = to_langchain(input_items)

        token = _request_context.set({"rollout_id": rollout_id, "cookies": request.cookies})
        try:
            final_state = await self.agent.ainvoke({"messages": input_messages})
        finally:
            _request_context.reset(token)

        new_messages = final_state["messages"][len(input_messages) :]
        return NeMoGymResponse.model_validate(to_responses(new_messages, self.config.model_server.name))
