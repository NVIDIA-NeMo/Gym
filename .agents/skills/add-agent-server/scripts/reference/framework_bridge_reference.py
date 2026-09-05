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
"""REFERENCE ONLY — not a runnable server, not imported by anything.

The pattern for wrapping an external framework that OWNS its own model-calling loop, annotated with the
rules from ../../references/correctness-checklist.md and ../../references/wrapping-external-frameworks.md.

The example uses LangChain's BaseChatModel because that is what is in-tree today
(responses_api_agents/langchain_deepagents_agent/), but the shape generalizes: any framework that calls
the model for you exposes some model interface, and that interface is your only intervention point.

If you write the loop or the graph nodes yourself, do NOT use this pattern — see
native_loop_agent_reference.py. The machinery below is the cost of a framework owning the loop, not a
general requirement for using LangChain (langgraph_agent uses LangChain-adjacent tooling with none of it).

In a real server this content is split across two files: `app.py` for the agent class, and a
package-local bridge module for the model adapter and converters. Reviewers ask for that split.
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

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


# Per-request state for a model object that is built once and shared across concurrent requests.
# ContextVar values are asyncio-task-local, so concurrent in-flight rollouts cannot leak into each
# other. Write a test that proves this: two staggered concurrent calls, each asserting its own
# rollout id.
_request_context: ContextVar[dict] = ContextVar("reference_agent_request_context")


# ---------------------------------------------------------------------------------------------------
# Conversions. Four of them, in two directions, at two different frequencies.
# ---------------------------------------------------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def to_langchain(input_items: list) -> list:
    """Gym input items -> framework messages. Once per request.

    RULE 3: handle all three item types. Filtering to `type == "message"` looks correct for fresh
    prompts (a fresh prompt IS all messages) and silently corrupts every replayed or continued
    trajectory, which legitimately carries function_call / function_call_output items.
    """
    # "developer" is a Responses API role with no framework equivalent; map it to the system role.
    roles = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "developer": SystemMessage}
    messages: list = []
    for item in input_items:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            messages.append(roles.get(item.role, HumanMessage)(content=_text(item.content)))
        elif item_type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}  # degrade, don't crash the request
            messages.append(AIMessage(content="", tool_calls=[tool_call(name=item.name, args=args, id=item.call_id)]))
        elif item_type == "function_call_output":
            messages.append(ToolMessage(content=_text(item.output), tool_call_id=item.call_id))
    return messages


def to_gym_input(messages: list) -> list[dict]:
    """Framework messages -> Gym input items. Once per internal model call."""
    items: list[dict] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            items.append(
                {"type": "function_call_output", "call_id": message.tool_call_id, "output": _text(message.content)}
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


def to_responses(new_messages: list, model_name: str) -> dict:
    """Framework messages -> a Gym response. Once per request.

    RULE 4: emit the whole trace. Taking only the last assistant message discards every tool call and
    tool result permanently, for every rollout.

    Most of to_gym_input()'s output is reusable here: NeMoGymResponseFunctionToolCall and
    NeMoGymFunctionCallOutput are literally the same pydantic classes on the input and output sides.
    `message` is the exception — the output-side NeMoGymResponseOutputMessage additionally requires an
    `id` and output_text-shaped `content`, so those items get patched.
    """
    output_items = []
    for item in to_gym_input(new_messages):
        if item["type"] == "message":
            item = {
                **item,
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "content": [{"type": "output_text", "text": item["content"], "annotations": []}],
            }
        output_items.append(item)
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "created_at": time.time(),
        "model": model_name,
        "object": "response",
        "output": output_items,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }


def to_framework_message(gym_response: NeMoGymResponse) -> AIMessage:
    """Gym response -> a single framework message. Once per internal model call."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in gym_response.output:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            text_parts.append(
                "".join(part.text for part in item.content if getattr(part, "type", "") == "output_text")
            )
        elif item_type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                # The framework's tool layer will fail this against the real schema and feed the model
                # an error result — same graceful degradation, one layer down.
                args = {}
            tool_calls.append(tool_call(name=item.name, args=args, id=item.call_id))
    return AIMessage(content="".join(text_parts), tool_calls=tool_calls, id=gym_response.id)


# ---------------------------------------------------------------------------------------------------
# The model adapter: the framework's model interface, backed by Gym's server_client.
# ---------------------------------------------------------------------------------------------------


class GymBackedChatModel(BaseChatModel):
    """RULE 5: this class exists because the framework's own client cannot be used.
    langchain_openai.ChatOpenAI runs on the openai SDK, which runs on httpx, and CLAUDE.md requires all
    async HTTP in a Gym server process to go through Gym's aiohttp-backed server_client instead
    (httpx/httpcore connection pooling is O(n^2) and hangs at high concurrency).

    A singleton bound to the owning agent, constructed once in the agent's __init__. Per-request state
    comes from _request_context, not constructor fields.
    """

    agent: Any = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "gym-responses-api"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("Gym agent servers are async-only; use ainvoke/astream_events.")

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # RULE 6: two transformations, both required. Un-nest the "function" dict to the top level, and
        # add a present-but-nullable "strict" key — Gym's FunctionToolParam requires it and LangChain's
        # converter does not emit it. Omitting it fails as a 422 from the model server, not as a
        # framework-layer error. Lock this in with a test.
        formatted = [{"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools]
        bind_kwargs: dict[str, Any] = {"tools": formatted}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        ctx = _request_context.get()
        request_body: dict[str, Any] = {"input": to_gym_input(messages)}
        for key in ("tools", "tool_choice"):
            if key in kwargs:
                request_body[key] = kwargs[key]

        resp = await self.agent.server_client.post(
            server_name=self.agent.config.model_server.name,
            # RULE 1: the path was resolved in responses() from the inbound request and passed through
            # the context. Do NOT rebuild it here — this code cannot see the inbound request, so it
            # cannot detect capture mode, and hand-building the prefix is how capture silently breaks.
            url_path=ctx["model_url_path"],
            cookies=ctx["cookies"],
            json=request_body,
        )
        await raise_for_status(resp)

        # RULE 2: chain model-server cookies forward across the framework's internal turns. `ctx` is the
        # same dict object stored in the ContextVar (not a copy), so mutating it is visible to the next
        # call in this task without a redundant .set().
        ctx["cookies"] = resp.cookies

        gym_response = NeMoGymResponse.model_validate(await get_response_json(resp))
        return ChatResult(generations=[ChatGeneration(message=to_framework_message(gym_response))])


# ---------------------------------------------------------------------------------------------------
# The agent server.
# ---------------------------------------------------------------------------------------------------


class FrameworkAgentConfig(SimpleAgentConfig):
    """Framework-specific fields belong on a concrete subclass's config, not here.

    Prefer a config field over a module constant — it lets each combo YAML set a different value
    (e.g. a system_prompt one benchmark's verifier requires and another's does not) without code changes.
    """


class FrameworkAgent(SimpleAgent):
    """Inherits SimpleAgent.run() unchanged (seed_session -> self-POST /v1/responses -> verify) and
    overrides only responses()."""

    config: FrameworkAgentConfig
    agent: Any = Field(default=None, exclude=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build once, here — not per request. This is why the model needs a ContextVar for per-request
        # state: it becomes a singleton shared across concurrent requests.
        self.agent = self.build_agent(GymBackedChatModel(agent=self))

    @abstractmethod
    def build_agent(self, model: BaseChatModel) -> Any:
        """Return the compiled framework agent, e.g. create_deep_agent(model=model, tools=[...])."""
        raise NotImplementedError

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # RULE 2: session-middleware keys are per-server (f"{class_name}___{server_name}"), so the
        # resources server's /seed_session cookie does not reach /verify unless re-set explicitly.
        for key, value in request.cookies.items():
            response.set_cookie(key, value)

        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None

        input_items = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
        )
        input_messages = to_langchain(input_items)

        # RULE 1: resolve the capture-aware path HERE, where the inbound request is visible.
        model_url_path = self.url_path_for_request("/v1/responses", request)

        token = _request_context.set(
            {"rollout_id": rollout_id, "cookies": request.cookies, "model_url_path": model_url_path}
        )
        try:
            final_state = await self.agent.ainvoke({"messages": input_messages})
        finally:
            _request_context.reset(token)

        # Slice off the echoed-back input; everything after it is what the agent produced this call.
        new_messages = final_state["messages"][len(input_messages) :]
        return NeMoGymResponse.model_validate(to_responses(new_messages, self.config.model_server.name))
