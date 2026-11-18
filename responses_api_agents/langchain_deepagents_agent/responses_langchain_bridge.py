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
"""The LangChain <-> Gym Responses API bridge used by `DeepAgentsAgent` (see app.py).

Model calls cannot go through `langchain_openai.ChatOpenAI`: it runs on the `openai` SDK, which runs on
`httpx`, and CLAUDE.md requires all async HTTP inside a Gym server process to go through Gym's own
aiohttp-backed `server_client` instead (httpx/httpcore's connection pooling degrades badly at high
concurrency). `GymResponsesChatModel` below sends the requests of `ResponsesChatModel`
(responses_chat_model.py), which implements the minimum LangChain model-calling surface deepagents needs
(`bind_tools()` + `_agenerate()`), through `server_client.post()`. The sandbox runner sends the same requests from
the task sandbox instead.

`GymResponsesChatModel` is a singleton bound to the owning agent (constructed once, in the owning
`DeepAgentsAgent.__init__`), and gets its per-request `model_url_path`/cookies from the ambient
`RunnableConfig`, set once per request by `DeepAgentsAgent.responses()` via
`self.agent.ainvoke(..., config={"configurable": {...}})`, rather than being rebuilt per request.
`_agenerate()` reads it back with `ensure_config()` — LangChain's own public helper for exactly this
"ambient config for a call I don't control the calling convention of" case (it's what backs LangChain's
`var_child_runnable_config` ContextVar internally). Using `RunnableConfig` instead of a Gym-specific
ContextVar means nested/subagent model calls get `model_url_path`/cookies for free: `deepagents`' own
`task`/`atask` subagent tools already rely on `configurable` merging automatically into subagent
invocations (see `deepagents/middleware/subagents.py`), so no extra plumbing is needed for that case.

Cookies and usage evolve within one rollout: `_agenerate()` writes them into mutable holder dicts referenced
from `configurable["model_cookies"]` and `configurable["model_usage"]`, created fresh per request in
`responses()`. The cookie holder chains model-server session state across turns, while the usage holder
accumulates token counts from every internal model call. This avoids replacing the `RunnableConfig` itself,
which nested/parallel branches may share by reference.
"""

import json
import time
import uuid
from typing import Any

from aiohttp import ClientResponseError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import ensure_config
from pydantic import Field

from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.core import NeMoGymResponse, accumulate_response_usage
from responses_api_agents.langchain_deepagents_agent.responses_chat_model import (
    ResponsesChatModel,
    ai_message_from_response,
    message_text,
    to_gym_input,
)


def to_langchain(input_items: list) -> list:
    """Gym Responses API input items -> LangChain messages: one input item -> one LangChain message,
    over a whole (possibly replayed) prior conversation. Runs once per responses() call, seeding
    self.agent.ainvoke({"messages": ...}).

    Not the same job as to_langchain_ai_message() below, despite the similar name. That function merges
    every output item from a *single* model turn (one message + N function_calls + reasoning) into the
    fields of one AIMessage, because LangChain's BaseChatModel contract allows exactly one AIMessage per
    generation — there's no "list of sibling output items" concept on that side for a single turn, and it
    runs once per internal model call inside GymResponsesChatModel._agenerate(). to_langchain() (this
    function) has no such constraint: it's converting a whole conversation, not one turn, so each item
    becomes its own message and the list can be any length."""
    # "developer" is an OpenAI Responses API role with no LangChain equivalent; map it to SystemMessage.
    roles = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "developer": SystemMessage}
    messages: list = []
    for item in input_items:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            messages.append(roles.get(item.role, HumanMessage)(content=message_text(item.content)))
        elif item_type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                # Malformed args from the caller. Same graceful-degradation choice as
                # to_langchain_ai_message()'s identical guard on the model-output side: pass through {}
                # rather than crashing the request.
                args = {}
            messages.append(AIMessage(content="", tool_calls=[tool_call(name=item.name, args=args, id=item.call_id)]))
        elif item_type == "function_call_output":
            messages.append(ToolMessage(content=message_text(item.output), tool_call_id=item.call_id))
    return messages


def to_responses(new_messages: list, model_name: str) -> dict:
    """LangChain messages -> a Gym Responses API response dict, preserving the full tool-call/tool-result
    trace (not just the final assistant text). Reuses to_gym_input()'s message-type mapping: its
    function_call/function_call_output dicts already satisfy NeMoGymResponseOutputItem's required fields
    (NeMoGymResponseFunctionToolCall and NeMoGymFunctionCallOutput are the same classes on the input and
    output sides), but its message dicts need an `id` and output_text-shaped `content` added, since
    NeMoGymResponseOutputMessage (output-only) requires both and to_gym_input()'s input-side message dicts
    don't carry them."""
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


def to_langchain_ai_message(gym_response: NeMoGymResponse) -> AIMessage:
    """Gym Responses API output -> a single LangChain AIMessage, for one internal deepagents model call."""
    return ai_message_from_response(gym_response.model_dump(mode="json"))


class GymResponsesChatModel(ResponsesChatModel):
    """LangChain-compatible chat model backed by Gym's own aiohttp-based server_client — see module
    docstring for why ChatOpenAI/langchain_openai can't be used here. A singleton per DeepAgentsAgent
    instance; per-request model_url_path/cookies come from the ambient `RunnableConfig`, not constructor
    fields."""

    agent: Any = Field(default=None, exclude=True)  # the owning DeepAgentsAgent instance

    async def create_response(self, request_body: dict[str, Any]) -> dict[str, Any]:
        # `BaseChatModel.ainvoke()` doesn't forward `configurable` into `_agenerate()`'s kwargs (it only
        # extracts callbacks/tags/metadata/run_name/run_id before calling `agenerate_prompt`), so we pull
        # the ambient config ourselves. `ensure_config()` with no argument reads whatever `RunnableConfig`
        # is currently in scope (LangChain's own `var_child_runnable_config` ContextVar) — this is what
        # `DeepAgentsAgent.responses()` set at the top of the request, and what deepagents' subagent tools
        # merge into automatically for nested calls.
        configurable = ensure_config().get("configurable") or {}
        try:
            model_url_path = configurable["model_url_path"]
            model_cookies = configurable["model_cookies"]
            model_usage = configurable["model_usage"]
        except KeyError as e:
            raise RuntimeError(
                "GymResponsesChatModel called without model_url_path/model_cookies/model_usage in "
                "RunnableConfig['configurable'] — it must be invoked via DeepAgentsAgent.responses(), "
                "which sets these once per request."
            ) from e
        # Optional (unlike the three above): most callers never set body.reasoning, so this is absent from
        # older/simpler configurable dicts (e.g. hand-built test fixtures) rather than required wiring.
        model_reasoning = configurable.get("model_reasoning")
        if model_reasoning is not None:
            request_body["reasoning"] = model_reasoning
        # The activation request's sampling controls, such as temperature and top_p, apply to every model call.
        for name, value in (configurable.get("model_request_defaults") or {}).items():
            request_body.setdefault(name, value)
        resp = await self.agent.server_client.post(
            server_name=self.agent.config.model_server.name,
            url_path=model_url_path,
            json=request_body,
            cookies=model_cookies["cookies"],
        )
        try:
            await raise_for_status(resp)
        except ClientResponseError as error:
            # Marks the error as a model call's, so a session can type it apart from a tool's HTTP error.
            model_failure = configurable.get("model_failure")
            if model_failure is not None:
                model_failure["error"] = error
            raise
        # `model_cookies` is the same holder dict object referenced from `configurable` (not a copy), so
        # this mutation is visible to the next `_agenerate()` call for this rollout — matches
        # `simple_agent`'s pattern of chaining model-server cookies from each response into the next call.
        # It's a deliberate, narrow exception to treating `configurable` as otherwise immutable.
        model_cookies["cookies"] = resp.cookies
        gym_response = NeMoGymResponse.model_validate(await get_response_json(resp))
        # Missing usage in any call makes the whole rollout unknown, including later successful calls.
        if gym_response.usage is None:
            model_usage["unknown"] = True
        model_usage["usage"] = (
            None if model_usage.get("unknown") else accumulate_response_usage(model_usage["usage"], gym_response.usage)
        )
        return gym_response.model_dump(mode="json")
