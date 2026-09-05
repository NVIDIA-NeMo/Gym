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
concurrency). `GymResponsesChatModel` below reimplements the minimum LangChain model-calling surface
deepagents needs (`bind_tools()` + `_agenerate()`) on top of `server_client.post()`.

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

Cookies specifically evolve within one rollout: `_agenerate()` writes each response's cookies into a
mutable holder dict referenced from `configurable["model_cookies"]`, created fresh per request in
`responses()`, so a multi-turn rollout chains model-server session state instead of resending the same
cookies on every turn — without mutating the `RunnableConfig` itself (which nested/parallel branches may
share by reference; only this one designated holder is mutable, everything else in `configurable` is not).
"""

import json
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import ensure_config
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputText
from nemo_gym.server_utils import get_response_json, raise_for_status


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
    messages: list = []
    for item in input_items:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            messages.append(roles.get(item.role, HumanMessage)(content=_text(item.content)))
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
            messages.append(ToolMessage(content=_text(item.output), tool_call_id=item.call_id))
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
    instance; per-request model_url_path/cookies come from the ambient `RunnableConfig`, not constructor
    fields."""

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
        formatted = [{"type": "function", "strict": False, **convert_to_openai_tool(t)["function"]} for t in tools]
        bind_kwargs: dict[str, Any] = {"tools": formatted}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
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
        except KeyError as e:
            raise RuntimeError(
                "GymResponsesChatModel called without model_url_path/model_cookies in "
                "RunnableConfig['configurable'] — it must be invoked via DeepAgentsAgent.responses(), "
                "which sets these once per request."
            ) from e

        request_body: dict[str, Any] = {"input": to_gym_input(messages)}
        if "tools" in kwargs:
            request_body["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            request_body["tool_choice"] = kwargs["tool_choice"]
        resp = await self.agent.server_client.post(
            server_name=self.agent.config.model_server.name,
            url_path=model_url_path,
            json=request_body,
            cookies=model_cookies["cookies"],
        )
        await raise_for_status(resp)
        # `model_cookies` is the same holder dict object referenced from `configurable` (not a copy), so
        # this mutation is visible to the next `_agenerate()` call for this rollout — matches
        # `simple_agent`'s pattern of chaining model-server cookies from each response into the next call.
        # It's a deliberate, narrow exception to treating `configurable` as otherwise immutable.
        model_cookies["cookies"] = resp.cookies
        gym_response = NeMoGymResponse.model_validate(await get_response_json(resp))
        return ChatResult(generations=[ChatGeneration(message=to_langchain_ai_message(gym_response))])
