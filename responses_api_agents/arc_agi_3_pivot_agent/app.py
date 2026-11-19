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
"""ARC-AGI-3 pivot agent: plays one pivot window of ``resources_servers/arc_agi_3_pivot``.

An environment server owns the episode (``single_agent_turn_legacy`` in the configs): it seeds the resources session
(the server restores the pivot), seeds this agent's session with a direct HTTP grant to that resources server, runs
one activation (``/v1/responses``), verifies, and closes both sessions on every path, so the window is released also
when the activation fails or is cancelled. Closing the agent session cancels an activation still running.

The activation: ``/window`` (the harness's pivot request) -> repeat {one chat completion on the policy model server
with the request's messages; ``/step`` with the reply, which returns the harness's next request} until the window is
done. A window is one chain: the server ends it when the harness's next request does not extend the previous one plus
its reply, so token capture sees one chain per rollout. Model calls go through the activation's capture and sampling
route. A failed model call ends the activation with the window unfinished, which the server masks (the error is in the
response's ``metadata.agent_error``).

Chat format on the Gym model server: the server's request model keeps OpenAI fields only, so ``top_k`` goes in
``metadata.extra_body`` and the chat template kwargs in ``metadata.chat_template_kwargs``; assistant history reasoning is
sent as a ``<think>...</think>`` content prefix, and a reply's ``<think>`` prefix is split back into
``reasoning_content`` (the Gym vLLM model server's reasoning format).
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from fastapi import Body, Request, Response
from pydantic import Field

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, rollout_context
from nemo_gym.models.token_capture.completion import completion_items
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class ArcAgi3PivotAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Sampling of the recorded pivot requests (the harness's own); a training framework's per-rollout values replace
    # temperature / top_p / the output cap through the model URL.
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    top_k: int | None = 20
    max_tokens: int = 12288
    # Safety net only: the window's own rules (the server's max_calls and action budget) end it first.
    max_calls: int = Field(64, ge=1)


@dataclass
class PivotSessionState(AgentSessionState):
    """One episode's window: the grant to its resources session, and the activation playing it."""

    tool_access: DirectHTTPToolAccess
    resources_cookies: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False
    execution: asyncio.Task | None = None


def openai_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls with the OpenAI fields only (recorded pivots carry engine extras, e.g. a streaming ``index``, that the
    Gym model server's request model refuses; chat templates do not render them)."""
    return [
        {
            "id": c.get("id"),
            "type": c.get("type") or "function",
            "function": {
                "name": (c.get("function") or {}).get("name"),
                "arguments": (c.get("function") or {}).get("arguments"),
            },
        }
        for c in calls or []
    ]


def to_gym_message(message: dict[str, Any]) -> dict[str, Any]:
    """An assistant history message with its reasoning as a ``<think>`` content prefix."""
    if message.get("role") != "assistant":
        return message
    out = {k: v for k, v in message.items() if k not in ("reasoning_content", "reasoning")}
    if message.get("tool_calls"):
        out["tool_calls"] = openai_tool_calls(message["tool_calls"])
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        out["content"] = f"<think>{reasoning}</think>" + (message.get("content") or "")
    return out


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def from_gym_reply(message: dict[str, Any]) -> dict[str, Any]:
    """A reply with its reasoning in ``reasoning_content`` (split from a ``<think>`` prefix when the server sent one)."""
    out = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": openai_tool_calls(message.get("tool_calls")),
    }
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    content = out["content"]
    if not reasoning and isinstance(content, str) and THINK.search(content):
        reasoning, out["content"] = "".join(THINK.findall(content)), THINK.sub("", content)
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


class ArcAgi3PivotAgent(SimpleResponsesAPIAgent):
    config: ArcAgi3PivotAgentConfig

    def chat_body(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], template: dict[str, Any]):
        c = self.config
        body: dict[str, Any] = {
            "messages": [to_gym_message(m) for m in messages],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": c.max_tokens,
        }
        if c.temperature is not None:
            body["temperature"] = c.temperature
        if c.top_p is not None:
            body["top_p"] = c.top_p
        metadata = {}
        if c.top_k is not None:
            metadata["extra_body"] = json.dumps({"top_k": c.top_k})
        if template:
            metadata["chat_template_kwargs"] = json.dumps(template)
        if metadata:
            body["metadata"] = metadata
        return body

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> PivotSessionState:
        direct = [a for a in self.effective_tool_accesses(body) if isinstance(a, DirectHTTPToolAccess)]
        if len(direct) != 1:
            raise AgentSeedRejectedError(
                "the pivot agent plays its window through one direct HTTP grant to its arc_agi_3_pivot resources "
                "session (resources_tool_transports: [direct_http])"
            )
        return PivotSessionState(request=body, tool_access=direct[0], resources_cookies=dict(direct[0].cookies))

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, PivotSessionState)
        async with state.lock:
            state.closing = True
            execution = state.execution
        if execution is not None and not execution.done():
            execution.cancel()  # no model call or step after close; the environment server releases the window
        if execution is not None:
            await asyncio.gather(execution, return_exceptions=True)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, resources_cookies=dict(state.resources_cookies)
        )

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Play the session's window (an agent session's activation); without a session, one policy call."""
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name, url_path="/v1/responses", json=body, cookies=request.cookies
            )
            await raise_for_status(model_response)
            return NeMoGymResponse.model_validate(await get_response_json(model_response))
        state = self._require_agent_session(session_id)
        assert isinstance(state, PivotSessionState)
        model_path = self.url_path_for_request("/v1/chat/completions", request)
        async with state.lock:
            if state.closing:
                raise AgentSessionConflictError("Agent session is closing")
            if state.execution is None:
                state.execution = asyncio.create_task(self._play(state, model_path))
            execution = state.execution
        # A dropped request does not stop the window; closing the session does.
        return await asyncio.shield(execution)

    async def _resources(self, state: PivotSessionState, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await http_request(
            method="POST",
            url=f"{str(state.tool_access.base_url).rstrip('/')}{path}",
            json=payload,
            cookies=state.resources_cookies,
            headers=dict(state.tool_access.headers),
            _internal=True,
        )
        await raise_for_status(response)
        state.resources_cookies.update({k: m.value for k, m in response.cookies.items()})
        return await get_response_json(response)

    async def _play(self, state: PivotSessionState, model_path: str) -> NeMoGymResponse:
        with rollout_context(state.request.episode_id.capture_key):
            window = await self._resources(state, "/window", {})
            messages: list[dict[str, Any]] = list(window.get("messages") or [])
            tools, template = window.get("tools") or [], window.get("chat_template_kwargs") or {}
            agent_error = None
            model_cookies: dict[str, str] = {}
            # The previous call's prompt + generated token ids, when the model server returns them.
            held: list[int] | None = None
            # The last served model response id: the rollout's response carries it, so Gym's training-token capture
            # attributes the window to its last captured call (terminal attribution) and rebuilds that chain.
            last_response_id: str | None = None
            # Every call's served output items, in order: the rollout's output, which token capture matches call by
            # call.
            served: list[dict[str, Any]] = []
            if window.get("restore_error") is None:
                for _ in range(self.config.max_calls):
                    try:
                        model_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=model_path,
                            json=self.chat_body(messages, tools, template),
                            cookies=model_cookies,
                        )
                        await raise_for_status(model_response)
                        model_cookies.update({k: m.value for k, m in model_response.cookies.items()})
                        completion = await get_response_json(model_response)
                    except Exception as exc:  # the window cannot go on; the rollout is masked, never scored as a loss
                        agent_error = f"{type(exc).__name__}: {exc}"
                        break
                    last_response_id = completion.get("id") or last_response_id
                    # capture fingerprints messages by content and drops their ids; the response model needs one
                    for item in completion_items(completion):
                        if item.get("type") == "message" and not item.get("id"):
                            item["id"] = f"msg_{last_response_id}_{len(served)}"
                        served.append(item)
                    choice = (completion.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    reply = from_gym_reply(message)
                    usage = completion.get("usage") or {}
                    # The window's cost counts the prompt prefix the previous call already holds as cached: measured
                    # on the token ids when the model server returns them, else derived by the resources server.
                    prompt_ids, gen_ids = message.get("prompt_token_ids"), message.get("generation_token_ids")
                    cached = None
                    if prompt_ids is not None and gen_ids is not None:
                        if held is not None:
                            cached = common_prefix(prompt_ids, held)
                        held = list(prompt_ids) + list(gen_ids)
                    step = await self._resources(
                        state,
                        "/step",
                        {
                            "message": reply,
                            "finish_reason": choice.get("finish_reason") or "stop",
                            "gen_tokens": usage.get("completion_tokens"),
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "cached_tokens": cached,
                        },
                    )
                    if step.get("request_messages") is not None:  # the harness's next request, as it sent it
                        messages = list(step["request_messages"])
                    else:
                        messages.append(reply)
                        messages.extend(step["messages"])
                    if step["done"]:
                        break
        pivot_id = (state.request.task_id.task_id or "pivot").replace("/", "_")
        return NeMoGymResponse(
            id=last_response_id or f"resp_{pivot_id}",
            created_at=0,
            model=self.config.model_server.name,
            object="response",
            output=served,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
            metadata={"agent_error": agent_error} if agent_error else None,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    ArcAgi3PivotAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ArcAgi3PivotAgent.run_webserver()  # noqa: F401
