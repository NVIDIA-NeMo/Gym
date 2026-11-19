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

The loop: seed (the server restores the pivot and returns the harness's request) -> repeat {one chat completion on the
policy model server with the request's messages; ``/step`` with the reply, which returns the harness's next request}
until the window is done -> ``/verify``. A window is one chain: the server ends it when the harness's next request does
not extend the previous one plus its reply, so token capture sees one chain per rollout. Model calls go through
``url_path_for_run`` (the rollout's capture and sampling route).

Chat format on the Gym model server: the server's request model keeps OpenAI fields only, so ``top_k`` goes in
``metadata.extra_body`` and the chat template kwargs in ``metadata.chat_template_kwargs``; assistant history reasoning is
sent as a ``<think>...</think>`` content prefix, and a reply's ``<think>`` prefix is split back into
``reasoning_content`` (the Gym vLLM model server's reasoning format).
"""

import json
import re
from typing import Any

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models.token_capture.completion import completion_items
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
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


class ArcAgi3PivotRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class ArcAgi3PivotVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


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
    if message.get("reasoning_content"):
        out["content"] = f"<think>{message['reasoning_content']}</think>" + (message.get("content") or "")
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

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """One policy call (the Responses API route every Gym agent serves); a window is played through ``/run``."""
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name, url_path="/v1/responses", json=body, cookies=request.cookies
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def _post(self, server: str, path: str, payload: Any, cookies: dict[str, str]) -> dict[str, Any]:
        response = await self.server_client.post(server_name=server, url_path=path, json=payload, cookies=cookies)
        await raise_for_status(response)
        cookies.update({k: m.value for k, m in response.cookies.items()})
        return await get_response_json(response)

    async def run(self, request: Request, body: ArcAgi3PivotRunRequest) -> ArcAgi3PivotVerifyResponse:
        resources = self.config.resources_server.name
        env_cookies, model_cookies = dict(request.cookies), dict(request.cookies)
        seed = await self._post(resources, "/seed_session", body.model_dump(), env_cookies)
        messages: list[dict[str, Any]] = list(seed.get("messages") or [])
        tools, template = seed.get("tools") or [], seed.get("chat_template_kwargs") or {}
        agent_error = None
        # The previous call's prompt + generated token ids, when the model server returns them.
        held: list[int] | None = None
        # The last served model response id: the rollout's response carries it, so Gym's training-token capture
        # attributes the window to its last captured call (terminal attribution) and rebuilds that chain.
        last_response_id: str | None = None
        # Every call's served output items, in order: the rollout's output, which token capture matches call by call.
        served: list[dict[str, Any]] = []
        if seed.get("restore_error") is None:
            model_path = self.url_path_for_run("/v1/chat/completions", body)
            for _ in range(self.config.max_calls):
                try:
                    completion = await self._post(
                        self.config.model_server.name,
                        model_path,
                        self.chat_body(messages, tools, template),
                        model_cookies,
                    )
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
                # The window's cost counts the prompt prefix the previous call already holds as cached: measured on the
                # token ids when the model server returns them, else derived by the resources server from the counts.
                prompt_ids, gen_ids = message.get("prompt_token_ids"), message.get("generation_token_ids")
                cached = None
                if prompt_ids is not None and gen_ids is not None:
                    if held is not None:
                        cached = common_prefix(prompt_ids, held)
                    held = list(prompt_ids) + list(gen_ids)
                step = await self._post(
                    resources,
                    "/step",
                    {
                        "message": reply,
                        "finish_reason": choice.get("finish_reason") or "stop",
                        "gen_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "cached_tokens": cached,
                    },
                    env_cookies,
                )
                if step.get("request_messages") is not None:  # the harness's next request, as it sent it
                    messages = list(step["request_messages"])
                else:
                    messages.append(reply)
                    messages.extend(step["messages"])
                if step["done"]:
                    break
        response = NeMoGymResponse(
            id=last_response_id or f"resp_{body.verifier_metadata.get('pivot_id', 'pivot')}",
            created_at=0,
            model=self.config.model_server.name,
            object="response",
            output=served,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )
        verify = await self._post(
            resources,
            "/verify",
            body.model_dump() | {"response": response.model_dump(), "agent_error": agent_error},
            env_cookies,
        )
        return ArcAgi3PivotVerifyResponse.model_validate(verify)

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
