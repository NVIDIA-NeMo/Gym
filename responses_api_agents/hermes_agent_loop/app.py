# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import sys
import types
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


def _ensure_hermes_on_path() -> None:
    try:
        import model_tools  # noqa: F401
    except ImportError:
        hermes_root = Path(__file__).resolve().parent.parent.parent.parent / "hermes-agent"
        if hermes_root.exists() and str(hermes_root) not in sys.path:
            sys.path.insert(0, str(hermes_root))


def _to_namespace(obj):
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(i) for i in obj]
    return obj


class GymChatCompletionServer:
    """Wraps Gym's /v1/chat/completions for use as a HermesAgentLoop server."""

    def __init__(self, server_client, model_server_name: str, cookies):
        self._client = server_client
        self._server_name = model_server_name
        self.cookies = cookies

    async def chat_completion(self, **kwargs) -> Any:
        resp = await self._client.post(
            server_name=self._server_name,
            url_path="/v1/chat/completions",
            json=kwargs,
            cookies=self.cookies,
        )
        await raise_for_status(resp)
        self.cookies = resp.cookies
        return _to_namespace(await get_response_json(resp))


def _input_to_messages(input_items) -> list[dict]:
    messages = []
    for item in input_items:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        messages.append({"role": role, "content": content or ""})
    return messages


def _output_messages_to_items(messages: list[dict], n_input: int) -> list:
    output_items = []

    for msg in messages[n_input:]:
        role = msg.get("role")

        if role == "assistant":
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls") or []

            if reasoning:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[NeMoGymSummary(text=reasoning, type="summary_text")],
                    )
                )

            if content or not tool_calls:
                base = NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                )
                gen_ids = msg.get("generation_token_ids")
                if gen_ids:
                    train_cls = RESPONSES_TO_TRAIN[NeMoGymResponseOutputMessage]
                    output_items.append(
                        train_cls(
                            **base.model_dump(),
                            prompt_token_ids=msg.get("prompt_token_ids") or [],
                            generation_token_ids=gen_ids,
                            generation_log_probs=msg.get("generation_log_probs") or [],
                        )
                    )
                else:
                    output_items.append(base)

            for tc in tool_calls:
                call_id = tc.get("id") or tc.get("call_id") or f"call_{uuid4().hex[:8]}"
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                        call_id=call_id,
                        id=call_id,
                        status="completed",
                    )
                )

        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=msg.get("tool_call_id", ""),
                    output=msg.get("content", ""),
                    status="completed",
                )
            )

    return output_items


class HermesAgentLoopConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 32
    max_turns: int = 30
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    terminal_backend: str = "local"
    terminal_timeout: int = 120
    tool_pool_size: int = 128
    temperature: float = 1.0
    system_prompt: Optional[str] = None


class HermesLoopRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesLoopVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class HermesAgentLoopAgent(SimpleResponsesAPIAgent):
    config: HermesAgentLoopConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        _ensure_hermes_on_path()
        self.sem = Semaphore(self.config.concurrency)
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)

        from environments.agent_loop import resize_tool_pool

        resize_tool_pool(self.config.tool_pool_size)

    def _load_tools(self):
        from model_tools import get_tool_definitions

        schemas = (
            get_tool_definitions(
                enabled_toolsets=self.config.enabled_toolsets,
                disabled_toolsets=self.config.disabled_toolsets,
                quiet_mode=True,
            )
            or []
        )
        names = {t["function"]["name"] for t in schemas}
        return schemas, names

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        from environments.agent_loop import HermesAgentLoop

        tool_schemas, valid_tool_names = self._load_tools()

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        if self.config.system_prompt:
            first_item = body.input[0] if body.input else None
            first_role = getattr(first_item, "role", None) or (
                first_item.get("role") if isinstance(first_item, dict) else None
            )
            if first_role != "system":
                body.input = [NeMoGymEasyInputMessage(role="system", content=self.config.system_prompt)] + list(
                    body.input
                )

        messages = _input_to_messages(body.input)
        n_input = len(messages)

        server = GymChatCompletionServer(
            server_client=self.server_client,
            model_server_name=self.config.model_server.name,
            cookies=request.cookies,
        )

        result = await HermesAgentLoop(
            server=server,
            tool_schemas=tool_schemas,
            valid_tool_names=valid_tool_names,
            max_turns=self.config.max_turns,
            temperature=self.config.temperature,
        ).run(messages)

        for k, v in server.cookies.items():
            response.set_cookie(k, v)

        output_items = _output_messages_to_items(result.messages, n_input)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=str(self.config.model_server.name),
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def run(self, request: Request, body: HermesLoopRunRequest) -> HermesLoopVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return HermesLoopVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    HermesAgentLoopAgent.run_webserver()
