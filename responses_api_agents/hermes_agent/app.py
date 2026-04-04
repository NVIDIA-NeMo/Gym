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

import asyncio
import concurrent.futures
import json
import logging
import os
import sys
from asyncio import Semaphore
from pathlib import Path
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
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)

# handle_function_call() is synchronous and some backends call asyncio.run()
# internally, which deadlocks inside a running event loop. Run in a thread pool.
_TOOL_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_tool_executor(max_workers: int) -> concurrent.futures.ThreadPoolExecutor:
    global _TOOL_EXECUTOR
    if _TOOL_EXECUTOR is None:
        _TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return _TOOL_EXECUTOR


def _ensure_hermes_on_path() -> None:
    try:
        import model_tools  # noqa: F401
    except ImportError:
        hermes_root = Path(__file__).resolve().parent.parent.parent.parent / "hermes-agent"
        if hermes_root.exists() and str(hermes_root) not in sys.path:
            sys.path.insert(0, str(hermes_root))


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 32
    max_turns: int = 30
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    terminal_backend: str = "local"
    terminal_timeout: int = 120
    tool_pool_size: int = 128
    system_prompt: Optional[str] = None


class HermesRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class HermesAgent(SimpleResponsesAPIAgent):
    config: HermesAgentConfig
    sem: Semaphore = None
    _executor: concurrent.futures.ThreadPoolExecutor = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        _ensure_hermes_on_path()
        self.sem = Semaphore(self.config.concurrency)
        self._executor = _get_tool_executor(self.config.tool_pool_size)
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)

    def _load_tools(self):
        from model_tools import get_tool_definitions

        # Hermes uses chat-completions format: {"type": "function", "function": {name, ...}}
        # Gym's Responses API ToolParam is flat: {"type": "function", "name": ..., ...}
        cc_schemas = (
            get_tool_definitions(
                enabled_toolsets=self.config.enabled_toolsets,
                disabled_toolsets=self.config.disabled_toolsets,
                quiet_mode=True,
            )
            or []
        )
        names = {t["function"]["name"] for t in cc_schemas}
        schemas = [{"type": "function", "strict": None, **t["function"]} for t in cc_schemas]
        return schemas, names

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        from model_tools import handle_function_call

        tool_schemas, valid_tool_names = self._load_tools()
        task_id = str(uuid4())
        loop = asyncio.get_running_loop()

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

        _user_task: Optional[str] = None
        for msg in body.input:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if role == "user" and isinstance(content, str) and content.strip():
                _user_task = content.strip()[:500]
                break

        if tool_schemas and not body.tools:
            body = body.model_copy(update={"tools": tool_schemas})

        accumulated_outputs = []
        usage = None
        model_cookies = None

        for turn in range(self.config.max_turns):
            turn_body = body.model_copy(update={"input": list(body.input) + accumulated_outputs})

            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=turn_body,
                cookies=model_cookies,
            )
            await raise_for_status(model_resp)
            model_resp_json = await get_response_json(model_resp)
            model_cookies = model_resp.cookies

            gym_response = NeMoGymResponse.model_validate(model_resp_json)

            if gym_response.usage:
                if usage is None:
                    usage = gym_response.usage
                else:
                    usage.input_tokens += gym_response.usage.input_tokens
                    usage.output_tokens += gym_response.usage.output_tokens
                    usage.total_tokens += gym_response.usage.total_tokens

            fn_calls = [o for o in gym_response.output if o.type == "function_call"]
            accumulated_outputs.extend(gym_response.output)

            if not fn_calls:
                break

            for fn_call in fn_calls:
                tool_name = fn_call.name
                tool_args_raw = fn_call.arguments

                if tool_name not in valid_tool_names:
                    tool_result = json.dumps(
                        {"error": f"Unknown tool '{tool_name}'. Available: {sorted(valid_tool_names)}"}
                    )
                    LOG.warning("Model called unknown tool '%s' on turn %d", tool_name, turn + 1)
                else:
                    try:
                        args = json.loads(tool_args_raw) if tool_args_raw else {}
                    except json.JSONDecodeError as e:
                        tool_result = json.dumps({"error": f"Invalid JSON in arguments: {e}"})
                        LOG.warning("Invalid JSON for tool '%s': %s", tool_name, tool_args_raw[:200])
                        args = None

                    if args is not None:
                        try:
                            _tn, _ta, _tid = tool_name, args, task_id
                            tool_result = await loop.run_in_executor(
                                self._executor,
                                lambda: handle_function_call(_tn, _ta, task_id=_tid, user_task=_user_task),
                            )
                        except Exception as exc:
                            tool_result = json.dumps({"error": f"Tool execution failed: {type(exc).__name__}: {exc}"})
                            LOG.error("Tool '%s' failed on turn %d: %s", tool_name, turn + 1, exc)

                accumulated_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=fn_call.call_id,
                        output=tool_result,
                        status="completed",
                    )
                )

        if model_cookies:
            for k, v in model_cookies.items():
                response.set_cookie(k, v)

        gym_response.output = accumulated_outputs
        gym_response.usage = usage
        return gym_response

    async def run(self, request: Request, body: HermesRunRequest) -> HermesVerifyResponse:
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

            return HermesVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    HermesAgent.run_webserver()
