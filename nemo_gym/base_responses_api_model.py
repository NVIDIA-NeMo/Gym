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
import inspect
from abc import abstractmethod

from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()

import inspect
import json
import os
import re
import tempfile
from abc import abstractmethod
from typing import Any, Callable, Dict, List

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import TypeAdapter

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.global_config import NEMO_GYM_LOG_DIR_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
)
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


# stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()

_parse_output_item = TypeAdapter(NeMoGymResponseOutputItem).validate_python


class RunTrajectory:
    def __init__(self, base_dir: str) -> None:
        self._dir = os.path.join(base_dir, "token_id_buffer")
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, token: str) -> str:
        return os.path.join(self._dir, f"{token}.trajectory.jsonl")

    def append(self, token: str, items: List[NeMoGymResponseOutputItem]) -> None:
        if not items:
            return
        with open(self._path(token), "a") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

    def read(self, token: str) -> List[NeMoGymResponseOutputItem]:
        try:
            with open(self._path(token)) as f:
                return [_parse_output_item(json.loads(line)) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def pop(self, token: str) -> List[NeMoGymResponseOutputItem]:
        items = self.read(token)
        try:
            os.remove(self._path(token))
        except FileNotFoundError:
            pass
        return items


def _longest_chain(prompts: List, fulls: List) -> List[int]:
    n = len(prompts)
    if not n:
        return []
    memo: Dict[int, List[int]] = {}

    # prefix merge, memo dfs for retries
    def from_i(i: int) -> List[int]:
        if i in memo:
            return memo[i]
        best: List[int] = []
        fi = fulls[i]
        if fi is not None:
            for j in range(i + 1, n):
                pj = prompts[j]
                if pj is not None and len(pj) >= len(fi) and pj[: len(fi)] == fi:
                    c = from_i(j)
                    if len(c) > len(best) or (len(c) == len(best) and best and c[-1] > best[-1]):
                        best = c
        memo[i] = [i, *best]
        return memo[i]

    # first request = main chain
    first = next((p for p in prompts if p is not None), None)

    # maybe theres a retry of turn 1
    # parallel first model calls though weird and unlikely might break this
    seeds = [i for i in range(n) if prompts[i] == first] or [0]

    # seeds usually shouldnt really happen
    return max((from_i(i) for i in seeds), key=lambda c: (len(c), c[-1]))


def _main_chain(items: List[NeMoGymResponseOutputItem]) -> List[NeMoGymResponseOutputItem]:
    """wrap _longest_chain"""
    groups: List[List[NeMoGymResponseOutputItem]] = []
    cur: List[NeMoGymResponseOutputItem] = []
    for it in items:
        cur.append(it)
        if getattr(it, "prompt_token_ids", None) is not None:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    if len(groups) <= 1:
        return items
    prompts, fulls = [], []
    for g in groups:
        last = g[-1] if getattr(g[-1], "prompt_token_ids", None) is not None else None
        p = list(last.prompt_token_ids) if last is not None else None
        ge = list(getattr(last, "generation_token_ids", None) or []) if last is not None else None
        prompts.append(p)
        fulls.append((p + ge) if p is not None else None)
    out: List[NeMoGymResponseOutputItem] = []
    for i in _longest_chain(prompts, fulls):
        out.extend(groups[i])
    return out


class TokenIDBufferingMixin:
    def token_id_buffer_dir(self) -> str:
        log_dir = (self.server_client.global_config_dict or {}).get(NEMO_GYM_LOG_DIR_KEY_NAME)
        return str(log_dir) if log_dir else tempfile.mkdtemp(prefix="nemo_gym_token_id_buffer_")

    def setup_token_id_buffering(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def _extract_run_token(request: Request, call_next):
            # vllm has no stateful session api, cookies doesnt survive
            match = re.compile(r"^/runs/([^/]+)(/.*)$").match(request.url.path)
            if match:
                request.scope["path"] = match.group(2)
                request.scope["raw_path"] = match.group(2).encode()
            request.state.run_token = match.group(1) if match else None
            return await call_next(request)

        app.get("/trajectory")(self._pop_trajectory)

    def buffer_turn(
        self,
        request: Request,
        request_messages: List[Dict[str, Any]],
        message_dict: Dict[str, Any],
        build_assistant_items: Callable[[Dict[str, Any]], List[NeMoGymResponseOutputItem]],
    ) -> None:
        token = getattr(request.state, "run_token", None)
        if token is None:
            return
        recorded = self._trajectory.read(token)
        new_outputs = self._new_tool_outputs(request_messages, recorded)
        assistant_items = build_assistant_items(message_dict)
        self._trajectory.append(token, [*new_outputs, *assistant_items])

    @staticmethod
    def _new_tool_outputs(
        request_messages: List[Dict[str, Any]], recorded: List[NeMoGymResponseOutputItem]
    ) -> List[NeMoGymFunctionCallOutput]:
        recorded_call_ids = {
            it.call_id for it in recorded if isinstance(it, NeMoGymFunctionCallOutput) and it.call_id is not None
        }
        outputs: List[NeMoGymFunctionCallOutput] = []
        for m in request_messages:
            if not isinstance(m, dict):
                continue
            call_id = output = None
            if m.get("role") == "tool":
                call_id, output = m.get("tool_call_id"), m.get("content")
            elif m.get("type") == "function_call_output":
                call_id, output = m.get("call_id"), m.get("output")
            if call_id is None or call_id in recorded_call_ids:
                continue
            recorded_call_ids.add(call_id)
            outputs.append(
                NeMoGymFunctionCallOutput(
                    call_id=call_id,
                    output="" if output is None else output if isinstance(output, str) else json.dumps(output),
                    status="completed",
                )
            )
        return outputs

    def attach_tokens_and_logprobs(self, request: Request, messages: List[Dict[str, Any]]) -> None:
        # linear prefix filter: isolates this stream from parallel subagent items in the buffer.
        # breaks on compaction or history rewrites; retries take first match (not longest chain).
        token = getattr(request.state, "run_token", None)
        if token is None:
            return
        for message in messages:
            if isinstance(message, dict):
                message.pop("prompt_token_ids", None)
                message.pop("generation_token_ids", None)
                message.pop("generation_log_probs", None)
        all_recorded = [it for it in self._trajectory.read(token) if getattr(it, "prompt_token_ids", None) is not None]
        # keep only items that extend the previous kept item's full sequence
        recorded: List = []
        expected_prefix: List[int] = []
        for it in all_recorded:
            p = list(it.prompt_token_ids)
            if not expected_prefix or (
                len(p) >= len(expected_prefix) and p[: len(expected_prefix)] == expected_prefix
            ):
                recorded.append(it)
                expected_prefix = p + list(getattr(it, "generation_token_ids", None) or [])
        ri = 0
        for message in messages:
            if not (isinstance(message, dict) and message.get("role") == "assistant"):
                continue
            if ri >= len(recorded):
                break
            r = recorded[ri]
            ri += 1
            message["prompt_token_ids"] = r.prompt_token_ids
            message["generation_token_ids"] = r.generation_token_ids
            message["generation_log_probs"] = r.generation_log_probs

    async def _pop_trajectory(self, request: Request) -> JSONResponse:
        token = getattr(request.state, "run_token", None)
        trajectory = getattr(self, "_trajectory", None)
        items = trajectory.pop(token) if (token is not None and trajectory is not None) else []
        return JSONResponse({"output": [it.model_dump() for it in _main_chain(items)]})





class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses)

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(self.messages)

        return app

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    async def messages(self, request: Request, body: dict = Body()):
        """Default Anthropic Messages <-> Responses mapping shared by every Gym model server.

        Translates the inbound Anthropic Messages request to the Responses API, delegates to this
        server's own ``responses()`` (so it reuses whatever backend the server has), and maps the
        result back to an Anthropic Messages response. When the client requested ``stream: true``
        (the Claude Code CLI always does), the complete response is re-emitted as a synthesized
        Anthropic SSE event stream. Servers may override this for native Messages handling.
        """
        params = _ANTHROPIC_CONVERTER.anthropic_request_to_responses(body)
        response = await self._invoke_responses(request, params)
        model_name = body.get("model") or response.model
        anthropic_response = _ANTHROPIC_CONVERTER.responses_to_anthropic_response(response, model=model_name)
        if body.get("stream"):
            return StreamingResponse(
                _ANTHROPIC_CONVERTER.anthropic_response_to_sse(anthropic_response),
                media_type="text/event-stream",
            )
        return anthropic_response

    async def _invoke_responses(
        self, request: Request, params: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        # responses() signatures vary across servers: some take a leading `request`, some only
        # `body`. Dispatch on whichever this server declares so the default messages() works for
        # all of them.
        if "request" in inspect.signature(self.responses).parameters:
            return await self.responses(request=request, body=params)
        return await self.responses(body=params)
