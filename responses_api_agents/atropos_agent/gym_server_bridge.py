# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Atropos style ServerManager for NeMo Gym to send inference through
NeMo Gym's responses_api_model atropos_model (which extends vllm_model with /v1/completions).

Token IDs are injected by the responses_api_model for NeMo RL's 
replace_prefix_tokens retokenization correction for on policy training.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai.types.chat.chat_completion import ChatCompletion

from nemo_gym.server_utils import get_global_aiohttp_client


logger = logging.getLogger(__name__)


@dataclass
class GymTokenInfo:
    prompt_token_ids: List[int]
    generation_token_ids: List[int]
    generation_log_probs: List[float]


@dataclass
class SequenceNode:
    full_text: str = ""
    tokens: List[int] = field(default_factory=list)
    masked_tokens: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class GymVLLMServer:
    def __init__(self, model_server_url: str, model_name: str) -> None:
        self._url = model_server_url.rstrip("/")
        self._model_name = model_name
        self._token_infos: List[GymTokenInfo] = []
        self.current_nodes: List[SequenceNode] = []
        self.sem = asyncio.Semaphore(512)
        self.server_healthy = True
        self.config = type("Config", (), {"model_name": model_name})()

    def get_token_infos(self) -> List[GymTokenInfo]:
        return list(self._token_infos)

    def clear_token_infos(self) -> None:
        self._token_infos.clear()
        self.current_nodes.clear()

    def get_state(self) -> Dict[str, Any]:
        return {"nodes": list(self.current_nodes)}

    def _record_choice(self, prompt_ids: List[int], gen_ids: List[int], gen_lps: List[float], text: str, finish: str):
        if len(gen_lps) > len(gen_ids):
            gen_lps = gen_lps[: len(gen_ids)]
        elif len(gen_lps) < len(gen_ids):
            gen_lps = gen_lps + [0.0] * (len(gen_ids) - len(gen_lps))

        self._token_infos.append(
            GymTokenInfo(
                prompt_token_ids=list(prompt_ids),
                generation_token_ids=gen_ids,
                generation_log_probs=gen_lps,
            )
        )
        self.current_nodes.append(
            SequenceNode(
                full_text=text,
                tokens=list(prompt_ids) + gen_ids,
                masked_tokens=[-100] * len(prompt_ids) + gen_ids,
                logprobs=[1.0] * len(prompt_ids) + gen_lps,
                metadata={"finish_reason": finish},
            )
        )

    async def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        session = get_global_aiohttp_client()
        async with session.post(f"{self._url}{path}", json=body) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Model server request failed ({resp.status}): {error_text}")
                resp.raise_for_status()
            return await resp.json()

    async def chat_completion(self, **kwargs: Any) -> ChatCompletion:
        kwargs.pop("model", None)
        kwargs.pop("split", None)

        request_body: Dict[str, Any] = {
            "model": self._model_name,
            "messages": kwargs.get("messages", []),
            "logprobs": True,
        }
        for key in ("n", "max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice"):
            if kwargs.get(key) is not None:
                request_body[key] = kwargs[key]

        response_dict = await self._post("/chat/completions", request_body)

        for choice_dict in response_dict.get("choices", []):
            message_dict = choice_dict.get("message", {})
            prompt_ids = message_dict.pop("prompt_token_ids", [])
            gen_ids = message_dict.pop("generation_token_ids", [])
            gen_lps = message_dict.pop("generation_log_probs", [])

            if prompt_ids and isinstance(prompt_ids[0], str):
                prompt_ids = [int(x) for x in prompt_ids]
            if gen_ids and isinstance(gen_ids[0], str):
                gen_ids = [int(x) for x in gen_ids]

            self._record_choice(
                prompt_ids,
                gen_ids,
                gen_lps,
                message_dict.get("content", "") or "",
                choice_dict.get("finish_reason", "stop"),
            )

        return ChatCompletion.model_validate(response_dict)

    async def completion(self, **kwargs: Any) -> Any:
        from openai.types.completion import Completion

        kwargs.pop("model", None)
        kwargs.pop("split", None)

        request_body: Dict[str, Any] = {
            "model": self._model_name,
            "prompt": kwargs.get("prompt", ""),
            "logprobs": 1,
        }
        for key in ("n", "max_tokens", "temperature", "top_p", "stop"):
            if kwargs.get(key) is not None:
                request_body[key] = kwargs[key]

        response_dict = await self._post("/completions", request_body)

        for choice_dict in response_dict.get("choices", []):
            prompt_ids = choice_dict.pop("prompt_token_ids", [])
            gen_ids = choice_dict.pop("generation_token_ids", [])
            gen_lps = choice_dict.pop("generation_log_probs", [])

            if isinstance(prompt_ids, list) and prompt_ids and isinstance(prompt_ids[0], str):
                prompt_ids = [int(x) for x in prompt_ids]
            if isinstance(gen_ids, list) and gen_ids and isinstance(gen_ids[0], str):
                gen_ids = [int(x) for x in gen_ids]

            gen_lps = [lp if lp is not None else 0.0 for lp in (gen_lps or [])]
            self._record_choice(
                prompt_ids or [],
                gen_ids or [],
                gen_lps,
                choice_dict.get("text", ""),
                choice_dict.get("finish_reason", "stop"),
            )

        return Completion.model_validate(response_dict)

    async def tokens_and_logprobs_completion(self, **kwargs: Any) -> tuple:
        kwargs.pop("model", None)
        kwargs.pop("split", None)
        input_ids = kwargs.pop("input_ids", None)
        prompt_text = kwargs.pop("prompt", "")

        request_body: Dict[str, Any] = {"model": self._model_name, "logprobs": 1}
        if input_ids is not None:
            request_body["prompt"] = {"prompt_token_ids": input_ids}
        else:
            request_body["prompt"] = prompt_text
        for key in ("n", "max_tokens", "temperature", "top_p", "stop"):
            if kwargs.get(key) is not None:
                request_body[key] = kwargs[key]

        response_dict = await self._post("/completions", request_body)

        prompt_token_ids = response_dict.get("choices", [{}])[0].pop("prompt_token_ids", input_ids or [])

        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons_list = []
        for choice_dict in response_dict.get("choices", []):
            gen_ids = choice_dict.pop("generation_token_ids", [])
            gen_lps = choice_dict.pop("generation_log_probs", [])
            gen_lps = [lp if lp is not None else 0.0 for lp in gen_lps]
            output_tokens_list.append(gen_ids)
            output_logprobs_list.append(gen_lps)
            finish_reasons_list.append(choice_dict.get("finish_reason", "stop"))

        return prompt_token_ids, output_tokens_list, output_logprobs_list, finish_reasons_list


class GymServerManager:
    def __init__(self, model_server_url: str, model_name: str) -> None:
        self._server = GymVLLMServer(model_server_url, model_name)
        self.servers = [self._server]

    def get_server(self) -> GymVLLMServer:
        return self._server

    async def chat_completion(self, **kwargs: Any) -> ChatCompletion:
        return await self._server.chat_completion(**kwargs)

    async def completion(self, **kwargs: Any) -> Any:
        return await self._server.completion(**kwargs)

    async def tokens_and_logprobs_completion(self, **kwargs: Any) -> tuple:
        return await self._server.tokens_and_logprobs_completion(**kwargs)

    @asynccontextmanager
    async def dedicated_server(self):
        async with self._server.sem:
            yield self._server

    @asynccontextmanager
    async def managed_server(self, **kwargs):
        async with self._server.sem:
            yield self._server
