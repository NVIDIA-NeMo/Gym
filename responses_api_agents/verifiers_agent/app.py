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

from __future__ import annotations

import json
import logging
import traceback
from http.cookiejar import CookieJar
from typing import Any

import verifiers as vf
from fastapi import Body, Request, Response
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import ConfigDict, Field
from verifiers.clients import NeMoRLChatCompletionsClient

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


logger = logging.getLogger(__name__)


def _as_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    return {k: getattr(msg, k, None) for k in ("role", "content", "tool_calls", "tool_call_id", "tokens")}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, default=str)


def _tok_kwargs(tokens: dict | None) -> dict:
    if not tokens:
        return {}
    kwargs = {
        "prompt_token_ids": tokens.get("prompt_ids", []),
        "generation_token_ids": tokens.get("completion_ids", []),
        "generation_log_probs": tokens.get("completion_logprobs", []),
    }
    routed_experts = tokens.get("routed_experts")
    if routed_experts is not None:
        kwargs["routed_experts"] = routed_experts
    return kwargs


def _normalize_tool_call(tool_call: Any) -> dict:
    if isinstance(tool_call, str):
        try:
            return json.loads(tool_call)
        except json.JSONDecodeError:
            return {"arguments": tool_call}
    return tool_call


def _build_tool_result_item(msg: dict, raw: Any) -> dict:
    call_id = msg.get("tool_call_id") or f"call_{id(raw)}"
    return NeMoGymFunctionCallOutput(
        call_id=call_id,
        id=call_id,
        output=_text(msg.get("content")),
        status="completed",
    ).model_dump()


def _build_function_call_item(tool_call: Any, tokens: dict | None) -> dict:
    tc = _normalize_tool_call(tool_call)
    function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    call_id = tc.get("id") or tc.get("call_id") or f"call_{id(tool_call)}"
    name = tc.get("name") or function.get("name", "")
    arguments = _text(tc.get("arguments") or function.get("arguments") or "{}")

    cls = NeMoGymResponseFunctionToolCallForTraining if tokens else NeMoGymResponseFunctionToolCall
    return cls(
        id=call_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
        **_tok_kwargs(tokens),
    ).model_dump()


def _build_message_item(raw: Any, body: str, tokens: dict | None) -> dict:
    cls = NeMoGymResponseOutputMessageForTraining if tokens else NeMoGymResponseOutputMessage
    return cls(
        id=f"msg_{id(raw)}",
        content=[NeMoGymResponseOutputText(text=body, annotations=[])],
        **_tok_kwargs(tokens),
    ).model_dump()


def _build_assistant_items(msg: dict, raw: Any, tokens: dict | None) -> list[dict]:
    tool_calls = msg.get("tool_calls") or []
    body = _text(msg.get("content"))
    items: list[dict] = []

    if body:
        items.append(_build_message_item(raw, body, tokens if not tool_calls else None))

    for i, tool_call in enumerate(tool_calls):
        is_last = i == len(tool_calls) - 1
        items.append(_build_function_call_item(tool_call, tokens if is_last else None))

    if not items:
        items.append(_build_message_item(raw, "", tokens))

    return items


class VerifiersNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    output: list[dict[str, Any]]
    reward: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: list = Field(default_factory=list)


class VerifiersAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: VerifiersNeMoGymResponse
    reward: float


class _NoStoreCookieJar(CookieJar):
    """A cookie jar that drops every Set-Cookie, so no request ever carries one.

    See VerifiersAgent._get_client: the policy server's session cookie decides
    which vLLM engine serves a request, and a jar that remembers it would pin
    every rollout in this process to one engine.
    """

    def set_cookie(self, cookie) -> None:
        return None


class VerifiersAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model_name: str = Field(default="", description="Model name")

    vf_env_id: str = Field(default="", description="Verifiers environment ID")
    vf_env_args: dict = Field(default_factory=dict, description="Verifiers environment arguments")

    max_tokens: int = Field(default=8192, description="Max tokens for generation")

    # nemo rl generation_config overrides these
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)


class VerifiersAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = Field(default=None, description="Verifiers environment ID")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    answer: str = Field(default="", description="Expected answer from dataset")
    task: str = Field(default="default", description="Task type from dataset")
    example_id: int | str = Field(default=0, description="Example ID from dataset")
    info: dict = Field(default_factory=dict, description="Extra info from dataset")


class VerifiersAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: VerifiersAgentConfig

    envs_cache: dict[str, Any] = Field(default_factory=dict)
    client_cache: dict[str, NeMoRLChatCompletionsClient] = Field(default_factory=dict)

    def _get_env(self, vf_env_id: str) -> vf.Environment:
        if vf_env_id not in self.envs_cache:
            self.envs_cache[vf_env_id] = vf.load_environment(vf_env_id, **self.config.vf_env_args)
        return self.envs_cache[vf_env_id]

    def _get_client(self, body: Any = None) -> NeMoRLChatCompletionsClient:
        """Return a rollout-prefixed client over one shared policy transport.

        The vllm_model server picks a vLLM engine per session
        (``sha256(session_id) % len(base_urls)`` in
        responses_api_models/vllm_model/app.py ``_resolve_client``) and mints the
        session id per cookie jar (nemo_gym/server_utils.py
        ``setup_session_middleware``). openai's AsyncOpenAI sits on an httpx client
        that persists cookies, so a plain shared client is one session and
        therefore one engine: on CMH job 3670120 (2026-09-10) 512 concurrent
        rollouts ran on 6 of 48 engines while 42 sat idle. Gym's own aiohttp
        client avoids exactly this with a DummyCookieJar; ``_NoStoreCookieJar``
        is the httpx equivalent. Every request is then a fresh session and the
        router spreads them over every engine.

        Why not a client per rollout: each rollout's single pooled connection sat
        idle through its tool phases, the router's uvicorn closes idle
        connections after 30 s, and the next turn raced that close -- CMH 3670792
        aborted 468 of 512 rollouts with
        ``APIConnectionError -> ReadError(BrokenResourceError)`` while the
        shared-client runs before it had zero. One shared pool keeps connections
        hot. The price is per-turn engine affinity, which Gym's own client does
        not have either.

        Model-call capture is keyed by the ``/ng-rollout/<id>`` URL prefix. The
        lightweight per-run OpenAI client copy changes only ``base_url`` and
        shares the cached client's transport, so calls remain correlated without
        accumulating a connection pool per rollout. ``rollout_id_from_run``
        returns ``None`` when capture is disabled, preserving the shared
        unprefixed client path.
        """
        cache_key = self.config.model_server.name
        if cache_key not in self.client_cache:
            openai_client = AsyncOpenAI(
                base_url=self.resolve_model_base_url(self.config.model_server.name),
                api_key="EMPTY",  # pragma: allowlist secret
                # DefaultAsyncHttpxClient keeps the SDK's pool limits and redirect
                # policy. Pass the bare CookieJar: httpx.Cookies adopts a CookieJar
                # instance as-is but COPIES an httpx.Cookies into a fresh stdlib
                # jar, which would silently discard the no-store behaviour.
                http_client=DefaultAsyncHttpxClient(cookies=_NoStoreCookieJar()),
            )
            self.client_cache[cache_key] = NeMoRLChatCompletionsClient(openai_client)

        shared_client = self.client_cache[cache_key]
        rollout_id = self.rollout_id_from_run(body) if body is not None else None
        if rollout_id is None:
            return shared_client

        model_server_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        return NeMoRLChatCompletionsClient(shared_client.client.copy(base_url=model_server_url))

    def _convert_trajectory_to_output(self, rollout_output: dict) -> list:
        assistant_tokens = self._collect_assistant_tokens(rollout_output.get("trajectory") or [])

        output: list[dict] = []
        assist_idx = 0
        for raw in rollout_output.get("completion") or []:
            msg = _as_dict(raw)
            role = msg.get("role", "user")

            if role == "tool":
                output.append(_build_tool_result_item(msg, raw))
            elif role == "assistant":
                tokens = assistant_tokens[assist_idx] if assist_idx < len(assistant_tokens) else None
                assist_idx += 1
                output.extend(_build_assistant_items(msg, raw, tokens))
            else:
                output.append(NeMoGymEasyInputMessage(role=role, content=_text(msg.get("content"))).model_dump())

        if not any(item.get("generation_token_ids") for item in output):
            err = rollout_output.get("error") or {}
            err_msg = err.get("error") or err.get("error_chain_repr") or "unknown (no error info on rollout_output)"
            logger.warning(
                "[verifiers_agent] rollout produced no trainable tokens. This can happen when sandbox concurrency quota is exceeded. Returning empty trajectory. "
                "Underlying error: %s",
                err_msg,
            )
            output.append(
                NeMoGymResponseOutputMessageForTraining(
                    id="msg_empty",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    prompt_token_ids=[0],
                    generation_token_ids=[0],
                    generation_log_probs=[0.0],
                ).model_dump()
            )

        return output

    @staticmethod
    def _collect_assistant_tokens(trajectory: list) -> list[dict | None]:
        tokens_per_turn: list[dict | None] = []
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            step_tokens = step.get("tokens")
            for m in step.get("completion") or []:
                if _as_dict(m).get("role") == "assistant":
                    tokens_per_turn.append(step_tokens)
        return tokens_per_turn

    async def responses(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersNeMoGymResponse:
        try:
            vf_env_id = body.vf_env_id or self.config.vf_env_id
            if not vf_env_id:
                raise ValueError("vf_env_id must be set on the request or in the agent config")
            vf_env = self._get_env(vf_env_id)
            task_idx = body.task_idx

            prompt_messages = []
            for item in body.responses_create_params.input or []:
                if hasattr(item, "role") and hasattr(item, "content"):
                    prompt_messages.append({"role": item.role, "content": item.content})
                elif isinstance(item, dict):
                    prompt_messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

            rollout_input = vf.RolloutInput(
                prompt=prompt_messages,
                answer=body.answer,
                info=body.info,
                example_id=body.example_id,
            )

            client = self._get_client(body)

            # prefer NeMo RL generation config set in responses_create_params
            # https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/experience/rollouts.py#L1045-L1046
            sampling_args = {
                "max_tokens": self.config.max_tokens,
                "temperature": getattr(body.responses_create_params, "temperature", None) or self.config.temperature,
                "top_p": getattr(body.responses_create_params, "top_p", None) or self.config.top_p,
            }
            outputs = await vf_env.run_group(
                group_inputs=[rollout_input],
                client=client,
                model=self.config.model_name,
                sampling_args=sampling_args,
                state_columns=["trajectory"],
            )

            rollout_output = outputs[0]
            reward = rollout_output.get("reward", 0.0) or 0.0
            metrics = rollout_output.get("metrics", {}) or {}

            output = self._convert_trajectory_to_output(rollout_output)

            return VerifiersNeMoGymResponse(
                id=f"verifiers-{vf_env_id}-{task_idx}",
                created_at=0,
                model=self.config.model_name,
                object="response",
                output=output,
                env_id=vf_env_id,
                group_id=str(task_idx),
                reward=reward,
                metrics=metrics,
            )
        except Exception as e:
            logger.error(f"Exception in responses(): {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def run(
        self,
        request: Request,
        response: Response,
        body: VerifiersAgentRunRequest = Body(),
    ) -> VerifiersAgentVerifyResponse:
        resp = await self.responses(request, response, body)

        return VerifiersAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=resp,
            reward=resp.reward,
        )


if __name__ == "__main__":
    VerifiersAgent.run_webserver()
