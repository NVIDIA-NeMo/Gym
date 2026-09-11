# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo-Sim's episode interaction protocol through Gym Agents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import Body, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.agents.responses_api_agent import INTERNAL_TRAJECTORY_KEY
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.server_utils import get_response_json, raise_for_status


_MODEL_ALIASES = frozenset(
    {
        "user_model",
        "assistant_model",
        "api_response_model",
        "judge_model",
        "summary_model",
    }
)


class NeMoSimScenario(BaseModel):
    """One resolved NeMo-Sim input row before conversation generation."""

    model_config = ConfigDict(extra="allow")

    persona: dict[str, Any]
    probe_type: str = "general_open_ended"
    theme: dict[str, Any] | str
    locale: str = "en_US"


class NeMoSimRunRequest(BaseRunRequest):
    """Gym request plus the NeMo-Sim row and per-alias response parameters."""

    model_config = ConfigDict(extra="allow")

    scenario: NeMoSimScenario
    model_responses_create_params: dict[str, NeMoGymResponseCreateParamsNonStreaming] = Field(default_factory=dict)
    simulation_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_unknown_model_aliases(self) -> "NeMoSimRunRequest":
        unknown = set(self.model_responses_create_params) - _MODEL_ALIASES
        if unknown:
            raise ValueError(f"model_responses_create_params contains unknown aliases: {sorted(unknown)}")
        return self


class NeMoSimInvocation(BaseModel):
    """One participant turn or support-model call made by NeMo-Sim."""

    alias: str
    executor: Literal["agent", "model"]
    call_index: int
    request: dict[str, Any]
    response: dict[str, Any]
    ng_trajectory: dict[str, Any] | None = None


class NeMoSimProcessorResponse(BaseVerifyResponse):
    """Output exposing both NeMo-Sim state and attributed Gym Agent calls."""

    model_config = ConfigDict(extra="allow")

    nemo_sim_result: dict[str, Any]
    invocations: list[NeMoSimInvocation]
    episode_interaction_protocol: str = "nemo_sim.ConversationLoop"


class NeMoSimProcessorConfig(BaseProcessorConfig):
    """Configure participant Agents separately from support Model Servers."""

    user_agent: AgentServerRef
    assistant_agent: AgentServerRef
    judge_model: ModelServerRef
    summary_model: ModelServerRef
    api_response_model: ModelServerRef
    max_turns: int = Field(5, ge=1)
    agent_call_timeout_s: float = Field(300.0, gt=0)
    skip_verification: Literal[True] = True

    def target_for_alias(self, alias: str) -> AgentServerRef | ModelServerRef:
        return {
            "user_model": self.user_agent,
            "assistant_model": self.assistant_agent,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
            "api_response_model": self.api_response_model,
        }[alias]


class _GymModelFacade:
    """Synchronous facade expected by NeMo-Sim's Data Designer integration."""

    def __init__(self, alias: str, bridge: "_ConversationBridge") -> None:
        self.alias = alias
        self.model_name = bridge.processor.config.target_for_alias(alias).name
        self._bridge = bridge

    def completion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        if kwargs.get("tools"):
            raise NotImplementedError(
                "NeMoSimProcessor currently supports non-tool probes only: tool execution ownership between "
                "ConversationLoop and Gym Agents remains an open design question."
            )
        unsupported = set(kwargs) - {"max_tokens", "tools"}
        if unsupported:
            raise NotImplementedError(f"Unsupported NeMo-Sim completion options: {sorted(unsupported)}")
        return self._bridge.complete_from_worker(self.alias, messages, max_tokens=kwargs.get("max_tokens"))


class _GeneratorHarness:
    """Duck-typed host for NeMo-Sim's existing Data Designer row adapter."""

    def __init__(self, config: Any, models: Mapping[str, _GymModelFacade]) -> None:
        self.config = config
        self._models = models

    def get_model(self, alias: str) -> _GymModelFacade:
        return self._models[alias]


class _ConversationBridge:
    """Bridge blocking NeMo-Sim calls to async Gym Agent invocations."""

    def __init__(
        self,
        *,
        processor: "NeMoSimProcessor",
        body: NeMoSimRunRequest,
        event_loop: asyncio.AbstractEventLoop,
        cookies: Mapping[str, Any],
    ) -> None:
        self.processor = processor
        self.body = body
        self.event_loop = event_loop
        self.cookies_by_alias = {alias: dict(cookies) for alias in _MODEL_ALIASES}
        self.invocations: list[NeMoSimInvocation] = []
        self.responses_by_alias: dict[str, list[NeMoGymResponse]] = {alias: [] for alias in _MODEL_ALIASES}

    def complete_from_worker(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
    ) -> SimpleNamespace:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.event_loop:
            raise RuntimeError("NeMo-Sim's synchronous ConversationLoop must run outside the Processor event loop")

        future = asyncio.run_coroutine_threadsafe(
            self._invoke_agent(alias, messages, max_tokens=max_tokens),
            self.event_loop,
        )
        try:
            return future.result(timeout=self.processor.config.agent_call_timeout_s)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError(
                f"Timed out after {self.processor.config.agent_call_timeout_s}s waiting for {alias}"
            ) from error

    async def _invoke_agent(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
    ) -> SimpleNamespace:
        params = self.body.model_responses_create_params.get(alias, self.body.responses_create_params)
        input_messages = [_to_responses_input(message) for message in messages]
        request_values = params.model_dump(mode="json", exclude_none=True)
        request_values.update({"input": input_messages, "instructions": None, "tools": []})
        if max_tokens is not None:
            request_values["max_output_tokens"] = max_tokens
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(request_values)
        request_json = request_params.model_dump(mode="json", exclude_none=True)

        target = self.processor.config.target_for_alias(alias)
        response = await self.processor.server_client.post(
            server_name=target.name,
            url_path=self.processor.url_path_for_run("/v1/responses", self.body),
            json=request_json,
            cookies=self.cookies_by_alias[alias],
        )
        await raise_for_status(response)
        response_json = await get_response_json(response)
        agent_trajectory = response_json.pop(INTERNAL_TRAJECTORY_KEY, None)
        gym_response = NeMoGymResponse.model_validate(response_json)
        self.cookies_by_alias[alias] = dict(response.cookies)
        self.responses_by_alias[alias].append(gym_response)
        self.invocations.append(
            NeMoSimInvocation(
                alias=alias,
                executor="agent" if isinstance(target, AgentServerRef) else "model",
                call_index=len(self.responses_by_alias[alias]) - 1,
                request=request_json,
                response=gym_response.model_dump(mode="json"),
                ng_trajectory=agent_trajectory,
            )
        )

        tool_calls = [
            {
                "id": item.call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": item.arguments},
            }
            for item in gym_response.output
            if isinstance(item, NeMoGymResponseFunctionToolCall)
        ]
        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(
                content=_response_text(gym_response),
                reasoning_content=None,
                tool_calls=tool_calls or None,
            ),
            usage=(
                SimpleNamespace(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                if usage is not None
                else None
            ),
        )


def _to_responses_input(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {
            "role": getattr(message, "role"),
            "content": getattr(message, "content", ""),
        }

    role = value.get("role")
    if hasattr(role, "value"):
        role = role.value
    if role not in {"system", "developer", "user", "assistant"}:
        raise NotImplementedError(f"NeMo-Sim message role {role!r} is not supported by NeMoSimProcessor")
    return {"type": "message", "role": role, "content": value.get("content", "")}


def _response_text(response: NeMoGymResponse) -> str:
    chunks: list[str] = []
    for item in response.output:
        if not isinstance(item, NeMoGymResponseOutputMessage):
            continue
        for content in item.content:
            text = getattr(content, "text", None)
            refusal = getattr(content, "refusal", None)
            if text:
                chunks.append(text)
            elif refusal:
                chunks.append(refusal)
    return "\n".join(chunks)


class NeMoSimProcessor(BaseProcessor):
    """Let NeMo-Sim orchestrate participant Agents and support Model Servers."""

    config: NeMoSimProcessorConfig

    def _run_nemo_sim(self, bridge: _ConversationBridge, body: NeMoSimRunRequest) -> dict[str, Any]:
        from conversation_plugin.config import ConversationSimulatorConfig
        from conversation_plugin.core.llm import set_debug_log_path
        from conversation_plugin.generator import ConversationSimulatorGenerator

        set_debug_log_path(None)
        config_values = dict(body.simulation_config)
        config_values.update(
            {
                "name": "conversation_messages",
                "locale": body.scenario.locale,
                "max_turns": self.config.max_turns,
            }
        )
        simulation_config = ConversationSimulatorConfig.model_validate(config_values)
        models = {alias: _GymModelFacade(alias, bridge) for alias in _MODEL_ALIASES}
        generator = _GeneratorHarness(simulation_config, models)
        scenario_data = body.scenario.model_dump(mode="python", exclude={"locale"})
        return ConversationSimulatorGenerator.generate(generator, scenario_data)

    async def run(
        self,
        request: Request,
        body: NeMoSimRunRequest = Body(),
    ) -> NeMoSimProcessorResponse:
        bridge = _ConversationBridge(
            processor=self,
            body=body,
            event_loop=asyncio.get_running_loop(),
            cookies=request.cookies,
        )
        nemo_sim_result = await asyncio.to_thread(self._run_nemo_sim, bridge, body)
        assistant_responses = bridge.responses_by_alias["assistant_model"]
        focal_response = assistant_responses[-1] if assistant_responses else _empty_assistant_response(self.config)

        result = body.model_dump(mode="json") | {
            "response": focal_response.model_dump(mode="json"),
            "reward": float(self.config.skip_verification_reward),
            "verification_skipped": True,
            "nemo_sim_result": nemo_sim_result,
            "invocations": [invocation.model_dump(mode="json") for invocation in bridge.invocations],
            "episode_interaction_protocol": "nemo_sim.ConversationLoop",
        }
        return NeMoSimProcessorResponse.model_validate(result)


def _empty_assistant_response(config: NeMoSimProcessorConfig) -> NeMoGymResponse:
    """Preserve a structured NeMo-Sim failure that occurs before an assistant turn."""

    return NeMoGymResponse.model_validate(
        {
            "id": "",
            "created_at": 0,
            "model": config.assistant_agent.name,
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
