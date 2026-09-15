# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo-Sim's conversation protocol through Gym Agents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from typing import Any

from fastapi import Body, Request
from pydantic import Field

from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.processors import (
    AgentTurn,
    BaseProcessor,
    BaseProcessorConfig,
    EpisodeFailure,
    EpisodeRequest,
    EpisodeResponse,
    EpisodeSeedRequest,
    ResourcesSessionCloseRequest,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ToolCallObservation, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status
from processors.nemo_sim_processor.contracts import (
    NEMO_SIM_MODEL_ALIASES,
    NeMoSimProtocolConfig,
    NeMoSimScenario,
    NeMoSimSeedResponse,
    NeMoSimSimulationResult,
    NeMoSimTaskData,
    NeMoSimVerification,
    NeMoSimVerifyRequest,
)


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_AGENT_ID_BY_ALIAS = {"user_model": "user", "assistant_model": "assistant"}


class NeMoSimProcessorConfig(BaseProcessorConfig):
    user_agent: AgentServerRef
    assistant_agent: AgentServerRef
    judge_model: ModelServerRef
    summary_model: ModelServerRef
    api_response_model: ModelServerRef
    resources_server: ResourcesServerRef
    max_turns: int = Field(5, ge=1)
    agent_call_timeout_s: float = Field(300.0, gt=0)
    protocol_config: NeMoSimProtocolConfig = Field(default_factory=NeMoSimProtocolConfig)

    def target_for_alias(self, alias: str) -> AgentServerRef | ModelServerRef:
        return {
            "user_model": self.user_agent,
            "assistant_model": self.assistant_agent,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
            "api_response_model": self.api_response_model,
        }[alias]


class _GymModelFacade:
    """Synchronous interface expected by NeMo-Sim's generator."""

    def __init__(self, alias: str, bridge: "_ConversationBridge") -> None:
        self.alias = alias
        self.model_name = bridge.processor.config.target_for_alias(alias).name
        self._bridge = bridge

    def completion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        if kwargs.get("tools"):
            raise NotImplementedError("NeMoSimProcessor currently supports non-tool probes only")
        unsupported = set(kwargs) - {"max_tokens", "tools"}
        if unsupported:
            raise NotImplementedError(f"Unsupported NeMo-Sim completion options: {sorted(unsupported)}")
        return self._bridge.complete_from_worker(self.alias, messages, max_tokens=kwargs.get("max_tokens"))


class _GeneratorHarness:
    def __init__(self, config: Any, models: Mapping[str, _GymModelFacade]) -> None:
        self.config = config
        self._models = models

    def get_model(self, alias: str) -> _GymModelFacade:
        return self._models[alias]


class _ConversationBridge:
    """Bridge synchronous NeMo-Sim calls to async Gym endpoints."""

    def __init__(
        self,
        processor: "NeMoSimProcessor",
        episode: EpisodeRequest,
        task: NeMoSimTaskData,
        event_loop: asyncio.AbstractEventLoop,
        cookies: Mapping[str, Any],
    ) -> None:
        self.processor = processor
        self.episode = episode
        self.task = task
        self.event_loop = event_loop
        self.cookies_by_alias = {alias: dict(cookies) for alias in NEMO_SIM_MODEL_ALIASES}
        self.agent_turns: list[AgentTurn] = []
        self.responses_by_alias: dict[str, list[NeMoGymResponse]] = {alias: [] for alias in NEMO_SIM_MODEL_ALIASES}

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
            self._invoke(alias, messages, max_tokens=max_tokens),
            self.event_loop,
        )
        try:
            return future.result(timeout=self.processor.config.agent_call_timeout_s)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError(
                f"Timed out after {self.processor.config.agent_call_timeout_s}s waiting for {alias}"
            ) from error

    async def _invoke(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
    ) -> SimpleNamespace:
        params = self.task.model_responses_create_params.get(alias, self.episode.responses_create_params)
        values = params.model_dump(mode="json", exclude_none=True)
        values["input"] = [_to_responses_input(message) for message in messages]
        if max_tokens is not None:
            values["max_output_tokens"] = max_tokens
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(values)

        target = self.processor.config.target_for_alias(alias)
        response = await self.processor.server_client.post(
            server_name=target.name,
            url_path=self.processor.url_path_for_run("/v1/responses", self.episode),
            json=request_params.model_dump(mode="json", exclude_none=True),
            cookies=self.cookies_by_alias[alias],
        )
        await raise_for_status(response)
        response_data = await get_response_json(response)
        trajectory_data = response_data.pop(_INTERNAL_TRAJECTORY_KEY, None)
        gym_response = NeMoGymResponse.model_validate(response_data)
        self.cookies_by_alias[alias].update(response.cookies)
        self.responses_by_alias[alias].append(gym_response)
        if alias in _AGENT_ID_BY_ALIAS:
            self.agent_turns.append(
                AgentTurn(
                    sequence=len(self.agent_turns),
                    agent_id=_AGENT_ID_BY_ALIAS[alias],
                    request=request_params,
                    response=gym_response,
                    observations=_agent_observations(target.name, trajectory_data),
                )
            )

        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(
                content=_response_text(gym_response),
                reasoning_content=None,
                # The Agent Server already executes its complete tool loop.
                tool_calls=None,
            ),
            usage=(
                SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                if usage is not None
                else None
            ),
        )


def _agent_observations(source: str, trajectory_data: Any) -> AgentObservationBundle | None:
    if trajectory_data is None:
        return None
    trajectory = TrajectoryRecord.model_validate(trajectory_data)
    tool_observations = [
        ToolCallObservation.model_validate(record.model_dump(exclude={"output"})) for record in trajectory.tool_calls
    ]
    return AgentObservationBundle(
        source=source,
        records=[*trajectory.invocations, *tool_observations],
        gaps=trajectory.gaps,
    )


def _to_responses_input(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {"role": getattr(message, "role"), "content": getattr(message, "content", "")}
    role = getattr(value.get("role"), "value", value.get("role"))
    if role not in {"system", "developer", "user", "assistant"}:
        raise NotImplementedError(f"NeMo-Sim message role {role!r} is not supported")
    return {"type": "message", "role": role, "content": value.get("content", "")}


def _response_text(response: NeMoGymResponse) -> str:
    chunks: list[str] = []
    for item in response.output:
        if not isinstance(item, NeMoGymResponseOutputMessage):
            continue
        for content in item.content:
            text = getattr(content, "text", None) or getattr(content, "refusal", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _last_turn_sequence(agent_turns: list[AgentTurn], agent_id: str) -> int | None:
    return next((turn.sequence for turn in reversed(agent_turns) if turn.agent_id == agent_id), None)


class NeMoSimProcessor(BaseProcessor):
    config: NeMoSimProcessorConfig

    def _run_nemo_sim(self, bridge: _ConversationBridge, scenario: NeMoSimScenario) -> dict[str, Any]:
        from conversation_plugin.config import ConversationSimulatorConfig
        from conversation_plugin.core.llm import set_debug_log_path
        from conversation_plugin.generator import ConversationSimulatorGenerator

        set_debug_log_path(None)
        config_values = self.config.protocol_config.model_dump(mode="python", exclude_none=True)
        config_values.update(
            {"name": "conversation_messages", "locale": scenario.locale, "max_turns": self.config.max_turns}
        )
        config = ConversationSimulatorConfig.model_validate(config_values)
        models = {alias: _GymModelFacade(alias, bridge) for alias in NEMO_SIM_MODEL_ALIASES}
        generator = _GeneratorHarness(config, models)
        return ConversationSimulatorGenerator.generate(
            generator,
            scenario.model_dump(mode="python", exclude={"locale"}),
        )

    async def run(self, request: Request, body: EpisodeRequest = Body()) -> EpisodeResponse:
        task = NeMoSimTaskData.model_validate(body.task_data)
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=EpisodeSeedRequest(
                episode_id=body.episode_id,
                task=body.task,
                task_data=body.task_data,
            ).model_dump(mode="json"),
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        seed = NeMoSimSeedResponse.model_validate(await get_response_json(seed_response))
        environment_cookies = dict(seed_response.cookies)
        bridge = _ConversationBridge(self, body, task, asyncio.get_running_loop(), environment_cookies)
        episode_response: EpisodeResponse
        try:
            raw_result = await asyncio.to_thread(self._run_nemo_sim, bridge, seed.scenario)
            result = NeMoSimSimulationResult.model_validate(raw_result)
            assistant_responses = bridge.responses_by_alias["assistant_model"]
            if not assistant_responses:
                episode_response = EpisodeResponse(
                    episode_id=body.episode_id,
                    task=body.task,
                    agent_turns=bridge.agent_turns,
                    failure=EpisodeFailure(
                        kind="agent",
                        message="NeMo-Sim completed without an Assistant response",
                        retryable=False,
                    ),
                )
            else:
                focal_response = assistant_responses[-1]
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=NeMoSimVerifyRequest(
                        episode_id=body.episode_id,
                        responses_create_params=body.responses_create_params,
                        response=focal_response,
                        scenario=seed.scenario,
                        nemo_sim_context=seed.nemo_sim_context,
                        nemo_sim_result=result,
                        agent_turns=bridge.agent_turns,
                    ).model_dump(mode="json"),
                    cookies=environment_cookies,
                )
                await raise_for_status(verify_response)
                verification = NeMoSimVerification.model_validate(await get_response_json(verify_response))
                episode_response = EpisodeResponse(
                    episode_id=body.episode_id,
                    task=body.task,
                    agent_turns=bridge.agent_turns,
                    output_turn_sequence=_last_turn_sequence(bridge.agent_turns, "assistant"),
                    verification=verification,
                )
        except Exception as error:
            assistant_responses = bridge.responses_by_alias["assistant_model"]
            episode_response = EpisodeResponse(
                episode_id=body.episode_id,
                task=body.task,
                agent_turns=bridge.agent_turns,
                output_turn_sequence=(
                    _last_turn_sequence(bridge.agent_turns, "assistant") if assistant_responses else None
                ),
                failure=EpisodeFailure(
                    kind="agent" if isinstance(error, TimeoutError) else "internal",
                    message=f"NeMo-Sim episode failed ({type(error).__name__})",
                    retryable=isinstance(error, TimeoutError),
                ),
            )

        try:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesSessionCloseRequest(resources_session_id=seed.resources_session_id).model_dump(
                    mode="json"
                ),
                cookies=environment_cookies,
            )
            await raise_for_status(close_response)
        except Exception as error:
            return EpisodeResponse(
                episode_id=body.episode_id,
                task=body.task,
                agent_turns=episode_response.agent_turns,
                output_turn_sequence=episode_response.output_turn_sequence,
                failure=EpisodeFailure(
                    kind="internal",
                    message=f"Failed to close NeMo-Sim resources ({type(error).__name__})",
                    retryable=True,
                ),
            )
        return episode_response


if __name__ == "__main__":
    NeMoSimProcessor.run_webserver()
