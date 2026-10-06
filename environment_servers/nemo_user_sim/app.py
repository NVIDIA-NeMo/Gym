# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo UserSim's conversation protocol through Gym servers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from aiohttp import ClientError, ClientResponseError
from fastapi import Body
from pydantic import ConfigDict, Field

from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    CleanupHandle,
    HandledEpisodeError,
)
from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import (
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
)
from nemo_gym.failure_kinds import FailureStage
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK, get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ToolCallObservation, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.server_utils import request as http_request
from resources_servers.nemo_user_sim.episode_contracts import (
    UserSimEpisodeFailure,
    UserSimEpisodeRequest,
    UserSimEpisodeResponse,
    UserSimEpisodeResult,
    UserSimInvocation,
    UserSimSeedResponse,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimVerification,
    UserSimVerificationInput,
    UserSimVerifyRequest,
)


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_INVOCATION_ROLE_BY_ALIAS = {
    "user_model": "user",
    "assistant_model": "assistant",
    "judge_model": "judge",
    "summary_model": "summary",
    "api_response_model": "tool_simulation",
}
_PARTICIPANT_ALIASES = ("user_model", "assistant_model")
_PARTICIPANT_ROLES = {"user", "assistant"}


class UserSimEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind UserSim participants to Agents and support aliases to Model Servers."""

    model_config = ConfigDict(extra="forbid")

    user_agent: AgentServerRef
    assistant_agent: AgentServerRef
    judge_model: ModelServerRef
    summary_model: ModelServerRef
    tool_simulation_model: ModelServerRef
    embedding_model: ModelServerRef
    resources_server: ResourcesServerRef
    actor_call_timeout_seconds: float = Field(300.0, gt=0)

    def target_for_alias(self, alias: str) -> AgentServerRef | ModelServerRef:
        return {
            "user_model": self.user_agent,
            "assistant_model": self.assistant_agent,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
            "api_response_model": self.tool_simulation_model,
            "embedding_model": self.embedding_model,
        }[alias]


@dataclass
class _AgentSession:
    alias: str
    target: AgentServerRef
    session_id: str
    cookies: dict[str, str]
    cleanup: CleanupHandle | None = None
    close_response: AgentCloseSessionResponse | None = None


class _ParticipantActivationError(Exception):
    """Identify which participant Agent failed to produce a response."""

    def __init__(self, alias: str, error: Exception) -> None:
        super().__init__(f"{alias} activation failed: {type(error).__name__}: {error}")
        self.alias = alias
        self.error = error


class _GymModelFacade:
    """Route UserSim's internal tool-simulation calls through a Gym Model Server."""

    def __init__(self, alias: str, bridge: "_ConversationBridge") -> None:
        self.alias = alias
        self.model_name = _configured_model_name(bridge.environment_server, alias)
        self._bridge = bridge

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        try:
            async with asyncio.timeout(self._bridge.environment_server.config.actor_call_timeout_seconds):
                return await self._bridge.invoke(
                    self.alias,
                    messages,
                    parameters=kwargs,
                )
        except TimeoutError as error:
            raise TimeoutError(
                f"Timed out after {self._bridge.environment_server.config.actor_call_timeout_seconds}s "
                f"waiting for {self.alias}"
            ) from error


class _GymEmbeddingFacade:
    """Route UserSim embeddings through the endpoint named by a Model Server reference."""

    def __init__(self, bridge: "_ConversationBridge") -> None:
        server = bridge.environment_server
        config = get_first_server_config_dict(
            server.server_client.global_config_dict,
            server.config.embedding_model.name,
        )
        base_url = str(config.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError("The UserSim embedding model requires a base_url")
        self.model_name = str(config.get("model") or server.config.embedding_model.name)
        self._url = f"{base_url}/embeddings"
        api_key = config.get("api_key")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout_seconds = server.config.actor_call_timeout_seconds

    async def agenerate_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await http_request(
                    method="POST",
                    url=self._url,
                    json={"input": list(texts), "model": self.model_name},
                    headers=self._headers,
                )
                await raise_for_status(response)
                response_data = await get_response_json(response)
        except TimeoutError as error:
            raise TimeoutError(f"Timed out after {self._timeout_seconds}s waiting for embedding_model") from error
        data = response_data.get("data")
        if not isinstance(data, list):
            raise ValueError("Embedding model response is missing data")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in ordered]
        if len(embeddings) != len(texts) or not all(isinstance(embedding, list) for embedding in embeddings):
            raise ValueError("Embedding model response does not contain one embedding per input")
        return embeddings


def _configured_model_name(environment_server: "UserSimEnvironmentServer", alias: str) -> str:
    """Resolve the upstream model ID behind a UserSim Agent or Model alias."""

    target = environment_server.config.target_for_alias(alias)
    target_config = get_first_server_config_dict(environment_server.server_client.global_config_dict, target.name)
    if isinstance(target, ModelServerRef):
        return str(target_config.get("model") or target.name)

    model_server = target_config.get("model_server") or {}
    model_server_name = model_server.get("name")
    if not model_server_name:
        return target.name
    model_config = get_first_server_config_dict(
        environment_server.server_client.global_config_dict,
        str(model_server_name),
    )
    return str(model_config.get("model") or model_server_name)


class _ConversationBridge:
    """Bridge UserSim model aliases to participant Agents and support Models."""

    def __init__(
        self,
        environment_server: "UserSimEnvironmentServer",
        request: UserSimEpisodeRequest,
        task: UserSimTaskInput,
        resources_cookies: dict[str, str],
        agent_sessions: dict[str, _AgentSession],
    ) -> None:
        self.environment_server = environment_server
        self.request = request
        self.task = task
        self.resources_cookies = resources_cookies
        self.agent_sessions = agent_sessions
        self.invocations: list[UserSimInvocation] = []

    async def invoke(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        parameters: Mapping[str, Any],
    ) -> SimpleNamespace:
        role = _INVOCATION_ROLE_BY_ALIAS[alias]
        base_params = self.task.role_request_params.get(role)
        if base_params is None:
            base_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        values = base_params.model_dump(mode="json", exclude_none=True)
        values["input"] = [item for message in messages for item in _to_responses_input_items(message)]
        _apply_activation_parameters(
            values,
            parameters,
            assistant_tools=list(parameters.get("tools") or []) if alias == "assistant_model" else None,
        )
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(values)

        target = self.environment_server.config.target_for_alias(alias)
        if alias in _PARTICIPANT_ALIASES:
            agent_session = self.agent_sessions[alias]
            try:
                response = await self.environment_server.server_client.post(
                    server_name=target.name,
                    url_path=self.environment_server.responses_path(
                        target.name,
                        self.request,
                        capture_training_tokens=alias == "assistant_model",
                    ),
                    json=request_params,
                    cookies=agent_session.cookies,
                )
                await raise_for_status(response)
                response_data = await get_response_json(response)
                trajectory_data = response_data.pop(_INTERNAL_TRAJECTORY_KEY, None)
                gym_response = NeMoGymResponse.model_validate(response_data)
            except Exception as error:
                raise _ParticipantActivationError(alias, error) from error
        else:
            response = await self.environment_server.server_client.post(
                server_name=target.name,
                url_path="/v1/responses",
                json=request_params,
            )
            await raise_for_status(response)
            response_data = await get_response_json(response)
            trajectory_data = response_data.pop(_INTERNAL_TRAJECTORY_KEY, None)
            gym_response = NeMoGymResponse.model_validate(response_data)
        if alias in _PARTICIPANT_ALIASES:
            response_cookies = _cookies(response)
            if response_cookies:
                agent_session.cookies = response_cookies

        self.invocations.append(
            UserSimInvocation(
                sequence=len(self.invocations),
                role=_INVOCATION_ROLE_BY_ALIAS[alias],
                request=request_params,
                response=gym_response,
                observations=(
                    _agent_observations(target.name, trajectory_data) if alias in _PARTICIPANT_ALIASES else None
                ),
            )
        )

        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(
                content=_response_text(gym_response),
                reasoning_content=_response_reasoning(gym_response) or None,
                tool_calls=_response_tool_calls(gym_response),
            ),
            usage=(
                SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                if usage is not None
                else None
            ),
        )

    async def invoke_activation(self, activation: Any) -> Any:
        """Execute one hosted UserSim activation and return its typed result."""
        from usersim.engine.external import ActivationResult, ActivationUsage

        try:
            async with asyncio.timeout(self.environment_server.config.actor_call_timeout_seconds):
                completion = await self.invoke(
                    activation.model_alias,
                    activation.messages,
                    parameters={
                        **activation.parameters,
                        "tools": list(activation.tools),
                    },
                )
        except TimeoutError as error:
            raise TimeoutError(
                f"Timed out after {self.environment_server.config.actor_call_timeout_seconds}s "
                f"waiting for {activation.model_alias}"
            ) from error
        usage = completion.usage
        return ActivationResult(
            activation_id=activation.activation_id,
            response=_completion_message(completion),
            usage=(
                ActivationUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                if usage is not None
                else None
            ),
        )

    def record_state(self, activation_index: int, state: dict[str, Any]) -> None:
        """Attach final UserSim evidence to the assistant activation that produced it."""
        self.invocations[activation_index] = self.invocations[activation_index].model_copy(
            update={"state_after": state}
        )


class UserSimEnvironmentServer(BaseEnvironmentServer[UserSimEpisodeRequest, UserSimEpisodeResponse]):
    """Run one UserSim ConversationLoop as a Gym episode."""

    ray_enabled = False
    config: UserSimEnvironmentServerConfig
    request_model = UserSimEpisodeRequest
    response_model = UserSimEpisodeResponse

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(
        self,
        request: UserSimEpisodeRequest,
        cleanup: CleanupContext,
    ) -> UserSimEpisodeResponse:
        task = request.task.task_input
        resources_session_id = f"resources-session-{uuid4().hex}"
        resources_cookies: dict[str, str] = {}

        async def close_resources() -> None:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                ),
                cookies=resources_cookies,
            )
            await raise_for_status(close_response)
            ResourcesCloseSessionResponse.model_validate(await get_response_json(close_response))

        resources_cleanup = cleanup.register_cleanup("resources session", close_resources)
        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=resources_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=task.model_dump(mode="json"),
                ),
            )
            resources_cookies.update(_cookies(seed_http_response))
            await raise_for_status(seed_http_response)
            if not resources_cookies:
                raise ValueError("Resources seed did not establish a session cookie")
            seed = UserSimSeedResponse.model_validate(await get_response_json(seed_http_response))
            if seed.resources_session_id != resources_session_id:
                raise ValueError("Resources seed returned a different resources_session_id")
        except Exception as error:
            raise self._failure("seed", error) from error

        agent_sessions: dict[str, _AgentSession] = {}
        agent_targets = {
            "user_model": self.config.user_agent,
            "assistant_model": self.config.assistant_agent,
        }
        for alias, target in agent_targets.items():
            agent_session_id = f"agent-session-{alias}-{uuid4().hex}"
            session = _AgentSession(
                alias=alias,
                target=target,
                session_id=agent_session_id,
                cookies={},
            )

            async def close_agent(current: _AgentSession = session) -> None:
                close_http_response = await self.server_client.post(
                    server_name=current.target.name,
                    url_path="/v1/agent_sessions/close",
                    json=AgentCloseSessionRequest(
                        agent_session_id=current.session_id,
                        episode_id=request.episode_id,
                    ),
                    cookies=current.cookies,
                )
                await raise_for_status(close_http_response)
                current.close_response = AgentCloseSessionResponse.model_validate(
                    await get_response_json(close_http_response)
                )

            session.cleanup = cleanup.register_cleanup(f"{alias} agent session", close_agent)
            try:
                session_request = AgentSeedSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=[],
                    sandbox_access=seed.sandbox_access if alias == "assistant_model" else None,
                )
                session_http_response = await self.server_client.post(
                    server_name=target.name,
                    url_path="/v1/agent_sessions",
                    json=session_request.model_dump(mode="json"),
                )
                session.cookies.update(_cookies(session_http_response))
                await raise_for_status(session_http_response)
                session_response = AgentSeedSessionResponse.model_validate(
                    await get_response_json(session_http_response)
                )
                if session_response.agent_session_id != agent_session_id:
                    raise ValueError(f"{alias} seed returned a different agent_session_id")
                if not session.cookies:
                    raise ValueError(f"{alias} seed did not establish a session cookie")
                agent_sessions[alias] = session
            except Exception as error:
                raise self._failure("agent", error) from error

        bridge = _ConversationBridge(
            self,
            request,
            task,
            resources_cookies,
            agent_sessions,
        )
        try:
            raw_result = await self._run_usersim(bridge, task.resolved_row)
            result = UserSimSimulationResult.model_validate(raw_result)
            _finalize_termination(bridge.invocations, result)
            if not any(invocation.role == "assistant" for invocation in bridge.invocations):
                raise ValueError("UserSim completed without an assistant_model invocation")
        except _ParticipantActivationError as error:
            if (
                error.alias != "assistant_model"
                or _is_retryable_dependency_error(error)
                or not _is_scorable_assistant_activation_error(error)
            ):
                raise self._failure("agent", error) from error
            result = UserSimSimulationResult(
                trajectory_id=task.resolved_row["trajectory_id"],
                conversation_messages=_participant_messages(bridge.invocations),
                conversation_status=False,
                simulation_outcome={
                    "status": "failed",
                    "failure_class": "assistant_activation_error",
                    "failure_attribution": "assistant_model",
                    "failure_detail": str(error)[:2000],
                    "failure_reason": str(error)[:2000],
                    "termination_reason": "assistant_model_activation_failed",
                },
            )
            _finalize_termination(bridge.invocations, result)
        except Exception as error:
            raise self._failure("agent", error) from error

        for alias in reversed(tuple(agent_targets)):
            session = agent_sessions[alias]
            try:
                if session.cleanup is None:
                    raise RuntimeError(f"{alias} cleanup was not registered")
                await session.cleanup.close()
            except Exception as error:
                raise self._failure("cleanup", error, terminal=True) from error
            if session.close_response is not None:
                if session.close_response.resources_cookies:
                    bridge.resources_cookies.clear()
                    bridge.resources_cookies.update(session.close_response.resources_cookies)

        try:
            verify_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=UserSimVerifyRequest(
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    verification_input=UserSimVerificationInput(
                        resolved_row=task.resolved_row,
                        usersim_result=result,
                        invocations=bridge.invocations,
                    ),
                ),
                cookies=bridge.resources_cookies,
            )
            await raise_for_status(verify_http_response)
            verification = UserSimVerification.model_validate(await get_response_json(verify_http_response))
            if verification.usersim_result is not None:
                result = verification.usersim_result
        except Exception as error:
            raise self._failure("verification", error) from error

        try:
            await resources_cleanup.close()
        except Exception as error:
            raise self._failure("cleanup", error, terminal=True) from error

        return UserSimEpisodeResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=UserSimEpisodeResult.from_verification(
                verification=verification,
                usersim_result=result,
                invocations=bridge.invocations,
            ),
        )

    async def _run_usersim(self, bridge: _ConversationBridge, resolved_row: dict[str, Any]) -> dict[str, Any]:
        from usersim.engine.external import (
            EpisodeLifecycleComplete,
            HostRoleModel,
            ProbeEpisodeRuntime,
        )

        models: dict[str, Any] = {
            alias: HostRoleModel(model_name=_configured_model_name(self, alias))
            for alias in ("user_model", "assistant_model", "judge_model", "summary_model")
        }
        models["api_response_model"] = _GymModelFacade("api_response_model", bridge)
        models["embedding_model"] = _GymEmbeddingFacade(bridge)
        runtime = await asyncio.to_thread(ProbeEpisodeRuntime.from_resolved_row, resolved_row, models=models)
        final_assistant_index: int | None = None
        try:
            event = await runtime.advance()
            while not isinstance(event, EpisodeLifecycleComplete):
                activation_result = await bridge.invoke_activation(event)
                activation_index = len(bridge.invocations) - 1
                if event.model_alias == "assistant_model":
                    final_assistant_index = activation_index
                    if policy_error := _assistant_response_validation_error(event, activation_result):
                        raise _ParticipantActivationError("assistant_model", policy_error)
                event = await runtime.advance(activation_result)
            if final_assistant_index is not None:
                bridge.record_state(final_assistant_index, await runtime.evidence())
            return event.result
        finally:
            await runtime.close()

    def responses_path(
        self,
        target_name: str,
        request: UserSimEpisodeRequest,
        *,
        capture_training_tokens: bool,
    ) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        target_config = get_first_server_config_dict(self.server_client.global_config_dict, target_name)
        token_capture = (
            capture_training_tokens
            and bool(block.get("enabled", False))
            and (bool(block.get("all_agents", False)) or bool(target_config.get("token_id_capture", False)))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses"

    @staticmethod
    def _failure(
        stage: FailureStage,
        error: Exception,
        *,
        terminal: bool | None = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            UserSimEpisodeFailure(
                stage=stage,
                failure_reason=f"{type(error).__name__}: {error}"[:2000],
                terminal=not _is_retryable_dependency_error(error) if terminal is None else terminal,
            )
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


def _to_responses_input_items(message: Any) -> list[dict[str, Any]]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {
            "role": getattr(message, "role"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        }
    role = getattr(value.get("role"), "value", value.get("role"))
    if role == "tool":
        call_id = value.get("tool_call_id")
        if not call_id:
            raise ValueError("UserSim tool message is missing tool_call_id")
        return [{"type": "function_call_output", "call_id": call_id, "output": value.get("content", "")}]
    if role not in {"system", "developer", "user", "assistant"}:
        raise NotImplementedError(f"UserSim message role {role!r} is not supported")
    items: list[dict[str, Any]] = []
    content = value.get("content", "")
    if content or not value.get("tool_calls"):
        items.append({"type": "message", "role": role, "content": content})
    for tool_call in value.get("tool_calls") or []:
        tool_value = (
            tool_call.model_dump(mode="json", exclude_none=True) if hasattr(tool_call, "model_dump") else tool_call
        )
        function = tool_value.get("function") if isinstance(tool_value, Mapping) else None
        if not isinstance(function, Mapping):
            raise ValueError(f"Invalid UserSim tool call: {tool_value!r}")
        items.append(
            {
                "type": "function_call",
                "call_id": tool_value["id"],
                "name": function["name"],
                "arguments": function.get("arguments", "{}"),
            }
        )
    return items


def _to_responses_tool(tool: Any) -> dict[str, Any]:
    value = tool.model_dump(mode="json", exclude_none=True) if hasattr(tool, "model_dump") else dict(tool)
    function = value.get("function")
    if not isinstance(function, Mapping):
        raise ValueError(f"Invalid UserSim function tool schema: {value!r}")
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description"),
        "parameters": function.get("parameters", {}),
        "strict": function.get("strict"),
    }


def _apply_activation_parameters(
    values: dict[str, Any],
    parameters: Mapping[str, Any],
    *,
    assistant_tools: list[dict[str, Any]] | None,
) -> None:
    translated = {
        "max_tokens",
        "max_completion_tokens",
        "tools",
        "tool_choice",
        "reasoning_effort",
        "response_format",
    }
    direct = {
        "include",
        "instructions",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "service_tier",
        "store",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
    unsupported = set(parameters) - translated - direct
    if unsupported:
        raise NotImplementedError(f"Unsupported UserSim activation options: {sorted(unsupported)}")
    for name in direct:
        if parameters.get(name) is not None:
            values[name] = parameters[name]
    max_tokens = parameters.get("max_tokens") or parameters.get("max_completion_tokens")
    if max_tokens is not None:
        values["max_output_tokens"] = max_tokens
    tools = parameters.get("tools")
    if assistant_tools is not None or tools:
        selected_tools = assistant_tools if assistant_tools is not None else list(tools)
        values["tools"] = [_to_responses_tool(tool) for tool in selected_tools]
    if parameters.get("tool_choice") is not None:
        values["tool_choice"] = parameters["tool_choice"]
    if parameters.get("reasoning_effort") is not None:
        values["reasoning"] = {"effort": parameters["reasoning_effort"]}
    response_format = parameters.get("response_format")
    if response_format is not None:
        json_schema = response_format.get("json_schema")
        if response_format.get("type") != "json_schema" or not isinstance(json_schema, Mapping):
            raise NotImplementedError(f"Unsupported response format: {response_format!r}")
        text_format = {
            "type": "json_schema",
            "name": json_schema["name"],
            "schema": json_schema["schema"],
        }
        if json_schema.get("strict") is not None:
            text_format["strict"] = json_schema["strict"]
        values["text"] = {"format": text_format}


def _completion_message(completion: SimpleNamespace) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": completion.message.content or "",
    }
    if completion.message.reasoning_content:
        message["reasoning_content"] = completion.message.reasoning_content
    if completion.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in completion.message.tool_calls
        ]
    return message


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


def _participant_messages(invocations: Sequence[UserSimInvocation]) -> list[dict[str, Any]]:
    return [
        {"role": invocation.role, "content": _response_text(invocation.response)}
        for invocation in invocations
        if invocation.role in _PARTICIPANT_ROLES
    ]


def _response_reasoning(response: NeMoGymResponse) -> str:
    chunks: list[str] = []
    for item in response.output:
        if not isinstance(item, NeMoGymResponseReasoningItem):
            continue
        for part in [*item.summary, *(item.content or [])]:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _response_tool_calls(response: NeMoGymResponse) -> list[SimpleNamespace] | None:
    calls = [
        SimpleNamespace(id=item.call_id, name=item.name, arguments_json=item.arguments)
        for item in response.output
        if isinstance(item, NeMoGymResponseFunctionToolCall)
    ]
    return calls or None


def _assistant_response_validation_error(activation: Any, result: Any) -> ValueError | None:
    """Return a policy-attributed error for invalid Assistant tool calls."""
    tool_calls = result.response.get("tool_calls") or []
    if not tool_calls:
        return None
    allowed_tools = {_to_responses_tool(tool)["name"] for tool in activation.tools}
    if not allowed_tools:
        return ValueError(f"Assistant activation {activation.activation_id!r} returned tool calls with tools disabled")
    seen_call_ids: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            return ValueError("Assistant tool calls must be mappings")
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            return ValueError("Assistant tool calls require a non-empty id")
        if call_id in seen_call_ids:
            return ValueError(f"Assistant tool call id {call_id!r} is repeated in one response")
        seen_call_ids.add(call_id)
        function = tool_call.get("function")
        tool_name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(tool_name, str) or tool_name not in allowed_tools:
            return ValueError(
                f"Assistant called unoffered tool {tool_name!r}; available tools: {sorted(allowed_tools)}"
            )
    return None


def _finalize_termination(invocations: list[UserSimInvocation], result: UserSimSimulationResult) -> None:
    participant_indexes = [
        index for index, invocation in enumerate(invocations) if invocation.role in _PARTICIPANT_ROLES
    ]
    if not participant_indexes:
        return
    metadata = result.conversation_metadata or {}
    reason = metadata.get("termination_reason") or result.simulation_outcome.get("termination_reason")
    if not isinstance(reason, str) or not reason:
        reason = None
    if reason is None and (metadata.get("early_stop") or result.simulation_outcome.get("early_stop")):
        reason = "usersim_early_stop"
    if reason is None:
        reason = "usersim_completed" if result.conversation_status else "usersim_incomplete"
    final_index = participant_indexes[-1]
    invocations[final_index] = invocations[final_index].model_copy(update={"termination_reason": reason})


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, _ParticipantActivationError):
        original = error.error
        if isinstance(original, ClientResponseError):
            return original.status in {404, 408, 409, 425, 429} or original.status >= 500
        return isinstance(original, (ClientError, TimeoutError))
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    return isinstance(error, (ClientError, TimeoutError))


def _is_scorable_assistant_activation_error(error: _ParticipantActivationError) -> bool:
    original = error.error
    if isinstance(original, ClientResponseError):
        return 400 <= original.status < 500 and original.status not in {404, 408, 409, 425, 429}
    return isinstance(original, (AttributeError, TypeError, ValueError))


if __name__ == "__main__":
    UserSimEnvironmentServer.run_webserver()
