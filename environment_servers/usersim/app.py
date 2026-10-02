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

from aiohttp import ClientConnectionError, ClientResponseError
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
from resources_servers.usersim.episode_contracts import (
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
_USERSIM_MODEL_ALIASES = tuple(_INVOCATION_ROLE_BY_ALIAS)
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
    resources_server: ResourcesServerRef
    actor_call_timeout_seconds: float = Field(300.0, gt=0)

    def target_for_alias(self, alias: str) -> AgentServerRef | ModelServerRef:
        return {
            "user_model": self.user_agent,
            "assistant_model": self.assistant_agent,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
            "api_response_model": self.tool_simulation_model,
        }[alias]


@dataclass
class _AgentSession:
    alias: str
    target: AgentServerRef
    session_id: str
    cookies: dict[str, str]
    cleanup: CleanupHandle | None = None
    close_response: AgentCloseSessionResponse | None = None


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
        base_params = self.task.responses_create_params.get(role)
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
            response = await self.environment_server.server_client.post(
                server_name=target.name,
                url_path=self.environment_server.responses_path(target.name, self.request),
                json=request_params,
                cookies=agent_session.cookies,
            )
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
                        **({"tools": list(activation.tools)} if activation.tools else {}),
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

    def record_state(self, state: dict[str, Any]) -> None:
        """Attach UserSim's post-activation evidence to the latest invocation."""
        if not self.invocations:
            return
        self.invocations[-1] = self.invocations[-1].model_copy(update={"state_after": state})


class UserSimEnvironmentServer(BaseEnvironmentServer[UserSimEpisodeRequest, UserSimEpisodeResponse]):
    """Run one UserSim ConversationLoop as a native Gym episode."""

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
        resources_cookies: dict[str, str]
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
            await raise_for_status(seed_http_response)
            resources_cookies = _cookies(seed_http_response)
            if not resources_cookies:
                raise ValueError("Resources seed did not establish a session cookie")
            seed = UserSimSeedResponse.model_validate(await get_response_json(seed_http_response))
            if seed.resources_session_id != resources_session_id:
                raise ValueError("Resources seed returned a different resources_session_id")
        except Exception as error:
            raise self._failure("seed", error) from error

        async def close_resources() -> None:
            close_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close_session",
                json=ResourcesCloseSessionRequest(
                    resources_session_id=seed.resources_session_id,
                    episode_id=request.episode_id,
                ),
                cookies=resources_cookies,
            )
            await raise_for_status(close_response)
            ResourcesCloseSessionResponse.model_validate(await get_response_json(close_response))

        resources_cleanup = cleanup.register_cleanup("resources session", close_resources)
        agent_sessions: dict[str, _AgentSession] = {}
        agent_targets = {
            "user_model": self.config.user_agent,
            "assistant_model": self.config.assistant_agent,
        }
        for alias, target in agent_targets.items():
            try:
                agent_session_id = f"agent-session-{alias}-{uuid4().hex}"
                session_request = AgentSeedSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=[],
                    sandbox_access=seed.sandbox_access if alias in {"user_model", "assistant_model"} else None,
                )
                session_http_response = await self.server_client.post(
                    server_name=target.name,
                    url_path="/v1/agent_sessions",
                    json=session_request.model_dump(mode="json"),
                )
                await raise_for_status(session_http_response)
                session_response = AgentSeedSessionResponse.model_validate(
                    await get_response_json(session_http_response)
                )
                if session_response.agent_session_id != agent_session_id:
                    raise ValueError(f"{alias} seed returned a different agent_session_id")
                session = _AgentSession(
                    alias=alias,
                    target=target,
                    session_id=session_response.agent_session_id,
                    cookies=_cookies(session_http_response),
                )
                if not session.cookies:
                    raise ValueError(f"{alias} seed did not establish a session cookie")
                agent_sessions[alias] = session
            except Exception as error:
                raise self._failure("participant", error) from error

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

        bridge = _ConversationBridge(
            self,
            request,
            task,
            resources_cookies,
            agent_sessions,
        )
        try:
            raw_result = await self._run_usersim(bridge, seed)
            result = UserSimSimulationResult.model_validate(raw_result)
            _finalize_termination(bridge.invocations, result)
            if not any(invocation.role == "assistant" for invocation in bridge.invocations):
                raise ValueError("UserSim completed without an assistant_model invocation")
        except Exception as error:
            raise self._failure("simulation", error) from error

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
                        resolved_row=seed.resolved_row,
                        usersim_result=result,
                        invocations=bridge.invocations,
                    ),
                ),
                cookies=bridge.resources_cookies,
            )
            await raise_for_status(verify_http_response)
            verification = UserSimVerification.model_validate(await get_response_json(verify_http_response))
            if verification.native_usersim_result is not None:
                result = verification.native_usersim_result
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

    async def _run_usersim(self, bridge: _ConversationBridge, seed: UserSimSeedResponse) -> dict[str, Any]:
        from usersim.engine.external import EpisodeLifecycleComplete, HostRoleModel, ProbeEpisodeRuntime

        models: dict[str, Any] = {
            alias: HostRoleModel(model_name=_configured_model_name(self, alias))
            for alias in ("user_model", "assistant_model", "judge_model", "summary_model")
        }
        models["api_response_model"] = _GymModelFacade("api_response_model", bridge)
        runtime = ProbeEpisodeRuntime.from_resolved_row(seed.resolved_row, models=models)
        try:
            event = await runtime.advance()
            while not isinstance(event, EpisodeLifecycleComplete):
                activation_result = await bridge.invoke_activation(event)
                event = await runtime.advance(activation_result)
                bridge.record_state(await runtime.evidence())
            return event.result
        finally:
            await runtime.close()

    def responses_path(self, target_name: str, request: UserSimEpisodeRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        target_config = get_first_server_config_dict(self.server_client.global_config_dict, target_name)
        token_capture = bool(block.get("enabled", False)) and (
            bool(block.get("all_agents", False)) or bool(target_config.get("token_id_capture", False))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses"

    @staticmethod
    def _failure(
        stage: str,
        error: Exception,
        *,
        terminal: bool | None = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            UserSimEpisodeFailure(
                stage=stage,
                message=f"{type(error).__name__}: {error}"[:2000],
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
    if assistant_tools or tools:
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
        strict = json_schema.get("strict", True)
        values["text"] = {
            "format": {
                "type": "json_schema",
                "name": json_schema["name"],
                "schema": json_schema["schema"],
                "strict": strict,
            }
        }


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


def _finalize_termination(invocations: list[UserSimInvocation], result: UserSimSimulationResult) -> None:
    participant_indexes = [
        index for index, invocation in enumerate(invocations) if invocation.role in _PARTICIPANT_ROLES
    ]
    if not participant_indexes:
        return
    metadata = result.conversation_metadata or {}
    reason = next(
        (
            invocations[index].termination_reason
            for index in reversed(participant_indexes)
            if invocations[index].termination_reason
        ),
        None,
    )
    if reason is None and (metadata.get("early_stop") or result.simulation_outcome.get("early_stop")):
        reason = "usersim_early_stop"
    if reason is None:
        reason = "usersim_completed" if result.conversation_status else "usersim_incomplete"
    final_index = participant_indexes[-1]
    invocations[final_index] = invocations[final_index].model_copy(update={"termination_reason": reason})


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    return isinstance(error, (ClientConnectionError, TimeoutError))


if __name__ == "__main__":
    UserSimEnvironmentServer.run_webserver()
