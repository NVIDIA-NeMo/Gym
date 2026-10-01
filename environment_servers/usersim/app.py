# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo UserSim's conversation protocol through Gym servers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
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
    AgentToolLoopPolicy,
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
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ToolCallObservation, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.tool_access import (
    DirectHTTPToolAccess,
    ToolAccess,
)
from resources_servers.usersim.episode_contracts import (
    ActivationRequest,
    ActivationResult,
    ActivationUsage,
    AssistantTurnRequest,
    CompletedAssistantTurn,
    CompletedTurnEvidence,
    ToolTurnContext,
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
from resources_servers.usersim.response_format import responses_json_schema


_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"
_INTERNAL_COMPLETED_TURN_KEY = "_ng_completed_turn"
_INVOCATION_ROLE_BY_ALIAS = {
    "user_model": "user",
    "assistant_model": "assistant",
    "judge_model": "judge",
    "summary_model": "summary",
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
    resources_server: ResourcesServerRef
    resources_tool_transports: list[Literal["direct_http", "mcp"]] = Field(default_factory=list)
    actor_call_timeout_seconds: float = Field(300.0, gt=0)

    def target_for_alias(self, alias: str) -> AgentServerRef | ModelServerRef:
        return {
            "user_model": self.user_agent,
            "assistant_model": self.assistant_agent,
            "judge_model": self.judge_model,
            "summary_model": self.summary_model,
        }[alias]


@dataclass
class _AgentSession:
    alias: str
    target: AgentServerRef
    session_id: str
    cookies: dict[str, str]
    cleanup: CleanupHandle | None = None
    close_response: AgentCloseSessionResponse | None = None


class _ConversationBridge:
    """Bridge UserSim model aliases to participant Agents and support Models."""

    def __init__(
        self,
        environment_server: "UserSimEnvironmentServer",
        request: UserSimEpisodeRequest,
        task: UserSimTaskInput,
        resources_cookies: dict[str, str],
        agent_sessions: dict[str, _AgentSession],
        assistant_tools: list[dict[str, Any]],
    ) -> None:
        self.environment_server = environment_server
        self.request = request
        self.task = task
        self.resources_cookies = resources_cookies
        self.agent_sessions = agent_sessions
        self.assistant_tools = assistant_tools
        self.invocations: list[UserSimInvocation] = []

    async def invoke(self, activation: ActivationRequest) -> ActivationResult:
        alias = activation.model_alias
        role = _INVOCATION_ROLE_BY_ALIAS[alias]
        if activation.role != role:
            raise ValueError(
                f"Activation {activation.activation_id!r} role {activation.role!r} does not match alias {alias!r}"
            )
        base_params = self.task.responses_create_params.get(role)
        if base_params is None:
            base_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        values = base_params.model_dump(mode="json", exclude_none=True)
        values["input"] = [item for message in activation.messages for item in _to_responses_input_items(message)]
        _apply_activation_parameters(
            values,
            activation.parameters,
            assistant_tools=self.assistant_tools if alias == "assistant_model" else None,
        )
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(values)

        target = self.environment_server.config.target_for_alias(alias)
        try:
            async with asyncio.timeout(self.environment_server.config.actor_call_timeout_seconds):
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
        except TimeoutError as error:
            raise TimeoutError(
                f"Timed out after {self.environment_server.config.actor_call_timeout_seconds}s waiting for {alias}"
            ) from error
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
        usage = _activation_usage(gym_response)
        return ActivationResult(
            activation_id=activation.activation_id,
            response=_response_chat_message(gym_response),
            usage=usage,
        )

    async def invoke_assistant(self, turn: AssistantTurnRequest) -> CompletedAssistantTurn:
        session = await self.environment_server._open_assistant_turn_session(
            self.request,
            turn,
            self.resources_cookies,
        )
        try:
            base_params = self.task.responses_create_params.get("assistant")
            values = (
                base_params.model_dump(mode="json", exclude_none=True) if base_params is not None else {"input": []}
            )
            values["input"] = [item for message in turn.messages for item in _to_responses_input_items(message)]
            _apply_activation_parameters(values, turn.parameters, assistant_tools=turn.tools)
            request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(values)
            response = await self.environment_server.server_client.post(
                server_name=session.target.name,
                url_path=self.environment_server.responses_path(session.target.name, self.request),
                json=request_params,
                cookies=session.cookies,
            )
            await raise_for_status(response)
            session.cookies.update(_cookies(response))
            response_data = await get_response_json(response)
            trajectory_data = response_data.pop(_INTERNAL_TRAJECTORY_KEY, None)
            completed_trace = response_data.pop(_INTERNAL_COMPLETED_TURN_KEY, None)
            gym_response = NeMoGymResponse.model_validate(response_data)
            if not isinstance(completed_trace, Mapping):
                raise ValueError("Assistant Agent did not return its completed-turn trace")
            self.invocations.append(
                UserSimInvocation(
                    sequence=len(self.invocations),
                    role="assistant",
                    request=request_params,
                    response=gym_response,
                    observations=_agent_observations(session.target.name, trajectory_data),
                )
            )
        finally:
            await self.environment_server._close_agent_session(session, self.request)
            if session.close_response and session.close_response.resources_cookies:
                self.resources_cookies.clear()
                self.resources_cookies.update(session.close_response.resources_cookies)
        resources_context = completed_trace.get("resources_context")
        evidence = (
            CompletedTurnEvidence.model_validate(resources_context)
            if isinstance(resources_context, Mapping)
            else CompletedTurnEvidence.from_unexecuted_turn(turn.tool_context)
        )
        return CompletedAssistantTurn.model_validate(
            {
                "turn_id": turn.turn_id,
                "transcript": completed_trace.get("transcript"),
                "model_calls": completed_trace.get("model_calls"),
                "evidence": evidence.model_dump(mode="json"),
            }
        )


class UserSimEnvironmentServer(BaseEnvironmentServer[UserSimEpisodeRequest, UserSimEpisodeResponse]):
    """Run one UserSim ConversationLoop as a native Gym episode."""

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
            seed.assistant_tools,
        )
        try:
            runtime = self._create_conversation_runtime(seed.resolved_row, task)
            raw_result = await self._run_usersim(runtime, bridge)
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
            result=UserSimEpisodeResult(
                verification=verification,
                usersim_result=result,
                invocations=bridge.invocations,
            ),
        )

    async def _run_usersim(self, runtime: Any, bridge: _ConversationBridge) -> dict[str, Any]:
        """Drive the Environment-owned outer lifecycle at completed-turn boundaries."""
        try:
            event = await runtime.advance()
            while not getattr(event, "result", None):
                value = event.to_dict()
                if value.get("role") == "assistant":
                    result = await bridge.invoke_assistant(AssistantTurnRequest.model_validate(value))
                else:
                    result = await bridge.invoke(ActivationRequest.model_validate(value))
                event = await runtime.advance(result.model_dump(mode="json"))
            return event.result
        finally:
            await runtime.close()

    def _create_conversation_runtime(self, resolved_row: dict[str, Any], task: UserSimTaskInput) -> Any:
        from usersim.engine.external import ConversationRuntime, HostRoleModel

        models: dict[str, Any] = {}
        for alias, role in _INVOCATION_ROLE_BY_ALIAS.items():
            target = self.config.target_for_alias(alias)
            target_config = get_first_server_config_dict(self.server_client.global_config_dict, target.name)
            if isinstance(target, AgentServerRef):
                model_ref = target_config.get("model_server")
                if not isinstance(model_ref, Mapping) or not model_ref.get("name"):
                    raise ValueError(f"Agent {target.name!r} does not declare a model_server")
                target_config = get_first_server_config_dict(
                    self.server_client.global_config_dict,
                    str(model_ref["name"]),
                )
            model_name = target_config.get("model")
            if not isinstance(model_name, str) or not model_name:
                raise ValueError(f"Configured server for {alias!r} does not declare its model identity")
            params = task.responses_create_params.get(role)
            models[alias] = HostRoleModel(
                model_name=model_name,
                max_tokens=params.max_output_tokens if params is not None else None,
            )
        return ConversationRuntime.from_resolved_row(resolved_row, models=models)

    async def _open_assistant_turn_session(
        self,
        request: UserSimEpisodeRequest,
        turn: AssistantTurnRequest,
        resources_cookies: dict[str, str],
    ) -> _AgentSession:
        target = self.config.assistant_agent
        agent_session_id = f"agent-session-assistant-{turn.turn_id}-{uuid4().hex}"
        response = await self.server_client.post(
            server_name=target.name,
            url_path="/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id=agent_session_id,
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                tool_accesses=self._resources_tool_accesses(
                    resources_cookies,
                    tool_context=turn.tool_context,
                ),
                tool_loop_policy=AgentToolLoopPolicy.model_validate(turn.loop_policy.model_dump(mode="json")),
            ).model_dump(mode="json"),
        )
        await raise_for_status(response)
        seeded = AgentSeedSessionResponse.model_validate(await get_response_json(response))
        cookies = _cookies(response)
        if seeded.agent_session_id != agent_session_id or not cookies:
            raise ValueError("Assistant turn session was not established")
        return _AgentSession(alias="assistant_model", target=target, session_id=agent_session_id, cookies=cookies)

    async def _close_agent_session(self, session: _AgentSession, request: UserSimEpisodeRequest) -> None:
        response = await self.server_client.post(
            server_name=session.target.name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=session.session_id,
                episode_id=request.episode_id,
            ),
            cookies=session.cookies,
        )
        await raise_for_status(response)
        session.close_response = AgentCloseSessionResponse.model_validate(await get_response_json(response))

    def responses_path(self, target_name: str, request: UserSimEpisodeRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        target_config = get_first_server_config_dict(self.server_client.global_config_dict, target_name)
        token_capture = bool(block.get("enabled", False)) and (
            bool(block.get("all_agents", False)) or bool(target_config.get("token_id_capture", False))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses"

    def _resources_tool_accesses(
        self,
        resources_cookies: dict[str, str],
        *,
        tool_context: ToolTurnContext,
    ) -> list[ToolAccess]:
        resources_base_url = self.server_client._resolve_base_url(self.config.resources_server.name).rstrip("/")
        accesses: list[ToolAccess] = []
        if "direct_http" in self.config.resources_tool_transports:
            accesses.append(
                DirectHTTPToolAccess(
                    name=f"{self.config.resources_server.name}.direct_http",
                    required=True,
                    base_url=resources_base_url,
                    cookies=resources_cookies,
                    tool_call_context=tool_context.model_dump(mode="json"),
                )
            )
        if "mcp" in self.config.resources_tool_transports:
            raise NotImplementedError("MCP does not support completed-turn semantic tool rounds")
        return accesses

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
        value = {"role": getattr(message, "role"), "content": getattr(message, "content", "")}
    role = getattr(value.get("role"), "value", value.get("role"))
    if role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": value["tool_call_id"],
                "output": value.get("content", ""),
            }
        ]
    if role == "assistant" and value.get("tool_calls"):
        items = []
        if value.get("content"):
            items.append({"type": "message", "role": role, "content": value["content"]})
        for call in value["tool_calls"]:
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("Assistant tool call must contain a function mapping")
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": function.get("arguments", "{}"),
                }
            )
        return items
    if role not in {"system", "developer", "user", "assistant"}:
        raise NotImplementedError(f"UserSim message role {role!r} is not supported")
    return [{"type": "message", "role": role, "content": value.get("content", "")}]


def _to_responses_tool(tool: Any) -> dict[str, Any]:
    value = tool.model_dump(mode="json", exclude_none=True) if hasattr(tool, "model_dump") else dict(tool)
    function = value.get("function")
    if isinstance(function, Mapping):
        value = dict(function)
    if not value.get("name"):
        raise ValueError(f"Invalid UserSim function tool schema: {value!r}")
    return {
        "type": "function",
        "name": value["name"],
        "description": value.get("description"),
        "parameters": value.get("parameters", {}),
        "strict": value.get("strict", False),
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
                "schema": responses_json_schema(json_schema["schema"], strict=strict),
                "strict": strict,
            }
        }


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


def _response_chat_message(response: NeMoGymResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": _response_text(response)}
    reasoning = _response_reasoning(response)
    if reasoning:
        message["reasoning_content"] = reasoning
    calls = [
        {
            "id": item.call_id,
            "type": "function",
            "function": {"name": item.name, "arguments": item.arguments},
        }
        for item in response.output
        if isinstance(item, NeMoGymResponseFunctionToolCall)
    ]
    if calls:
        message["tool_calls"] = calls
    return message


def _activation_usage(response: NeMoGymResponse) -> ActivationUsage | None:
    if response.usage is None:
        return None
    return ActivationUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


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


def _response_output_messages(response: NeMoGymResponse) -> list[dict[str, Any]]:
    """Convert one complete Agent activation to canonical chat messages."""
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []

    def flush_calls() -> None:
        if pending_calls:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": list(pending_calls),
            }
            if pending_reasoning:
                message["reasoning_content"] = "\n".join(pending_reasoning)
            messages.append(message)
            pending_calls.clear()
            pending_reasoning.clear()

    for item in response.output:
        if isinstance(item, NeMoGymResponseReasoningItem):
            for part in [*item.summary, *(item.content or [])]:
                text = getattr(part, "text", None)
                if text:
                    pending_reasoning.append(text)
            continue
        if isinstance(item, NeMoGymResponseFunctionToolCall):
            pending_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )
            continue
        flush_calls()
        if isinstance(item, NeMoGymFunctionCallOutput):
            messages.append(
                {
                    "role": "tool",
                    "content": item.output if isinstance(item.output, str) else json.dumps(item.output),
                    "tool_call_id": item.call_id,
                }
            )
        elif isinstance(item, NeMoGymResponseOutputMessage):
            message = {"role": "assistant", "content": _response_message_text(item)}
            if pending_reasoning:
                message["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning.clear()
            messages.append(message)
    flush_calls()
    if pending_reasoning:
        messages.append({"role": "assistant", "content": "", "reasoning_content": "\n".join(pending_reasoning)})
    return messages


def _response_message_text(message: NeMoGymResponseOutputMessage) -> str:
    return "\n".join(
        text
        for content in message.content
        if (text := getattr(content, "text", None) or getattr(content, "refusal", None))
    )


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
