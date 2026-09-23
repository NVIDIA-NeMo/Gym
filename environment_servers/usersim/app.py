# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo UserSim's conversation protocol through Gym servers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

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
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ToolCallObservation, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.tool_access import (
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ToolAccess,
)
from resources_servers.usersim.types import (
    USERSIM_MODEL_ALIASES,
    UserSimEpisodeFailure,
    UserSimEpisodeRequest,
    UserSimEpisodeResponse,
    UserSimEpisodeResult,
    UserSimInvocation,
    UserSimProtocolConfig,
    UserSimScenario,
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
}
_PARTICIPANT_ROLES = {"user", "assistant"}


class UserSimEnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind UserSim's model aliases to Gym Agent and Model Servers."""

    model_config = ConfigDict(extra="forbid")

    user_agent: AgentServerRef
    assistant_agent: AgentServerRef
    judge_model: ModelServerRef
    summary_model: ModelServerRef
    resources_server: ResourcesServerRef
    resources_tool_transports: list[Literal["direct_http", "mcp"]] = Field(default_factory=list)
    max_turns: int = Field(5, ge=1)
    actor_call_timeout_seconds: float = Field(300.0, gt=0)
    protocol_config: UserSimProtocolConfig = Field(default_factory=UserSimProtocolConfig)

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


class _GymModelFacade:
    """Synchronous model facade expected by UserSim's generator."""

    def __init__(self, alias: str, bridge: "_ConversationBridge") -> None:
        self.alias = alias
        self.model_name = bridge.environment_server.config.target_for_alias(alias).name
        self._bridge = bridge

    def completion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        unsupported = set(kwargs) - {"max_tokens", "max_completion_tokens", "tools"}
        if unsupported:
            raise NotImplementedError(f"Unsupported UserSim completion options: {sorted(unsupported)}")
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        return self._bridge.complete_from_worker(
            self.alias,
            messages,
            max_tokens=max_tokens,
            tools=kwargs.get("tools"),
        )


class _ResourcesOwnedModelFacade:
    """Placeholder for UserSim aliases that the Resources Server owns."""

    model_name = "resources-server"

    def completion(self, _messages: Sequence[Any], **_kwargs: Any) -> SimpleNamespace:
        raise RuntimeError(
            "UserSim attempted to invoke api_response_model in the Environment Server; "
            "API-response synthesis must run through the Resources Server tool endpoint"
        )


def _create_usersim_generator(
    generator_type: type[Any],
    config: Any,
    models: Mapping[str, Any],
) -> Any:
    """Instantiate UserSim's generator with Gym-backed model lookup."""

    class _GymConversationGenerator(generator_type):
        def __init__(self) -> None:
            self.config = config
            self._models = models

        def get_model(self, alias: str) -> Any:
            return self._models[alias]

    return _GymConversationGenerator()


class _ConversationBridge:
    """Bridge UserSim's synchronous model facade to Gym's async clients."""

    def __init__(
        self,
        environment_server: "UserSimEnvironmentServer",
        request: UserSimEpisodeRequest,
        task: UserSimTaskInput,
        event_loop: asyncio.AbstractEventLoop,
        resources_cookies: dict[str, str],
        agent_sessions: dict[str, _AgentSession],
        assistant_tools: list[dict[str, Any]],
    ) -> None:
        self.environment_server = environment_server
        self.request = request
        self.task = task
        self.event_loop = event_loop
        self.resources_cookies = resources_cookies
        self.agent_sessions = agent_sessions
        self.assistant_tools = assistant_tools
        self.invocations: list[UserSimInvocation] = []

    def complete_from_worker(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
        tools: Sequence[Any] | None = None,
    ) -> SimpleNamespace:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.event_loop:
            raise RuntimeError("UserSim's synchronous ConversationLoop must run outside the Environment Server loop")

        future = asyncio.run_coroutine_threadsafe(
            self._invoke(alias, messages, max_tokens=max_tokens, tools=tools),
            self.event_loop,
        )
        try:
            return future.result(timeout=self.environment_server.config.actor_call_timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError(
                f"Timed out after {self.environment_server.config.actor_call_timeout_seconds}s waiting for {alias}"
            ) from error

    async def _invoke(
        self,
        alias: str,
        messages: Sequence[Any],
        *,
        max_tokens: int | None,
        tools: Sequence[Any] | None,
    ) -> SimpleNamespace:
        base_params = self.task.model_responses_create_params.get(alias)
        if base_params is None:
            base_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        values = base_params.model_dump(mode="json", exclude_none=True)
        values["input"] = [_to_responses_input(message) for message in messages]
        if max_tokens is not None:
            values["max_output_tokens"] = max_tokens
        if tools:
            if alias != "assistant_model":
                raise ValueError(f"UserSim requested tools for non-Assistant alias {alias!r}")
            values["tools"] = [_to_responses_tool(tool) for tool in self.assistant_tools]
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(values)

        target = self.environment_server.config.target_for_alias(alias)
        agent_session = self.agent_sessions.get(alias)
        response = await self.environment_server.server_client.post(
            server_name=target.name,
            url_path=self.environment_server.responses_path(target.name, self.request),
            json=request_params,
            cookies=agent_session.cookies if agent_session is not None else None,
        )
        await raise_for_status(response)
        response_data = await get_response_json(response)
        trajectory_data = response_data.pop(_INTERNAL_TRAJECTORY_KEY, None)
        gym_response = NeMoGymResponse.model_validate(response_data)
        if agent_session is not None:
            response_cookies = _cookies(response)
            if response_cookies:
                agent_session.cookies = response_cookies

        self.invocations.append(
            UserSimInvocation(
                sequence=len(self.invocations),
                role=_INVOCATION_ROLE_BY_ALIAS[alias],
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
                tool_calls=None,
            ),
            usage=(
                SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                if usage is not None
                else None
            ),
        )

    def attach_session_observations(self, alias: str, observations: AgentObservationBundle | None) -> None:
        if observations is None:
            return
        for index in range(len(self.invocations) - 1, -1, -1):
            invocation = self.invocations[index]
            if invocation.role == _INVOCATION_ROLE_BY_ALIAS[alias]:
                self.invocations[index] = invocation.model_copy(update={"observations": observations})
                return


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
        resources_cookies: dict[str, str]
        try:
            seed_http_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
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
        tool_accesses = self._resources_tool_accesses(seed, resources_cookies)

        agent_sessions: dict[str, _AgentSession] = {}
        for alias, target in (
            ("user_model", self.config.user_agent),
            ("assistant_model", self.config.assistant_agent),
        ):
            try:
                session_http_response = await self.server_client.post(
                    server_name=target.name,
                    url_path="/v1/agent_sessions",
                    json=AgentSeedSessionRequest(
                        episode_id=request.episode_id,
                        task_id=request.task.task_id,
                        tool_accesses=tool_accesses if alias == "assistant_model" else [],
                        sandbox_access=seed.sandbox_access,
                    ),
                )
                await raise_for_status(session_http_response)
                session_response = AgentSeedSessionResponse.model_validate(
                    await get_response_json(session_http_response)
                )
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
            asyncio.get_running_loop(),
            resources_cookies,
            agent_sessions,
            seed.assistant_tools,
        )
        try:
            raw_result = await asyncio.to_thread(self._run_usersim, bridge, seed.scenario)
            result = UserSimSimulationResult.model_validate(raw_result)
            _finalize_termination(bridge.invocations, result)
            if not any(invocation.role == "assistant" for invocation in bridge.invocations):
                raise ValueError("UserSim completed without an assistant_model invocation")
        except Exception as error:
            raise self._failure("simulation", error) from error

        for alias in ("assistant_model", "user_model"):
            session = agent_sessions[alias]
            try:
                if session.cleanup is None:
                    raise RuntimeError(f"{alias} cleanup was not registered")
                await session.cleanup.close()
            except Exception as error:
                raise self._failure("cleanup", error, terminal=True) from error
            if session.close_response is not None:
                bridge.attach_session_observations(alias, session.close_response.agent_observations)
                if session.close_response.resources_cookies is not None:
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
                        scenario=seed.scenario,
                        usersim_context=seed.usersim_context,
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

    def _run_usersim(self, bridge: _ConversationBridge, scenario: UserSimScenario) -> dict[str, Any]:
        from usersim.engine.config import ConversationSimulatorConfig
        from usersim.engine.core.llm import set_debug_log_path
        from usersim.engine.generator import ConversationSimulatorGenerator

        set_debug_log_path(None)
        config_values = self.config.protocol_config.model_dump(mode="python", exclude_none=True)
        config_values.update(
            {"name": "conversation_messages", "locale": scenario.locale, "max_turns": self.config.max_turns}
        )
        config = ConversationSimulatorConfig.model_validate(config_values)
        models: dict[str, Any] = {alias: _GymModelFacade(alias, bridge) for alias in USERSIM_MODEL_ALIASES}
        # UserSim currently resolves all model aliases eagerly. This alias is used only by
        # probe tool runtimes, which execute in the Resources Server.
        models["api_response_model"] = _ResourcesOwnedModelFacade()
        generator = _create_usersim_generator(ConversationSimulatorGenerator, config, models)
        data = scenario.model_dump(mode="python", exclude={"locale", "probe_data"})
        data.update(scenario.probe_data)
        return generator.generate(data)

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
        seed: UserSimSeedResponse,
        resources_cookies: dict[str, str],
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
                )
            )
        if "mcp" in self.config.resources_tool_transports:
            if seed.resources_tools is None:
                raise self._failure(
                    "seed",
                    ValueError("Resources seed did not return requested MCP metadata"),
                    terminal=True,
                )
            if seed.resources_tools.transport != "http":
                raise self._failure(
                    "seed",
                    ValueError(f"Unsupported resources MCP transport: {seed.resources_tools.transport}"),
                    terminal=True,
                )
            accesses.append(
                MCPToolAccess(
                    name=seed.resources_tools.server_name,
                    required=True,
                    connection=MCPStreamableHTTPConnection(
                        url=f"{resources_base_url}/{seed.resources_tools.url_path.lstrip('/')}",
                        headers=seed.resources_tools.headers,
                    ),
                )
            )
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


def _to_responses_input(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {"role": getattr(message, "role"), "content": getattr(message, "content", "")}
    role = getattr(value.get("role"), "value", value.get("role"))
    if role not in {"system", "developer", "user", "assistant"}:
        raise NotImplementedError(f"UserSim message role {role!r} is not supported")
    return {"type": "message", "role": role, "content": value.get("content", "")}


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
