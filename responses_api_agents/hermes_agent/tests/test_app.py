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
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.agents import (
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    SandboxPythonRuntime,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.agents.base_responses_api_agent import _AgentSessionRecord
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    EpisodeId,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseReasoningItem,
    TaskId,
)
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxExecResult, SandboxSpec
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.trajectory import AgentEpisode, AgentObservationBundle
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _gym_mcp_tool_name,
    _split_input_to_user_and_history,
    _trajectory_to_output_items,
)
from responses_api_agents.hermes_agent.sandbox import (
    _HERMES_RUNTIME,
    HermesSandboxSession,
)


class _FakeResponse:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _config(**kwargs) -> HermesAgentConfig:
    return HermesAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        **kwargs,
    )


def _mcp_access(name: str, *, required: bool = True) -> MCPToolAccess:
    return MCPToolAccess(
        name=name,
        required=required,
        connection=MCPStreamableHTTPConnection(
            url="http://resources:8000/mcp", headers={"X-NeMo-Gym-Session-Token": f"token-{name}"}
        ),
    )


def test_gym_mcp_tool_name_maps_hermes_names_of_granted_servers() -> None:
    assert _gym_mcp_tool_name("mcp__weather__get_weather", ["weather"]) == "mcp__weather__get_weather"
    # Hermes replaced "-" and "." in the server name; the granted name is restored.
    assert _gym_mcp_tool_name("mcp__my_store_v1__append", ["my-store.v1"]) == "mcp__my-store.v1__append"
    # A granted name that extends another does not lose its tools to the shorter one.
    assert _gym_mcp_tool_name("mcp__store_b__append", ["store", "store_b"]) == "mcp__store_b__append"
    assert _gym_mcp_tool_name("mcp__store__append", ["store", "store_b"]) == "mcp__store__append"
    # Built-in tools and servers that were not granted are left alone.
    assert _gym_mcp_tool_name("terminal", ["weather"]) == "terminal"
    assert _gym_mcp_tool_name("mcp__other__get", ["weather"]) == "mcp__other__get"


class TestSanity:
    def test_construct(self) -> None:
        HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_concurrency_semaphore_initialized(self) -> None:
        agent = HermesAgent(config=_config(concurrency=4), server_client=MagicMock(spec=ServerClient))
        assert agent.sem._value == 4

    def test_model_defaults_to_server_name(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._model_name() == ""

    def test_configured_model_overrides_server_name(self) -> None:
        agent = HermesAgent(config=_config(model="Qwen3.6-35B-A3B"), server_client=MagicMock(spec=ServerClient))
        assert agent._model_name() == "Qwen3.6-35B-A3B"

    def test_context_length_is_written_as_the_model_context_length(self) -> None:
        agent = HermesAgent(
            config=_config(model="policy", context_length=262144), server_client=MagicMock(spec=ServerClient)
        )
        assert yaml.safe_load(agent._build_config())["model"] == {"default": "policy", "context_length": 262144}

    @pytest.mark.parametrize("threshold_tokens", [None, 20000])
    def test_compression_token_threshold_is_written_to_the_compression_config(self, threshold_tokens) -> None:
        agent = HermesAgent(
            config=_config(compression_threshold=0.5, compression_threshold_tokens=threshold_tokens),
            server_client=MagicMock(spec=ServerClient),
        )
        assert yaml.safe_load(agent._build_config())["compression"] == {
            "enabled": True,
            "threshold": 0.5,
            "threshold_tokens": threshold_tokens,
        }

    def test_model_call_stale_timeout_is_written_for_the_custom_provider(self) -> None:
        agent = HermesAgent(
            config=_config(model_call_stale_timeout_seconds=1800), server_client=MagicMock(spec=ServerClient)
        )
        assert yaml.safe_load(agent._build_config())["providers"] == {"custom": {"stale_timeout_seconds": 1800.0}}

    def test_provider_tuning_is_left_to_hermes_without_a_model_call_stale_timeout(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert "providers" not in yaml.safe_load(agent._build_config())

    @pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
    def test_model_call_stale_timeout_must_be_a_positive_finite_number(self, seconds) -> None:
        with pytest.raises(ValidationError):
            _config(model_call_stale_timeout_seconds=seconds)

    def test_model_stays_a_name_without_context_length(self) -> None:
        agent = HermesAgent(config=_config(model="policy"), server_client=MagicMock(spec=ServerClient))
        assert yaml.safe_load(agent._build_config())["model"] == "policy"

    @pytest.mark.parametrize("with_mcp", [False, True])
    async def test_sandbox_access_selects_runtime_provider(self, monkeypatch, with_mcp) -> None:
        hermes = HermesAgent(
            config=_config(
                enabled_toolsets=["terminal", "web"],
                sandbox_config={"ttl_s": 123, "workdir": "/unused"},
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        provider_config = {"opensandbox": {"connection": {}}}
        resolve = MagicMock(return_value=provider_config)
        provider = AsyncMock()
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
        monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
        connect = AsyncMock(return_value=sandbox)
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.get_global_config_dict",
            lambda: {"runtime": {}},
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.resolve_provider_config", resolve)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: provider)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.AsyncSandbox.connect", connect)

        hermes._session_records["session"] = _AgentSessionRecord()
        state = await hermes._initialize_agent_session_state(
            "session",
            AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="test", task_id="task"),
                tool_accesses=[_mcp_access("weather")] if with_mcp else [],
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(
                        provider_config_ref="runtime",
                        descriptor={"sandbox_id": "sandbox"},
                    ),
                    workdir="/app",
                ),
            ),
        )

        resolve.assert_called_once_with("runtime", {"runtime": {}})
        connect.assert_awaited_once_with(
            {"sandbox_id": "sandbox"},
            provider=provider,
        )
        sandbox.start.assert_not_awaited()
        assert hermes.config.sandbox_config == {"ttl_s": 123, "workdir": "/unused"}
        assert state.session.sandbox is sandbox
        assert state.session.owns_sandbox is False
        assert state.session.workdir == "/app"
        assert state.session.session_dir.startswith("/tmp/nemo-gym-hermes-sessions/")
        assert len(state.session.session_dir.rsplit("/", 1)[-1]) == 32
        assert sandbox.exec.await_count == 2
        assert {call.args[0].name for call in sandbox.upload.await_args_list} == {
            "sandbox_runner.py",
            "sandbox_observer.py",
            "model_kwargs.py",
        }

    async def test_seed_installs_hermes_when_it_does_not_import(self, monkeypatch) -> None:
        hermes = HermesAgent(
            config=_config(sandbox_install_timeout_seconds=123), server_client=MagicMock(spec=ServerClient)
        )
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
        is_installed = AsyncMock(return_value=False)
        install = AsyncMock()
        monkeypatch.setattr(SandboxPythonRuntime, "is_installed", is_installed)
        monkeypatch.setattr(SandboxPythonRuntime, "install", install)
        monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.get_global_config_dict",
            lambda: {"runtime": {}},
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.resolve_provider_config", MagicMock())
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: AsyncMock())
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.AsyncSandbox.connect", AsyncMock(return_value=sandbox)
        )

        hermes._session_records["session"] = _AgentSessionRecord()
        await hermes._initialize_agent_session_state(
            "session",
            AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="test", task_id="task"),
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                    workdir="/app",
                ),
            ),
        )

        is_installed.assert_awaited_once_with(sandbox)
        install.assert_awaited_once_with(sandbox, timeout_s=123)
        assert _HERMES_RUNTIME.python_version.startswith("3.14.")
        assert "mcp_tool_discovery" in _HERMES_RUNTIME.import_check
        # Seeding prepares paths from the root; it never runs in, or writes to, the task's working directory.
        assert [call.kwargs["cwd"] for call in sandbox.exec.await_args_list] == ["/"]
        assert {call.args[0].name for call in sandbox.upload.await_args_list} == {
            "sandbox_runner.py",
            "sandbox_observer.py",
            "model_kwargs.py",
        }

    @pytest.mark.parametrize(
        "sandbox_config, install_timeout, runner_timeout, expected_ttl",
        [
            ({}, 900, 21600, 23100),
            ({"workdir": "/fallback"}, 2.5, 3.25, 605.75),
            ({"workdir": "/fallback", "ttl_s": 42}, 900, 21600, 42),
            ({"workdir": "/fallback", "ttl_s": None}, 900, 21600, None),
        ],
    )
    async def test_missing_sandbox_access_uses_configured_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sandbox_config: dict[str, object],
        install_timeout: float,
        runner_timeout: float,
        expected_ttl: float | None,
    ) -> None:
        hermes = HermesAgent(
            config=_config(
                enabled_toolsets=["terminal", "web"],
                sandbox_provider="runtime",
                sandbox_config=sandbox_config,
                sandbox_install_timeout_seconds=install_timeout,
                sandbox_runner_timeout_seconds=runner_timeout,
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        provider = AsyncMock()
        sandbox = AsyncMock()
        sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
        monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
        sandbox_factory = MagicMock(return_value=sandbox)
        original_config = hermes.config.sandbox_config.copy()
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.get_global_config_dict",
            lambda: {"runtime": {}},
        )
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.resolve_provider_config",
            MagicMock(return_value={"local": {}}),
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: provider)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.AsyncSandbox", sandbox_factory)

        hermes._session_records["session"] = _AgentSessionRecord()
        state = await hermes._initialize_agent_session_state(
            "session",
            AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="test", task_id="task"),
            ),
        )

        sandbox_factory.assert_called_once_with(provider)
        sandbox.start.assert_awaited_once_with(SandboxSpec(**{**original_config, "ttl_s": expected_ttl}))
        assert hermes.config.sandbox_config == original_config
        assert state.session.sandbox is sandbox
        assert state.session.workdir == original_config.get("workdir")
        assert state.session.owns_sandbox is True
        await hermes._close_agent_session_state(state)
        sandbox.stop.assert_awaited_once()
        sandbox.disconnect.assert_not_awaited()

    async def test_close_rejects_a_different_episode(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        seeded = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
        )
        hermes._session_records["session"] = _AgentSessionRecord(
            state=MagicMock(request=seeded), episode_id=seeded.episode_id
        )
        hermes._close_agent_session_state = AsyncMock()
        request = SimpleNamespace(session={"agent_session_id": "session"})

        with pytest.raises(HTTPException, match="episode_id does not match"):
            await hermes.close_agent_session(
                request,
                AgentCloseSessionRequest(
                    agent_session_id="session",
                    episode_id=EpisodeId(rollout_id="other"),
                ),
            )

        hermes._close_agent_session_state.assert_not_awaited()

    async def test_session_seed_and_close_are_idempotent(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
        )
        state = MagicMock(request=body)
        hermes._initialize_agent_session_state = AsyncMock(return_value=state)
        hermes._close_agent_session_state = AsyncMock(
            return_value=AgentCloseSessionResponse(agent_session_id="session")
        )
        request = SimpleNamespace(session={})

        first = await hermes.seed_agent_session(request, body)
        second = await hermes.seed_agent_session(request, body)
        assert first == second
        hermes._initialize_agent_session_state.assert_awaited_once()

        close_body = AgentCloseSessionRequest(
            agent_session_id="session",
            episode_id=body.episode_id,
        )
        await hermes.close_agent_session(request, close_body)
        await hermes.close_agent_session(request, close_body)
        hermes._close_agent_session_state.assert_awaited_once_with(state)

    async def test_several_workers_seed_sessions(self) -> None:
        """Session routing sends a session's later requests to the worker that seeded it."""
        hermes = HermesAgent(config=_config(num_workers=2), server_client=MagicMock(spec=ServerClient))
        hermes._initialize_agent_session_state = AsyncMock()
        request = SimpleNamespace(session={})
        body = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
        )
        await hermes.seed_agent_session(request, body)

        hermes._initialize_agent_session_state.assert_awaited_once_with("session", body)
        assert request.session[AGENT_SESSION_COOKIE_KEY] == "session"

    async def test_seed_rejects_required_grants_other_than_mcp(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
            tool_accesses=[
                DirectHTTPToolAccess(
                    name="required-tools",
                    required=True,
                    base_url="http://resources:8000",
                )
            ],
        )
        hermes._initialize_agent_session_state = AsyncMock()

        with pytest.raises(HTTPException, match="supports only MCP tool grants.*required-tools") as error:
            await hermes.seed_agent_session(SimpleNamespace(session={}), body)

        assert error.value.status_code == 422
        hermes._initialize_agent_session_state.assert_not_awaited()

    async def test_seed_accepts_required_mcp_grants_and_rejects_toolset_names(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        hermes._initialize_agent_session_state = AsyncMock()

        def seed(name: str) -> AgentSeedSessionRequest:
            return AgentSeedSessionRequest(
                agent_session_id=f"session-{name}",
                episode_id=EpisodeId(rollout_id=f"rollout-{name}"),
                task_id=TaskId(taskset="test", task_id="task"),
                tool_accesses=[_mcp_access(name)],
            )

        await hermes.seed_agent_session(SimpleNamespace(session={}), seed("weather"))
        hermes._initialize_agent_session_state.assert_awaited_once()

        # Hermes exposes an MCP server as a toolset of the same name, so it cannot shadow a built-in toolset.
        with pytest.raises(ValueError, match="collide with Hermes toolsets: terminal"):
            await hermes.seed_agent_session(SimpleNamespace(session={}), seed("terminal"))

    @pytest.mark.parametrize(
        ("unsupported", "reason"),
        [
            ({"tool_accesses": [_mcp_access("terminal")]}, "MCP tool grants collide with Hermes toolsets: terminal"),
            # No sandbox_access, and this config names no sandbox_provider either.
            ({}, "Hermes requires sandbox_access or a configured sandbox_provider for an episode session"),
        ],
    )
    def test_unsupported_seed_is_refused_with_a_terminal_422(self, unsupported: dict, reason: str) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        seed = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
            **unsupported,
        )

        assert check_agent_seed_rejected(hermes.setup_webserver(), seed) == reason
        assert not hermes._session_records

    @staticmethod
    def _sandbox_session(
        monkeypatch, sandbox, *, config=None, tool_accesses=()
    ) -> tuple[HermesAgent, SimpleNamespace, AgentSeedSessionRequest]:
        import nemo_gym.agents.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://model-server:1"
        hermes = HermesAgent(config=config or _config(), server_client=server_client)
        seed = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
            tool_accesses=list(tool_accesses),
        )
        hermes._session_records["session"] = _AgentSessionRecord(
            state=HermesSandboxSession(
                request=seed,
                session=SandboxSession(sandbox=sandbox, workdir=None, session_dir="/session", harness="Hermes"),
            ),
            episode_id=seed.episode_id,
        )
        request = SimpleNamespace(
            session={"agent_session_id": "session"},
            path_params={"rollout_id": seed.episode_id.capture_key},
        )
        original_download = sandbox.download

        async def download(remote_path, local_path):
            if remote_path.endswith("/cleanup.json"):
                commands = getattr(sandbox, "commands", None)
                Path(local_path).write_text(
                    json.dumps(
                        {
                            "cleanup_confirmed": commands is None
                            or any("touch" in cmd and "stop.request" in cmd for cmd in commands),
                            "error": None,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                await original_download(remote_path, local_path)

        sandbox.download = AsyncMock(side_effect=download)
        return hermes, request, seed

    async def test_sandbox_activation_calls_the_model_server_directly(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.uploaded: dict = {}

            async def upload(self, local_path, remote_path) -> None:
                self.uploaded[remote_path] = Path(local_path).read_text()

            # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                return SandboxExecResult(stdout="", stderr="", return_code=0)

            async def download(self, remote_path, local_path) -> None:
                result = {"messages": [{"role": "assistant", "content": "done"}], "final_response": "done"}
                output = {"result": result, "runtime": {"hostname": "sandbox", "pid": 1, "python": "python"}}
                Path(local_path).write_text(json.dumps(output))

        sandbox = _Sandbox()
        hermes, request, seed = self._sandbox_session(monkeypatch, sandbox)

        response = await hermes.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))

        runner_input = json.loads(sandbox.uploaded["/session/input.json"])
        assert "/session/process_supervisor.sh" in sandbox.uploaded
        assert runner_input["model_base_url"] == f"http://model-server:1/ng-rollout/{seed.episode_id.capture_key}/v1"
        assert runner_input["user_message"] == "fix bug"
        assert response.metadata["harness_execution"] == "sandbox"

    async def test_activation_configures_granted_mcp_servers(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.uploaded: dict = {}

            async def upload(self, local_path, remote_path) -> None:
                self.uploaded[remote_path] = Path(local_path).read_text()

            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                return SandboxExecResult(stdout="", stderr="", return_code=0)

            async def download(self, remote_path, local_path) -> None:
                tool_call = {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "mcp__weather_v1_0__get_weather", "arguments": '{"city": "Paris"}'},
                }
                messages = [
                    {"role": "user", "content": "what is the weather"},
                    {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
                    {"role": "assistant", "content": "done"},
                ]
                result = {"messages": messages, "final_response": "done"}
                output = {"result": result, "runtime": {"hostname": "sandbox", "pid": 1, "python": "python"}}
                Path(local_path).write_text(json.dumps(output))

        sandbox = _Sandbox()
        hermes, request, _seed = self._sandbox_session(
            monkeypatch,
            sandbox,
            config=_config(enabled_toolsets=["terminal"]),
            tool_accesses=[_mcp_access("weather-v1.0"), _mcp_access("search", required=False)],
        )

        response = await hermes.responses(
            request, NeMoGymResponseCreateParamsNonStreaming(input="what is the weather")
        )

        runner_input = json.loads(sandbox.uploaded["/session/input.json"])
        assert runner_input["mcp_servers"] == ["weather-v1.0", "search"]
        assert runner_input["required_mcp_servers"] == ["weather-v1.0"]
        assert runner_input["enabled_toolsets"] == ["terminal", "weather-v1.0", "search"]
        mcp_servers = yaml.safe_load(runner_input["config_yaml"])["mcp_servers"]
        assert mcp_servers["weather-v1.0"] == {
            "url": "http://resources:8000/mcp",
            "headers": {"X-NeMo-Gym-Session-Token": "token-weather-v1.0"},
            "tools": {"resources": False, "prompts": False},
        }
        # Verifiers see Gym's MCP naming, not Hermes'.
        assert [item.name for item in response.output if item.type == "function_call"] == [
            "mcp__weather-v1.0__get_weather"
        ]

    async def test_provider_timeout_without_output_confirms_cleanup_and_fails(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def upload(self, *_args) -> None:
                pass

            async def download(self, *_args) -> None:
                raise FileNotFoundError("worker did not checkpoint")

            # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                self.commands.append(command)
                if "supervisor.pid" in command and "exec " in command:
                    return SandboxExecResult(stdout=None, stderr="timed out", return_code=124, error_type="timeout")
                return SandboxExecResult(stdout="", stderr="", return_code=0)

        sandbox = _Sandbox()
        hermes, request, _ = self._sandbox_session(monkeypatch, sandbox)

        with pytest.raises(TimeoutError, match="exceeded its execution deadline"):
            await hermes.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))

        stop = [command for command in sandbox.commands if "touch" in command and "stop.request" in command]
        assert len(stop) == 1

    @staticmethod
    def _sandbox_with_saved_transcript(result: dict):
        class _Sandbox:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def upload(self, *_args) -> None:
                pass

            async def download(self, remote_path, local_path) -> None:
                output = {"result": result, "runtime": {"hostname": "sandbox", "pid": 1, "python": "python"}}
                Path(local_path).write_text(json.dumps(output))

            # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                self.commands.append(command)
                if "supervisor.pid" in command and "exec " in command:
                    return SandboxExecResult(stdout=None, stderr="timed out", return_code=124, error_type="timeout")
                return SandboxExecResult(stdout="", stderr="", return_code=0)

        return _Sandbox()

    async def test_early_provider_timeout_with_a_transcript_stays_a_failure(self, monkeypatch) -> None:
        # A transport TimeoutError long before the agent deadline is an infrastructure failure, even
        # though a transcript was saved: it must not become a gradable wall_time outcome.
        sandbox = self._sandbox_with_saved_transcript(
            {"messages": [{"role": "assistant", "content": "partial"}], "final_response": None, "completed": False}
        )
        hermes, request, _ = self._sandbox_session(monkeypatch, sandbox)
        clock = iter([0.0, 5.0])
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: next(clock))

        with pytest.raises(TimeoutError, match="exceeded its execution deadline"):
            await hermes.responses(
                request,
                NeMoGymResponseCreateParamsNonStreaming(input="fix bug", metadata={"agent_timeout_sec": "600"}),
            )

    async def test_provider_timeout_at_the_agent_deadline_returns_the_partial_transcript(self, monkeypatch) -> None:
        sandbox = self._sandbox_with_saved_transcript(
            {"messages": [{"role": "assistant", "content": "partial"}], "final_response": None, "completed": False}
        )
        hermes, request, _ = self._sandbox_session(monkeypatch, sandbox)
        clock = iter([0.0, 600.5])
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: next(clock))

        response = await hermes.responses(
            request,
            NeMoGymResponseCreateParamsNonStreaming(input="fix bug", metadata={"agent_timeout_sec": "600"}),
        )

        assert response.metadata["stop_reason"] == "wall_time"
        assert response.metadata["hermes_error_type"] == "TimeoutError"

    async def test_runner_recorded_deadline_counts_as_deadline_evidence(self, monkeypatch) -> None:
        # The supervisor stopped Hermes at the deadline and the runner wrote a wall_time transcript; the
        # provider's own timeout fired a little later. No wall clock needed.
        sandbox = self._sandbox_with_saved_transcript(
            {
                "messages": [],
                "final_response": None,
                "completed": False,
                "interrupted": True,
                "stop_reason": "wall_time",
            }
        )
        hermes, request, _ = self._sandbox_session(monkeypatch, sandbox)
        clock = iter([0.0, 1.0])
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: next(clock))

        response = await hermes.responses(
            request,
            NeMoGymResponseCreateParamsNonStreaming(input="fix bug", metadata={"agent_timeout_sec": "600"}),
        )

        assert response.metadata["stop_reason"] == "wall_time"

    async def test_cleanup_cancellation_after_an_early_provider_timeout_stays_a_failure(self, monkeypatch) -> None:
        # An early provider TimeoutError makes the session clean up: close publishes stop.request, the
        # supervisor signals the runner, and the runner records an interrupted transcript with
        # stop_reason "cancelled". That interruption is a consequence of the failure, not deadline
        # evidence, so the failure must stand rather than become a scoreable wall_time outcome.
        sandbox = self._sandbox_with_saved_transcript(
            {
                "messages": [],
                "final_response": None,
                "completed": False,
                "interrupted": True,
                "stop_reason": "cancelled",
            }
        )
        hermes, request, _ = self._sandbox_session(monkeypatch, sandbox)
        clock = iter([0.0, 1.0])
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: next(clock))

        with pytest.raises(TimeoutError, match="exceeded its execution deadline"):
            await hermes.responses(
                request,
                NeMoGymResponseCreateParamsNonStreaming(input="fix bug", metadata={"agent_timeout_sec": "900"}),
            )

    async def test_close_during_activation_stops_the_runner(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.runner_started = asyncio.Event()

            async def upload(self, *_args) -> None:
                pass

            async def disconnect(self, **_) -> None:
                pass

            async def download(self, remote_path, local_path) -> None:
                raise FileNotFoundError("worker was interrupted")

            # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                self.commands.append(command)
                if "supervisor.pid" in command and "exec " in command:
                    self.runner_started.set()
                    await asyncio.Event().wait()
                return SandboxExecResult(stdout="", stderr="", return_code=0)

        sandbox = _Sandbox()
        hermes, request, seed = self._sandbox_session(monkeypatch, sandbox)

        activation = asyncio.create_task(
            hermes.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))
        )
        await asyncio.wait_for(sandbox.runner_started.wait(), timeout=5)
        await asyncio.wait_for(
            hermes.close_agent_session(
                request, AgentCloseSessionRequest(agent_session_id="session", episode_id=seed.episode_id)
            ),
            timeout=5,
        )

        stop = [command for command in sandbox.commands if "touch" in command and "stop.request" in command]
        assert len(stop) == 1
        assert hermes._session_records["session"].state is None
        with pytest.raises(asyncio.CancelledError):
            await activation

    async def test_close_before_seed_prevents_late_creation(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
        )
        request = SimpleNamespace(session={})

        await hermes.close_agent_session(
            request,
            AgentCloseSessionRequest(
                agent_session_id="session",
                episode_id=body.episode_id,
            ),
        )

        with pytest.raises(HTTPException, match="already closed"):
            await hermes.seed_agent_session(request, body)


class _FakeAgent:
    """Stand-in for AIAgent — only needs .interrupt() for the SIGTERM dispatch path."""

    def __init__(self) -> None:
        self.interrupt_reason = None

    returncode = None

    def terminate(self) -> None:
        self.interrupt_reason = "timeout"


class TestSigtermHandler:
    """Regression tests for the concurrency-safe SIGTERM dispatcher.

    The old per-call add_signal_handler/remove_signal_handler approach raced: concurrent responses()
    calls clobbered each other's handler and the first to finish removed the only one left, so a
    later SIGTERM interrupted nobody. The fix registers a single dispatcher over a shared set of
    in-flight agents.
    """

    def test_active_agents_initialized_empty(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent.active_agents == set()
        assert agent.sigterm_installed is False

    def test_handler_installed_once_and_interrupts_all_in_flight(self) -> None:
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

        registered: list = []
        loop = asyncio.new_event_loop()
        loop.add_signal_handler = lambda sig, cb, *a: registered.append(cb)  # type: ignore[method-assign]
        asyncio.set_event_loop(loop)
        try:
            agent._ensure_sigterm_handler()
            assert agent.sigterm_installed is True
            assert len(registered) == 1  # exactly one dispatcher registered

            # Idempotent: a second concurrent call must NOT register another handler.
            agent._ensure_sigterm_handler()
            assert len(registered) == 1

            dispatch = registered[0]

            # Two concurrent in-flight agents: SIGTERM must interrupt BOTH (the old code lost one).
            a, b = _FakeAgent(), _FakeAgent()
            agent.active_agents.update({a, b})
            dispatch()
            assert a.interrupt_reason == "timeout"
            assert b.interrupt_reason == "timeout"

            # Once an agent finishes (discarded), a later SIGTERM no longer touches it.
            a.interrupt_reason = None
            b.interrupt_reason = None
            agent.active_agents.discard(a)
            dispatch()
            assert a.interrupt_reason is None
            assert b.interrupt_reason == "timeout"
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_handler_install_survives_unsupported_platform(self) -> None:
        # On platforms where add_signal_handler raises (e.g. non-main thread), install is a no-op
        # rather than an error, and the agent stays usable.
        agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))

        loop = asyncio.new_event_loop()

        def _raise(*_a, **_k):
            raise NotImplementedError

        loop.add_signal_handler = _raise  # type: ignore[method-assign]
        asyncio.set_event_loop(loop)
        try:
            agent._ensure_sigterm_handler()
            assert agent.sigterm_installed is False
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_interrupted_agent_id_cleared_when_run_conversation_raises(self, monkeypatch) -> None:
        import nemo_gym.agents.base_responses_api_agent as base_agent
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        hermes = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(hermes, "_ensure_sigterm_handler", lambda: None)

        process = MagicMock(returncode=1)

        async def communicate(_):
            hermes.interrupted_agents.add(id(process))
            raise RuntimeError("agent failed")

        process.communicate = communicate
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.host_python", lambda: "python")

        with pytest.raises(RuntimeError, match="agent failed"):
            asyncio.run(hermes.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))

        assert hermes.active_agents == set()
        assert hermes.interrupted_agents == set()


class TestResultUsage:
    @pytest.mark.parametrize("partial", [False, True])
    def test_reports_cache_inclusive_native_totals(self, partial) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        response = hermes._response_from_result(
            body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            result={
                "messages": [{"role": "assistant", "content": "done"}],
                "partial": partial,
                "input_tokens": 100,
                "cache_read_tokens": 200,
                "cache_write_tokens": 300,
                "prompt_tokens": 600,
                "completion_tokens": 80,
                "reasoning_tokens": 30,
            },
            model_name="model",
        )
        assert response.usage.input_tokens == 600
        assert response.usage.input_tokens_details.cached_tokens == 200
        assert response.usage.output_tokens == 80
        assert response.usage.output_tokens_details.reasoning_tokens == 30
        assert response.usage.total_tokens == 680

    @pytest.mark.parametrize(
        "usage",
        [{}, {"prompt_tokens": 10}, {"completion_tokens": 20}, {"prompt_tokens": None, "completion_tokens": 20}],
    )
    def test_missing_native_totals_remain_unknown(self, usage) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        response = hermes._response_from_result(
            body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            result={"messages": [{"role": "assistant", "content": "partial answer"}], **usage},
            model_name="model",
        )
        assert response.usage is None

    def test_missing_details_remain_unknown(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        response = hermes._response_from_result(
            body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            result={
                "messages": [{"role": "assistant", "content": "done"}],
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
            model_name="model",
        )
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.usage.input_tokens_details.cached_tokens is None
        assert response.usage.output_tokens_details.reasoning_tokens is None


class TestResultClassification:
    @pytest.mark.parametrize(
        "error",
        [
            "HTTP 429 Too Many Requests",
            "HTTP 500 Internal Server Error",
            "HTTP 403 Forbidden",
            "Connection refused",
            "Invalid API response shape. Likely rate limited or malformed provider response.",
            None,
        ],
    )
    @pytest.mark.parametrize("has_partial_patch", [False, True])
    def test_provider_failure_is_not_a_gradable_model_outcome(self, error, has_partial_patch) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        messages = [{"role": "assistant", "content": "Applied a partial patch"}] if has_partial_patch else []
        with pytest.raises(RuntimeError, match="Hermes agent failed"):
            hermes._response_from_result(
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                result={"failed": True, "error": error, "messages": messages},
                model_name="model",
            )

    @pytest.mark.parametrize(
        "error", ["HTTP 400: controlled model failure", "HTTP 413: payload too large", "HTTP 422: bad params"]
    )
    def test_terminal_model_rejection_is_a_gradable_outcome(self, error) -> None:
        # The model endpoint refused the request for what it asked, after Hermes's retries. The conformance
        # suite's model_error scenario expects this to reach the verifier as an unmasked zero.
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        response = hermes._response_from_result(
            body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            result={"failed": True, "error": error, "messages": []},
            model_name="model",
        )
        assert response.status == "failed"
        assert response.error is not None and response.error.message == error
        assert response.metadata["model_rejected"] == error.split()[1].rstrip(":")
        assert "provider_failed" not in response.metadata

    @pytest.mark.parametrize(
        "outcome",
        [
            {"failed": True, "error": "First response truncated due to output length limit"},
            {"partial": True, "error": "Response truncated due to output length limit"},
            {"partial": True, "error": "Context length exceeded (100 tokens). Cannot compress further."},
            {
                "failed": True,
                "failure_reason": "context_overflow",
                "compression_exhausted": True,
                "error": "This conversation has grown too long for model to read, and Hermes couldn't shrink it",
            },
            {
                "failed": True,
                "failure_reason": "truncated",
                "error": "The model's reply was cut off before it finished (it hit its output length limit)",
            },
            {"partial": True, "error": "Model generated invalid tool call: invalid"},
            {"completed": True},
            {"completed": False, "stop_reason": "max_iterations"},
        ],
    )
    def test_model_outcomes_remain_gradable_with_partial_trajectory(self, outcome) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        response = hermes._response_from_result(
            body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            result=outcome | {"messages": [{"role": "assistant", "content": "Applied a partial patch"}]},
            model_name="model",
        )
        assert response.output[0].content[0].text == "Applied a partial patch"
        assert response.metadata["partial"] == str(bool(outcome.get("partial"))).lower()
        if outcome.get("stop_reason") == "max_iterations":
            assert response.status == "incomplete"
            assert response.error is None
            assert response.metadata["stop_reason"] == "max_iterations"

    @pytest.mark.parametrize("totals", [None, (0, 0), (20, 10)])
    async def test_host_path_retains_provider_failure_for_masked_verification(self, monkeypatch, totals) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model:8000/v1")
        monkeypatch.setattr(HermesAgent, "_ensure_sigterm_handler", lambda *_: None)
        result = {"failed": True, "error": "HTTP 500", "messages": []}
        if totals is not None:
            result.update(prompt_tokens=totals[0], completion_tokens=totals[1])
        monkeypatch.setattr(HermesAgent, "_run_host_worker", AsyncMock(return_value=({"result": result}, False)))
        response = await hermes._create_response(NeMoGymResponseCreateParamsNonStreaming(input="hi"))
        assert response.status == "failed"
        assert response.error.message == "HTTP 500"
        assert response.metadata["provider_failed"] == "true"
        if totals is None:
            assert response.usage is None
        else:
            assert response.usage.input_tokens == totals[0]
            assert response.usage.output_tokens == totals[1]
            assert response.usage.total_tokens == sum(totals)


def test_host_and_sandbox_share_an_immutable_runtime_source():
    from responses_api_agents.hermes_agent.runtime import HERMES_COMMIT, HERMES_PYTHON_VERSION, HERMES_SOURCE

    assert _HERMES_RUNTIME.python_version == HERMES_PYTHON_VERSION
    assert re.fullmatch("[0-9a-f]{40}", HERMES_COMMIT)
    assert _HERMES_RUNTIME.requirements == (HERMES_SOURCE,)
    assert HERMES_SOURCE.endswith("@" + HERMES_COMMIT)


class TestSplitInputToUserAndHistory:
    def test_string_input_is_the_user_message(self) -> None:
        assert _split_input_to_user_and_history("fix bug") == ("fix bug", [], None)

    def test_user_only(self) -> None:
        items = [NeMoGymEasyInputMessage(role="user", content="hi")]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "hi"
        assert history == []
        assert system is None

    def test_system_plus_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be helpful"),
            NeMoGymEasyInputMessage(role="user", content="hi"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "hi"
        assert history == []
        assert system == "be helpful"

    def test_history_then_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="user", content="first"),
            NeMoGymEasyInputMessage(role="assistant", content="reply"),
            NeMoGymEasyInputMessage(role="user", content="follow-up"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "follow-up"
        assert history == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        assert system is None

    def test_resumed_ends_on_assistant(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="user", content="q"),
            NeMoGymEasyInputMessage(role="assistant", content="a"),
        ]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == ""
        assert history == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_dict_inputs(self) -> None:
        items = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "ok"}]
        user, history, system = _split_input_to_user_and_history(items)
        assert user == "ok"
        assert history == []
        assert system == "be brief"


class TestTrajectoryToOutputItems:
    def test_empty(self) -> None:
        assert _trajectory_to_output_items([], 0) == []

    def test_drops_input_prefix(self) -> None:
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        out = _trajectory_to_output_items(msgs, 1)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)

    def test_assistant_with_tokens(self) -> None:
        routed_experts = [
            [[0, 1]],
            [[2, 3]],
            [[4, 5]],
            [[6, 7]],
        ]
        msgs = [
            {
                "role": "assistant",
                "content": "answer",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [0.0, -0.1],
                "routed_experts": routed_experts,
            }
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert out[0].generation_token_ids == [3, 4]
        assert out[0].prompt_token_ids == [1, 2]
        assert out[0].routed_experts == routed_experts

    def test_assistant_reasoning_strips_inline_think_from_message(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "reasoning": "structured thoughts",
                "content": "<think>structured thoughts</think>final answer",
            }
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 2
        assert isinstance(out[0], NeMoGymResponseReasoningItem)
        assert out[0].summary[0].text == "structured thoughts"
        assert isinstance(out[1], NeMoGymResponseOutputMessageForTraining)
        assert out[1].content[0].text == "final answer"

    def test_assistant_with_tool_call_and_tool_result(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file.txt\n"},
        ]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 3
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert isinstance(out[1], NeMoGymResponseFunctionToolCall)
        assert out[1].name == "terminal"
        assert out[1].arguments == '{"cmd":"ls"}'
        assert isinstance(out[2], NeMoGymFunctionCallOutput)
        assert out[2].call_id == "c1"
        assert out[2].output == "file.txt\n"

    def test_skips_non_dict_items(self) -> None:
        msgs = [None, "string", {"role": "assistant", "content": "ok"}]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1


class TestRolloutCorrelation:
    def test_responses_applies_rollout_prefix(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        import nemo_gym.agents.base_responses_api_agent as base_agent
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        seen: dict = {}

        async def worker(_, payload):
            seen["base_url"] = payload["model_base_url"]
            return ({"result": {"messages": [{"role": "assistant", "content": "ok"}]}}, False)

        monkeypatch.setattr(HermesAgent, "_run_host_worker", worker)
        client = TestClient(agent.setup_webserver())

        assert client.post("/ng-rollout/rid/v1/responses", json={"input": "hi"}).status_code == 200
        assert seen["base_url"] == "http://h:1/ng-rollout/rid/v1"

        direct = asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["base_url"] == "http://h:1/v1"
        assert "_ng_agent_observations" not in direct.model_dump(mode="json")

        episode = asyncio.run(
            agent._create_episode(
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                rollout_id="rid",
            )
        )
        assert seen["base_url"] == "http://h:1/ng-rollout/rid/v1"
        assert episode.observations.source == "hermes"
        assert episode.observations.records[0].invocation_id == "root"


class TestMaxTokens:
    def _agent_and_seen(self, monkeypatch, **config_kwargs) -> tuple[HermesAgent, dict]:
        import nemo_gym.agents.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(**config_kwargs), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        seen: dict = {}

        async def worker(_, payload):
            seen["max_tokens"] = payload["max_tokens"]
            return ({"result": {"messages": [{"role": "assistant", "content": "ok"}]}}, False)

        monkeypatch.setattr(HermesAgent, "_run_host_worker", worker)
        return agent, seen

    def test_max_tokens_passed_to_ai_agent(self, monkeypatch) -> None:
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch, max_tokens=4096)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] == 4096

    def test_max_tokens_defaults_to_none(self, monkeypatch) -> None:
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] is None


class TestObservability:
    @pytest.mark.parametrize(
        ("terminal_backend", "runtime_gap"),
        [("local", "no_sandbox_runtime"), ("docker", "sandbox_observation_unavailable")],
    )
    def test_observation_failure_does_not_change_response(self, monkeypatch, terminal_backend, runtime_gap) -> None:
        import nemo_gym.agents.base_responses_api_agent as base_agent
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(terminal_backend=terminal_backend), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        result = {
            "completed": True,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
        }
        monkeypatch.setattr(HermesAgent, "_run_host_worker", AsyncMock(return_value=({"result": result}, False)))
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        baseline = asyncio.run(agent.responses(request=None, body=body))

        def fail_finish(*args, **kwargs):
            raise RuntimeError("observer failed")

        monkeypatch.setattr(HermesAgent, "_sandbox_observations", fail_finish)
        episode = asyncio.run(agent._create_episode(body=body, rollout_id="rid"))

        assert episode.response.output == baseline.output
        assert episode.response.usage == baseline.usage
        assert [gap.code for gap in episode.observations.gaps] == [
            "observation_capture_failed",
            runtime_gap,
        ]

    @pytest.mark.parametrize("provider_failed", [False, True, "model_rejected"])
    def test_run_returns_observations_without_leaking_internal_attachment(self, provider_failed) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": True}
        agent = HermesAgent(config=_config(), server_client=server_client)
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp-1",
                "created_at": 1,
                "model": "model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        if provider_failed == "model_rejected":
            # A terminal HTTP 400 from the model: graded, not masked (conformance model_error scenario).
            response = response.model_copy(update={"metadata": {"model_rejected": "400"}, "status": "failed"})
            provider_failed = False
        elif provider_failed:
            response = response.model_copy(
                update={
                    "metadata": {"provider_failed": "true"},
                    "status": "failed",
                }
            )
        observed_response = AsyncMock(
            return_value=AgentEpisode(
                response=response,
                observations=AgentObservationBundle(source="hermes"),
            )
        )

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/seed_session":
                return _FakeResponse({}, {"session": "1"})
            if url_path.endswith("/v1/responses"):
                response = await agent.responses(MagicMock(path_params={"rollout_id": "1-2"}), json)
                return _FakeResponse(response.model_dump(mode="json"), cookies)
            return _FakeResponse(json | {"reward": 1.0})

        server_client.post = AsyncMock(side_effect=post)
        request = MagicMock()
        request.cookies = {}
        body = HermesAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
            }
        )

        with patch.object(HermesAgent, "_create_episode", observed_response):
            result = asyncio.run(agent.run(request, body))

        assert result.reward == 1.0  # Retain the verifier's actual result even when masked.
        assert result.mask_sample is provider_failed
        if provider_failed:
            assert result.failure_kind == "agent_request_failed"
            assert result.failure_reason
            assert result.finished_naturally is False
        assert result.ng_agent_observations is not None
        assert result.ng_agent_observations.source == "hermes"
        verify_json = server_client.post.await_args_list[-1].kwargs["json"]
        assert "_ng_agent_observations" not in verify_json["response"]
        assert "rollout_id" not in verify_json

    def test_observer_failure_does_not_mask_agent_exception(self, monkeypatch) -> None:
        import nemo_gym.agents.base_responses_api_agent as base_agent
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        monkeypatch.setattr(HermesAgent, "_run_host_worker", AsyncMock(side_effect=ValueError("agent failed")))
        monkeypatch.setattr(
            HermesAgent, "_sandbox_observations", MagicMock(side_effect=RuntimeError("observer failed"))
        )

        with pytest.raises(ValueError, match="agent failed"):
            asyncio.run(
                agent._create_episode(
                    body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                    rollout_id="rid",
                )
            )


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_native_model_error_is_reported_with_classification(status):
    import json

    agent = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    native = {"failed": True, "error": f"HTTP {status} provider error", "messages": []}
    response = agent._response_from_result(
        body=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        result=native,
        model_name="model",
        retain_provider_failure=True,
    )
    response = agent.report_outcome(response, native)
    assert response.status == "failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": status,
    }


def _host_worker_agent(monkeypatch, tmp_path, script: str) -> HermesAgent:
    """A Hermes agent whose host worker interpreter is the given shell script."""
    worker = tmp_path / "worker"
    worker.write_text("#!/bin/sh\n" + script)
    worker.chmod(0o755)
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.host_python", lambda: worker)
    hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(hermes, "_ensure_sigterm_handler", lambda: None)
    return hermes


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("cat >/dev/null\nexit 3\n", "exited with status 3 without a response"),
        (
            'cat >/dev/null\necho \'{"error_type": "ImportError", "error": "no module"}\'\nexit 1\n',
            r"Hermes worker failed \(ImportError\): no module",
        ),
    ],
)
def test_host_worker_failure_raises_with_its_reason(monkeypatch, tmp_path, script, message) -> None:
    hermes = _host_worker_agent(monkeypatch, tmp_path, script)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(hermes._run_host_worker({"history": []}))
    assert hermes.active_agents == set()


def test_host_worker_cancel_kills_its_process_group(monkeypatch, tmp_path) -> None:
    import os
    import signal

    child_pid = tmp_path / "child.pid"
    # The worker starts a tool-like grandchild in its own process group, then waits.
    hermes = _host_worker_agent(monkeypatch, tmp_path, f"sleep 300 &\necho $! > {child_pid}\nwait\n")

    async def cancel_running_worker() -> None:
        task = asyncio.create_task(hermes._run_host_worker({"history": []}))
        for _ in range(500):
            if child_pid.exists() and child_pid.read_text().strip():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        # A surviving grandchild keeps the worker's stdout open, so cleanup would wait for it.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    try:
        asyncio.run(cancel_running_worker())
    finally:
        pid = int(child_pid.read_text())
        try:
            os.kill(pid, signal.SIGKILL)
            survived = True
        except ProcessLookupError:
            survived = False
    assert not survived, "the worker's process group outlived the cancelled activation"
    assert hermes.active_agents == set()
