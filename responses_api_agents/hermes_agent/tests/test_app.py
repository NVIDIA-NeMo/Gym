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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_responses_api_agent import (
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    _AgentSessionRecord,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import ServerClient
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
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
from responses_api_agents.hermes_agent.observability import HermesAgentObserver
from responses_api_agents.hermes_agent.sandbox import HermesSandboxSession, _sandbox_hermes_install


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
    assert _gym_mcp_tool_name("mcp_weather_get_weather", ["weather"]) == "mcp__weather__get_weather"
    # Hermes replaced "-" and "." in the server name; the granted name is restored.
    assert _gym_mcp_tool_name("mcp_my_store_v1_append", ["my-store.v1"]) == "mcp__my-store.v1__append"
    # A granted name that extends another does not lose its tools to the shorter one.
    assert _gym_mcp_tool_name("mcp_store_b_append", ["store", "store_b"]) == "mcp__store_b__append"
    assert _gym_mcp_tool_name("mcp_store_append", ["store", "store_b"]) == "mcp__store__append"
    # Built-in tools and servers that were not granted are left alone.
    assert _gym_mcp_tool_name("terminal", ["weather"]) == "terminal"
    assert _gym_mcp_tool_name("mcp_other_get", ["weather"]) == "mcp_other_get"


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
        sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
        connect = AsyncMock(return_value=sandbox)
        monkeypatch.setattr("responses_api_agents.hermes_agent.sandbox.shutil.which", lambda name: "/test/uv")
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
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        sandbox = AsyncMock()
        import_results = iter([1, 0])

        async def exec_(command, **_kwargs):
            if "import run_agent" in command:
                return MagicMock(return_code=next(import_results), stdout="", stderr="")
            return MagicMock(return_code=0, stdout="", stderr="")

        sandbox.exec.side_effect = exec_
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.get_global_config_dict",
            lambda: {"runtime": {}},
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.resolve_provider_config", MagicMock())
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: AsyncMock())
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.AsyncSandbox.connect", AsyncMock(return_value=sandbox)
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.sandbox.shutil.which", lambda _name: "/usr/bin/uv")

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

        commands = [call.args[0] for call in sandbox.exec.await_args_list]
        install = [command for command in commands if "pip install" in command]
        assert len(install) == 1 and "rm -rf" in install[0]
        # The import is checked before installing and again after.
        assert sum("import run_agent" in command for command in commands) == 2
        assert sandbox.upload.await_args_list[0].args[0] == "/usr/bin/uv"

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
        sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
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

    @staticmethod
    def _sandbox_session(
        monkeypatch, sandbox, *, config=None, tool_accesses=()
    ) -> tuple[HermesAgent, SimpleNamespace, AgentSeedSessionRequest]:
        import nemo_gym.base_responses_api_agent as base_agent

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
                            "cleanup_confirmed": commands is None or any("kill -TERM" in cmd for cmd in commands),
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
        assert "/session/process_supervisor.py" in sandbox.uploaded
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
                    "function": {"name": "mcp_weather_get_weather", "arguments": '{"city": "Paris"}'},
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
            tool_accesses=[_mcp_access("weather"), _mcp_access("search", required=False)],
        )

        response = await hermes.responses(
            request, NeMoGymResponseCreateParamsNonStreaming(input="what is the weather")
        )

        runner_input = json.loads(sandbox.uploaded["/session/input.json"])
        assert runner_input["mcp_servers"] == ["weather", "search"]
        assert runner_input["required_mcp_servers"] == ["weather"]
        assert runner_input["enabled_toolsets"] == ["terminal", "weather", "search"]
        mcp_servers = yaml.safe_load(runner_input["config_yaml"])["mcp_servers"]
        assert mcp_servers["weather"] == {
            "url": "http://resources:8000/mcp",
            "headers": {"X-NeMo-Gym-Session-Token": "token-weather"},
            "tools": {"resources": False, "prompts": False},
        }
        # Verifiers see Gym's MCP naming, not Hermes'.
        assert [item.name for item in response.output if item.type == "function_call"] == ["mcp__weather__get_weather"]

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

        stop = [command for command in sandbox.commands if "stop.request" in command and "kill -TERM" in command]
        assert len(stop) == 1

    async def test_close_during_activation_stops_the_runner(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.runner_started = asyncio.Event()

            async def upload(self, *_args) -> None:
                pass

            async def disconnect(self) -> None:
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

        stop = [command for command in sandbox.commands if "stop.request" in command and "kill -TERM" in command]
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

    def interrupt(self, reason: str) -> None:
        self.interrupt_reason = reason


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
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        hermes = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(hermes, "_ensure_sigterm_handler", lambda: None)

        class _FailingInterruptedAIAgent:
            def __init__(self, **kwargs) -> None:
                self._build_api_kwargs = lambda _messages: {}

            def run_conversation(self, *args, **kwargs) -> dict:
                hermes.interrupted_agents.add(id(self))
                raise RuntimeError("agent failed")

        monkeypatch.setattr("run_agent.AIAgent", _FailingInterruptedAIAgent)

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
        "outcome",
        [
            {"failed": True, "error": "First response truncated due to output length limit"},
            {"partial": True, "error": "Response truncated due to output length limit"},
            {"partial": True, "error": "Context length exceeded (100 tokens). Cannot compress further."},
            {"partial": True, "error": "Model generated invalid tool call: invalid"},
            {"completed": True},
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

    async def test_host_path_retains_provider_failure_for_masked_verification(self, monkeypatch) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args: "http://model:8000/v1")
        monkeypatch.setattr(HermesAgent, "_ensure_sigterm_handler", lambda *_: None)
        runner = MagicMock()
        runner.run_conversation.return_value = {"failed": True, "error": "HTTP 500", "messages": []}
        monkeypatch.setattr("run_agent.AIAgent", MagicMock(return_value=runner))
        response = await hermes._create_response(NeMoGymResponseCreateParamsNonStreaming(input="hi"))
        assert response.status == "failed"
        assert response.error.message == "HTTP 500"
        assert response.metadata["provider_failed"] == "true"


class TestSandboxHermesInstall:
    """The sandbox installs whatever Hermes this server has installed, so requirements.txt is the only pin."""

    @staticmethod
    def _installed(monkeypatch, *, version: str, direct_url: dict | None) -> None:
        distribution = SimpleNamespace(
            version=version,
            read_text=lambda name: json.dumps(direct_url) if direct_url is not None else None,
        )
        monkeypatch.setattr("importlib.metadata.distribution", lambda name: distribution)

    def test_git_install_becomes_a_github_archive_keyed_by_commit(self, monkeypatch) -> None:
        commit = "a" * 40
        self._installed(
            monkeypatch,
            version="0.6.0",
            direct_url={"url": "https://github.com/cmunley1/hermes-agent.git", "vcs_info": {"commit_id": commit}},
        )

        requirement, key = _sandbox_hermes_install()

        assert requirement == f"hermes-agent[mcp] @ https://github.com/cmunley1/hermes-agent/archive/{commit}.tar.gz"
        assert key == commit[:12]

    def test_release_install_pins_the_version(self, monkeypatch) -> None:
        self._installed(monkeypatch, version="0.7.1", direct_url=None)

        assert _sandbox_hermes_install() == ("hermes-agent[mcp]==0.7.1", "0.7.1")

    def test_git_install_outside_github_is_rejected(self, monkeypatch) -> None:
        self._installed(
            monkeypatch,
            version="0.6.0",
            direct_url={"url": "https://gitlab.example.com/hermes-agent", "vcs_info": {"commit_id": "abc"}},
        )

        with pytest.raises(RuntimeError, match="gitlab.example.com"):
            _sandbox_hermes_install()


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

        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        seen: dict = {}

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                seen["base_url"] = kwargs.get("base_url")
                self._build_api_kwargs = lambda _messages: {}
                self.compression_enabled = True

            def run_conversation(self, *args, **kwargs) -> dict:
                return {"messages": [{"role": "assistant", "content": "ok"}]}

        monkeypatch.setattr("run_agent.AIAgent", _StubAIAgent)
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
        import nemo_gym.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(**config_kwargs), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        seen: dict = {}

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                seen["max_tokens"] = kwargs.get("max_tokens")
                self._build_api_kwargs = lambda _messages: {}
                self.compression_enabled = True

            def run_conversation(self, *args, **kwargs) -> dict:
                return {"messages": [{"role": "assistant", "content": "ok"}]}

        monkeypatch.setattr("run_agent.AIAgent", _StubAIAgent)
        return agent, seen

    def test_max_tokens_passed_to_ai_agent(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch, max_tokens=4096)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] == 4096

    def test_max_tokens_defaults_to_none(self, monkeypatch) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        agent, seen = self._agent_and_seen(monkeypatch)
        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["max_tokens"] is None


class TestObservability:
    @pytest.mark.parametrize(
        ("terminal_backend", "runtime_gap"),
        [("local", "no_sandbox_runtime"), ("docker", "sandbox_observation_unavailable")],
    )
    def test_observation_failure_does_not_change_response(self, monkeypatch, terminal_backend, runtime_gap) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(terminal_backend=terminal_backend), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        class _StubAIAgent:
            def __init__(self, **kwargs) -> None:
                self._build_api_kwargs = lambda _messages: {}

            def run_conversation(self, *args, **kwargs) -> dict:
                return {
                    "completed": True,
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok"},
                    ],
                }

        monkeypatch.setattr("run_agent.AIAgent", _StubAIAgent)
        body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
        baseline = asyncio.run(agent.responses(request=None, body=body))

        def fail_finish(*args, **kwargs):
            raise RuntimeError("observer failed")

        monkeypatch.setattr(HermesAgentObserver, "finish", fail_finish)
        episode = asyncio.run(agent._create_episode(body=body, rollout_id="rid"))

        assert episode.response.output == baseline.output
        assert episode.response.usage == baseline.usage
        assert [gap.code for gap in episode.observations.gaps] == [
            "observation_capture_failed",
            runtime_gap,
        ]

    @pytest.mark.parametrize("provider_failed", [False, True])
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
        if provider_failed:
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
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(agent, "_ensure_sigterm_handler", lambda: None)

        class _FailingAIAgent:
            def __init__(self, **kwargs) -> None:
                self._build_api_kwargs = lambda _messages: {}

            def run_conversation(self, *args, **kwargs) -> dict:
                raise ValueError("agent failed")

        monkeypatch.setattr("run_agent.AIAgent", _FailingAIAgent)
        monkeypatch.setattr(
            HermesAgentObserver,
            "finish",
            MagicMock(side_effect=RuntimeError("observer failed")),
        )

        with pytest.raises(ValueError, match="agent failed"):
            asyncio.run(
                agent._create_episode(
                    body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                    rollout_id="rid",
                )
            )
