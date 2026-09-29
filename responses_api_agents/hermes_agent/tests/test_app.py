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
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
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
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    HermesAgentSessionState,
    ModelServerRef,
    ResourcesServerRef,
    _split_input_to_user_and_history,
    _trajectory_to_output_items,
)
from responses_api_agents.hermes_agent.observability import HermesAgentObserver


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

    async def test_sandbox_access_selects_runtime_provider(self, monkeypatch) -> None:
        hermes = HermesAgent(
            config=_config(enabled_toolsets=["terminal", "web"]),
            server_client=MagicMock(spec=ServerClient),
        )
        provider_config = {"opensandbox": {"connection": {}}}
        resolve = MagicMock(return_value=provider_config)
        provider = AsyncMock()
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
        connect = AsyncMock(return_value=sandbox)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.shutil.which", lambda name: "/test/uv")
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.get_global_config_dict", lambda: {"runtime": {}})
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.resolve_provider_config", resolve)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: provider)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.AsyncSandbox.connect", connect)

        state = await hermes._initialize_agent_session_state(
            "session",
            AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="test", task_id="task"),
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
        assert state.sandbox is sandbox
        assert state.workdir == "/app"
        assert state.session_dir.startswith("/tmp/nemo-gym-hermes-sessions/")
        assert len(state.session_dir.rsplit("/", 1)[-1]) == 32
        assert sandbox.exec.await_count == 2
        assert sandbox.upload.await_count == 2

    async def test_seed_installs_hermes_when_it_does_not_import(self, monkeypatch) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        sandbox = AsyncMock()
        import_results = iter([1, 0])

        async def exec_(command, **_kwargs):
            if "import run_agent" in command:
                return MagicMock(return_code=next(import_results), stdout="", stderr="")
            return MagicMock(return_code=0, stdout="", stderr="")

        sandbox.exec.side_effect = exec_
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.get_global_config_dict", lambda: {"runtime": {}})
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.resolve_provider_config", MagicMock())
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: AsyncMock())
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.AsyncSandbox.connect", AsyncMock(return_value=sandbox)
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.shutil.which", lambda _name: "/usr/bin/uv")

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

    async def test_missing_sandbox_access_uses_configured_fallback(self, monkeypatch) -> None:
        hermes = HermesAgent(
            config=_config(
                enabled_toolsets=["terminal", "web"],
                sandbox_provider="runtime",
                sandbox_config={"workdir": "/fallback"},
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        provider = AsyncMock()
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
        sandbox_factory = MagicMock(return_value=sandbox)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.get_global_config_dict", lambda: {"runtime": {}})
        monkeypatch.setattr(
            "responses_api_agents.hermes_agent.app.resolve_provider_config",
            MagicMock(return_value={"local": {}}),
        )
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.create_provider", lambda config: provider)
        monkeypatch.setattr("responses_api_agents.hermes_agent.app.AsyncSandbox", sandbox_factory)

        state = await hermes._initialize_agent_session_state(
            "session",
            AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="test", task_id="task"),
            ),
        )

        sandbox_factory.assert_called_once_with(provider)
        sandbox.start.assert_awaited_once_with(SandboxSpec(workdir="/fallback"))
        assert state.sandbox is sandbox
        assert state.workdir == "/fallback"
        assert state.owns_sandbox is True
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
        hermes._agent_sessions["session"] = MagicMock(request=seeded)
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
        hermes._close_agent_session_state = AsyncMock(return_value=None)
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

    async def test_seed_rejects_required_episode_tool_grants(self) -> None:
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

        with pytest.raises(HTTPException, match="required"):
            await hermes.seed_agent_session(SimpleNamespace(session={}), body)

        hermes._initialize_agent_session_state.assert_not_awaited()

    @staticmethod
    def _sandbox_session(monkeypatch, sandbox) -> tuple[HermesAgent, SimpleNamespace, AgentSeedSessionRequest]:
        import nemo_gym.base_responses_api_agent as base_agent

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://model-server:1"
        hermes = HermesAgent(config=_config(), server_client=server_client)
        seed = AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
        )
        hermes._agent_sessions["session"] = HermesAgentSessionState(
            request=seed, sandbox=sandbox, workdir=None, session_dir="/session"
        )
        request = SimpleNamespace(
            session={"agent_session_id": "session"},
            path_params={"rollout_id": seed.episode_id.capture_key},
        )
        original_download = hermes._download_json

        async def download(sandbox, remote_path):
            if remote_path.endswith("/cleanup.json"):
                commands = getattr(sandbox, "commands", None)
                return {"cleanup_confirmed": commands is None or any("kill -TERM" in cmd for cmd in commands)}
            return await original_download(sandbox, remote_path)

        hermes._download_json = AsyncMock(side_effect=download)
        return hermes, request, seed

    async def test_sandbox_activation_calls_the_model_server_directly(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.uploaded: dict = {}

            async def upload(self, local_path, remote_path) -> None:
                self.uploaded[remote_path] = json.loads(Path(local_path).read_text())

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

        runner_input = sandbox.uploaded["/session/input.json"]
        assert runner_input["model_base_url"] == f"http://model-server:1/ng-rollout/{seed.episode_id.capture_key}/v1"
        assert runner_input["user_message"] == "fix bug"
        assert response.metadata["harness_execution"] == "sandbox"

    async def test_runner_timeout_stops_the_runner(self, monkeypatch) -> None:
        class _Sandbox:
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def upload(self, *_args) -> None:
                pass

            # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                self.commands.append(command)
                if "runner.pid" in command and "exec " in command:
                    return SandboxExecResult(stdout=None, stderr="timed out", return_code=124, error_type="timeout")
                return SandboxExecResult(stdout="", stderr="", return_code=0)

        sandbox = _Sandbox()
        hermes, request, _ = self._sandbox_session(monkeypatch, sandbox)

        with pytest.raises(TimeoutError, match="sandbox_runner_timeout_seconds"):
            await hermes.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))

        stop = [command for command in sandbox.commands if "runner.stop" in command and "kill -TERM" in command]
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

            # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
            async def exec(
                self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
            ) -> SandboxExecResult:
                self.commands.append(command)
                if "runner.pid" in command and "exec " in command:
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

        stop = [command for command in sandbox.commands if "runner.stop" in command and "kill -TERM" in command]
        assert len(stop) == 1
        assert "session" not in hermes._agent_sessions
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

    def test_explicit_fail_on_error_rejects_hermes_error_result(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        with pytest.raises(RuntimeError, match="model request failed"):
            hermes._response_from_result(
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                result={"error": "model request failed", "messages": []},
                model_name="model",
                fail_on_error=True,
            )


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

    def test_run_returns_observations_without_leaking_internal_attachment(self) -> None:
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
