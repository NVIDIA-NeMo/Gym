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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from nemo_gym.sandbox import SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent import app as hermes_module
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _download_uv,
    _host_uv_version,
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
        assert state.session_dir.endswith("/session")
        # uname probe, mkdir, install
        assert sandbox.exec.await_count == 3
        assert sandbox.upload.await_count == 3

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

        with pytest.raises(ValueError, match="episode_id does not match"):
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

        with pytest.raises(ValueError, match="already closed"):
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

    def test_session_activation_rejects_hermes_error_result(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        with pytest.raises(RuntimeError, match="model request failed"):
            hermes._response_from_result(
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                result={"error": "model request failed", "messages": []},
                model_name="model",
                fail_on_error=True,
            )


class TestSplitInputToUserAndHistory:
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


class TestSandboxUv:
    """The uploaded uv must match the sandbox's architecture, not the agent host's."""

    @staticmethod
    def _sandbox(arch: str) -> AsyncMock:
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0, stdout=f"{arch}\n", stderr="")
        return sandbox

    async def test_same_architecture_uses_host_uv(self, monkeypatch, tmp_path) -> None:
        hermes = HermesAgent(config=_config(uv_cache_dir=str(tmp_path)), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(hermes_module.shutil, "which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr(hermes_module.platform, "machine", lambda: "aarch64")

        assert await hermes._uv_for_sandbox(self._sandbox("aarch64"), "/app") == hermes_module.Path("/usr/bin/uv")
        assert not any(tmp_path.iterdir())

    async def test_other_architecture_downloads_matching_build_once(self, monkeypatch, tmp_path) -> None:
        hermes = HermesAgent(config=_config(uv_cache_dir=str(tmp_path)), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(hermes_module.shutil, "which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr(hermes_module.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(hermes_module, "_host_uv_version", lambda uv: "0.12.3")
        downloads: list[tuple[str, str]] = []

        def fake_download(version, target, destination):
            downloads.append((version, target))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"uv")

        monkeypatch.setattr(hermes_module, "_download_uv", fake_download)

        first = await hermes._uv_for_sandbox(self._sandbox("x86_64"), "/app")
        second = await hermes._uv_for_sandbox(self._sandbox("x86_64"), "/app")

        assert first == second == tmp_path / "0.12.3" / "x86_64-unknown-linux-gnu" / "uv"
        assert downloads == [("0.12.3", "x86_64-unknown-linux-gnu")]

    async def test_unknown_architecture_is_an_error(self, monkeypatch, tmp_path) -> None:
        hermes = HermesAgent(config=_config(uv_cache_dir=str(tmp_path)), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(hermes_module.shutil, "which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr(hermes_module.platform, "machine", lambda: "aarch64")
        with pytest.raises(RuntimeError, match="No uv build"):
            await hermes._uv_for_sandbox(self._sandbox("riscv64"), "/app")

    async def test_missing_host_uv_is_an_error(self, monkeypatch) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        monkeypatch.setattr(hermes_module.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="requires uv"):
            await hermes._uv_for_sandbox(self._sandbox("x86_64"), "/app")

    def test_host_uv_version_parses(self, monkeypatch) -> None:
        monkeypatch.setattr(
            hermes_module.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout="uv 0.12.3 (aarch64-unknown-linux-gnu)\n"),
        )
        assert _host_uv_version("/usr/bin/uv") == "0.12.3"
        monkeypatch.setattr(hermes_module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="nope"))
        with pytest.raises(RuntimeError, match="Unexpected uv version"):
            _host_uv_version("/usr/bin/uv")

    def test_download_uv_extracts_binary(self, monkeypatch, tmp_path) -> None:
        import tarfile

        release_dir = tmp_path / "0.12.3"
        release_dir.mkdir()
        archive = release_dir / "uv-x86_64-unknown-linux-gnu.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            binary = tmp_path / "uv"
            binary.write_bytes(b"#!/bin/sh\necho uv\n")
            tar.add(binary, arcname="uv-x86_64-unknown-linux-gnu/uv")
        monkeypatch.setattr(hermes_module, "_UV_RELEASE_URL", tmp_path.as_uri() + "/{version}/uv-{target}.tar.gz")
        destination = tmp_path / "cache" / "0.12.3" / "x86_64-unknown-linux-gnu" / "uv"

        _download_uv("0.12.3", "x86_64-unknown-linux-gnu", destination)

        assert destination.read_bytes().startswith(b"#!/bin/sh")
        assert destination.stat().st_mode & 0o111
