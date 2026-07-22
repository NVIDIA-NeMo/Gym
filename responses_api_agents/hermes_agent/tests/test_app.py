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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
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

        asyncio.run(agent.responses(request=None, body=NeMoGymResponseCreateParamsNonStreaming(input="hi")))
        assert seen["base_url"] == "http://h:1/v1"

        episode = asyncio.run(
            agent.responses_with_observations(
                request=None,
                body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                rollout_id="rid",
            )
        )
        assert seen["base_url"] == "http://h:1/ng-rollout/rid/v1"
        assert episode.observations.source == "hermes"
        assert episode.observations.invocations[0].invocation_id == "root"


class TestObservability:
    def test_observation_failure_does_not_change_response(self, monkeypatch) -> None:
        import nemo_gym.base_responses_api_agent as base_agent
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
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
        episode = asyncio.run(agent.responses_with_observations(request=None, body=body, rollout_id="rid"))

        assert episode.response.output == baseline.output
        assert episode.response.usage == baseline.usage
        assert [gap.code for gap in episode.observations.gaps] == ["observation_capture_failed"]

    def test_run_passes_rollout_id_to_verifier(self) -> None:
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

        with patch.object(HermesAgent, "responses_with_observations", observed_response):
            asyncio.run(agent.run(request, body))

        assert server_client.post.await_args_list[-1].kwargs["json"]["rollout_id"] == "1-2"

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
                agent.responses_with_observations(
                    request=None,
                    body=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
                    rollout_id="rid",
                )
            )
