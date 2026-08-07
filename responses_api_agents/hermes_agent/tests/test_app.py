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
from unittest.mock import MagicMock

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
    _split_input_to_user_and_history,
    _trajectory_to_output_items,
)


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
            {"role": "assistant", "content": "a", "generation_token_ids": [3, 4]},
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
                "generation_token_ids": [5, 6],
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
        msgs = [None, "string", {"role": "assistant", "content": "ok", "generation_token_ids": [7]}]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1

    def test_assistant_without_tokens_is_not_trainable(self) -> None:
        """Locally-synthesised assistant turns (max-iteration summary, error apology) carry no
        token ids. They must not be presented to the trainer as policy output."""
        msgs = [{"role": "assistant", "content": "I hit the iteration limit."}]
        out = _trajectory_to_output_items(msgs, 0)
        assert len(out) == 1
        assert isinstance(out[0], NeMoGymResponseOutputMessage)
        assert not isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert out[0].content[0].text == "I hit the iteration limit."

    def test_empty_token_ids_are_not_trainable(self) -> None:
        """An explicitly empty generation_token_ids is as untrainable as an absent one."""
        msgs = [{"role": "assistant", "content": "x", "prompt_token_ids": [], "generation_token_ids": []}]
        out = _trajectory_to_output_items(msgs, 0)
        assert isinstance(out[0], NeMoGymResponseOutputMessage)
        assert not isinstance(out[0], NeMoGymResponseOutputMessageForTraining)

    def test_assistant_without_tokens_still_emits_tool_calls(self) -> None:
        """Downgrading the message type must not drop the tool calls attached to it."""
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
        assert isinstance(out[0], NeMoGymResponseOutputMessage)
        assert not isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert isinstance(out[1], NeMoGymResponseFunctionToolCall)
        assert out[1].name == "terminal"
        assert isinstance(out[2], NeMoGymFunctionCallOutput)

    def test_routed_experts_survive_on_trainable_turns(self) -> None:
        """The trainable branch must keep forwarding routed_experts (added upstream in #1716)."""
        msgs = [{"role": "assistant", "content": "a", "generation_token_ids": [2], "routed_experts": [[[1, 2]]]}]
        out = _trajectory_to_output_items(msgs, 0)
        assert isinstance(out[0], NeMoGymResponseOutputMessageForTraining)
        assert out[0].routed_experts == [[[1, 2]]]


class TestRolloutCorrelation:
    """The prefixed self-call path makes responses() build AIAgent with the same model URL prefix."""

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


class TestResponsesEmitsNoFabricatedTokens:
    """The /responses fallback path must never invent training token ids."""

    async def _run(self, monkeypatch, messages, error="boom") -> list:
        import asyncio as _asyncio

        import run_agent

        import nemo_gym.base_responses_api_agent as base_agent

        # resolve_model_base_url is a public method on a pydantic model, so it cannot be
        # monkeypatched on the instance. Stub out what it reads instead, as TestRolloutCorrelation does.
        monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        server_client._build_server_base_url = lambda _cfg: "http://h:1"
        agent = HermesAgent(config=_config(), server_client=server_client)
        monkeypatch.setattr(run_agent, "AIAgent", MagicMock())

        async def _fake_to_thread(fn, *args, **kwargs):
            return {"messages": messages, "error": error}

        monkeypatch.setattr(_asyncio, "to_thread", _fake_to_thread)
        body = NeMoGymResponseCreateParamsNonStreaming(input="q", model="m")
        resp = await agent.responses(MagicMock(), body)
        return list(resp.output)

    async def test_untrainable_trajectory_is_not_padded_with_fake_tokens(self, monkeypatch) -> None:
        """Previously this emitted ...ForTraining with generation_token_ids=[0]."""
        out = await self._run(monkeypatch, [{"role": "assistant", "content": "synthetic summary"}])
        assert not any(isinstance(item, NeMoGymResponseOutputMessageForTraining) for item in out)
        assert any(isinstance(item, NeMoGymResponseOutputMessage) for item in out)

    async def test_does_not_reuse_token_ids_from_input_history(self, monkeypatch) -> None:
        """Previously the fallback copied token ids off the last assistant message anywhere in
        `messages` — including the prompt history, i.e. tokens this rollout never generated."""
        messages = [
            {"role": "assistant", "content": "from history", "prompt_token_ids": [1, 2], "generation_token_ids": [9]},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "synthetic summary"},
        ]
        out = await self._run(monkeypatch, messages)
        assert all(getattr(item, "generation_token_ids", None) != [9] for item in out)
        assert not any(isinstance(item, NeMoGymResponseOutputMessageForTraining) for item in out)

    async def test_real_generation_is_still_trainable(self, monkeypatch) -> None:
        """A genuine policy turn is unaffected and keeps its token ids."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "prompt_token_ids": [1], "generation_token_ids": [2, 3]},
        ]
        out = await self._run(monkeypatch, messages, error=None)
        trainable = [item for item in out if isinstance(item, NeMoGymResponseOutputMessageForTraining)]
        assert len(trainable) == 1
        assert trainable[0].generation_token_ids == [2, 3]
