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
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.sandbox import SandboxPtyError
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    HermesAgentSessionState,
    ModelServerRef,
    ResourcesServerRef,
    RunnerCleanup,
    _split_input_to_user_and_history,
    _trajectory_to_output_items,
)
from responses_api_agents.hermes_agent.observability import HermesAgentObserver
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


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
            config=_config(enabled_toolsets=["terminal"]),
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
        assert sandbox.exec.await_count == 2
        assert sandbox.upload.await_count == 3

    async def test_sandbox_access_requires_terminal_only_mode(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = AgentSeedSessionRequest(
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="test", task_id="task"),
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref="runtime",
                    descriptor={"sandbox_id": "sandbox"},
                ),
                workdir="/app",
            ),
        )

        with pytest.raises(HTTPException, match=r"enabled_toolsets: \[terminal\]"):
            await hermes._initialize_agent_session_state("session", body)

    async def test_close_rejects_a_different_episode(self) -> None:
        hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        seeded = AgentSeedSessionRequest(
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


class TestSandboxSessionCleanup:
    def _session(self) -> tuple[HermesAgent, HermesAgentSessionState]:
        hermes = HermesAgent(
            config=_config(enabled_toolsets=["terminal"]),
            server_client=MagicMock(spec=ServerClient),
        )
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
        hermes._download_json = AsyncMock(return_value={"cleanup_confirmed": True})
        state = HermesAgentSessionState(
            request=AgentSeedSessionRequest(
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="test", task_id="task"),
            ),
            sandbox=sandbox,
            workdir="/app",
            session_dir="/tmp/nemo-gym-hermes-sessions/session",
            runner_session=AsyncMock(),
            runner_cleanup=RunnerCleanup.UNCONFIRMED,
            observations=AgentObservationBundle(source="hermes"),
        )
        return hermes, state

    async def test_close_terminates_runner_before_disconnect(self) -> None:
        hermes, state = self._session()
        exited = asyncio.Event()

        async def wait_exit() -> int:
            await exited.wait()
            return -signal.SIGTERM

        runner = state.runner_session
        runner.send_signal.side_effect = lambda _: exited.set()
        state.runner_exit_task = asyncio.create_task(wait_exit())
        events = MagicMock()
        events.attach_mock(runner, "runner")
        events.attach_mock(state.sandbox, "sandbox")

        result = await hermes._close_agent_session_state(state)

        assert [call[0] for call in events.mock_calls] == [
            "runner.send_signal",
            "runner.close",
            "sandbox.exec",
            "sandbox.disconnect",
        ]
        runner.send_signal.assert_awaited_once_with("SIGTERM")
        assert state.runner_session is None
        assert state.runner_exit_task is None
        assert state.runner_cleanup is RunnerCleanup.CONFIRMED
        assert state.sandbox.exec.await_args.args[0] == f"rm -rf {state.session_dir}"
        state.sandbox.disconnect.assert_awaited_once()
        state.sandbox.stop.assert_not_awaited()
        assert result is state.observations

    @pytest.mark.parametrize("error_type", [TimeoutError, asyncio.CancelledError, SandboxPtyError])
    async def test_interrupted_termination_retains_handles_for_retry(self, error_type: type[BaseException]) -> None:
        hermes, state = self._session()
        runner = state.runner_session
        state.runner_exit_task = asyncio.create_task(asyncio.sleep(60, result=0))
        watcher = state.runner_exit_task
        runner.send_signal.side_effect = error_type("interrupted")

        try:
            with pytest.raises(error_type):
                await hermes._close_agent_session_state(state)
            assert state.runner_session is runner
            assert state.runner_exit_task is watcher
            assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
            runner.close.assert_not_awaited()
            state.sandbox.disconnect.assert_not_awaited()
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

        # The same provider session can be retried after a confirmed exit.
        state.runner_exit_task = asyncio.create_task(asyncio.sleep(0, result=0))
        await state.runner_exit_task
        await hermes._close_agent_session_state(state)

        runner.close.assert_awaited_once()
        assert state.runner_session is None
        state.sandbox.disconnect.assert_awaited_once()
        state.sandbox.stop.assert_not_awaited()

    async def test_failed_exit_watcher_does_not_complete_close(self) -> None:
        hermes, state = self._session()

        async def failed_watch() -> int:
            raise SandboxPtyError("exit status unavailable")

        state.runner_exit_task = asyncio.create_task(failed_watch())
        await asyncio.gather(state.runner_exit_task, return_exceptions=True)
        with pytest.raises(SandboxPtyError, match="exit status unavailable"):
            await hermes._close_agent_session_state(state)
        assert state.runner_session is not None
        state.runner_session.close.assert_not_awaited()
        state.sandbox.disconnect.assert_not_awaited()
        state.sandbox.stop.assert_not_awaited()

    async def test_provider_close_failure_retains_handles(self) -> None:
        hermes, state = self._session()
        runner = state.runner_session
        state.runner_exit_task = asyncio.create_task(asyncio.sleep(0, result=0))
        await state.runner_exit_task
        runner.close.side_effect = [SandboxPtyError("close failed"), None]
        with pytest.raises(SandboxPtyError, match="close failed"):
            await hermes._close_agent_session_state(state)
        assert state.runner_session is runner
        assert state.runner_cleanup is RunnerCleanup.CONFIRMED
        state.sandbox.disconnect.assert_not_awaited()
        await hermes._close_agent_session_state(state)
        assert runner.close.await_count == 2
        assert state.runner_session is None
        hermes._download_json.assert_awaited_once()

    @pytest.mark.skipif(os.name != "posix", reason="Requires POSIX process groups")
    async def test_termination_escalates_and_waits_for_process_exit(self) -> None:
        hermes, state = self._session()
        hermes.config.session_close_timeout_seconds = 0.05
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            assert await asyncio.wait_for(process.stdout.readline(), timeout=5) == b"ready\n"
            runner = state.runner_session
            runner.send_signal.side_effect = lambda name: os.killpg(process.pid, getattr(signal, name))
            state.runner_exit_task = asyncio.create_task(process.wait())
            hermes._download_json = AsyncMock(return_value={"cleanup_confirmed": False})
            with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
                await hermes._terminate_sandbox_runner(state)
            assert [call.args[0] for call in runner.send_signal.await_args_list] == ["SIGTERM", "SIGKILL"]
            assert state.runner_session is runner
            assert process.returncode == -signal.SIGKILL
            with pytest.raises(ProcessLookupError):
                os.killpg(process.pid, 0)
        finally:
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)


@pytest.mark.parametrize("template_location", ["extra_body", "top_level"])
@pytest.mark.parametrize(
    "hermes_error",
    [None, "Model generated invalid tool call: finish", "Model generated only think blocks after 3 retries"],
)
@pytest.mark.parametrize(
    "existing_metadata",
    [None, {"trace": "kept", "chat_template_kwargs": '{"existing": true, "enable_thinking": false}'}],
)
async def test_sandbox_relay_preserves_template_overrides_in_gym_metadata(
    monkeypatch: pytest.MonkeyPatch,
    template_location: str,
    hermes_error: str | None,
    existing_metadata: dict[str, str] | None,
) -> None:
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(return_value=_FakeResponse({"id": "completion"}))
    hermes = HermesAgent(config=_config(enabled_toolsets=["terminal"]), server_client=client)
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        MagicMock(return_code=0, stdout=value, stderr="") for value in ("request", "", "output")
    ]
    sandbox.pty = MagicMock()
    sandbox.pty.create = AsyncMock(return_value=AsyncMock())
    state = HermesAgentSessionState(
        request=AgentSeedSessionRequest(
            episode_id=EpisodeId(rollout_id="rollout", attempt=2),
            task_id=TaskId(taskset="test", task_id="task"),
        ),
        sandbox=sandbox,
        workdir="/app",
        session_dir="/tmp/session",
    )
    template_kwargs = {"enable_thinking": True, "truncate_history_thinking": False}
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "working", "reasoning_content": "preserved thinking"},
    ]
    model_request = {"model": "model", "messages": messages, "metadata": existing_metadata}
    if template_location == "extra_body":
        model_request["extra_body"] = {"chat_template_kwargs": template_kwargs}
    else:
        model_request["chat_template_kwargs"] = template_kwargs
        # An explicit top-level value already takes precedence over extra_body.
        model_request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    output = {
        "result": {
            "messages": messages,
            "completed": hermes_error is None,
            "partial": hermes_error is not None,
            "error": hermes_error,
            "api_calls": 1,
        },
        "runtime": {"pid": 123},
    }
    monkeypatch.setattr(HermesAgent, "_upload_json", AsyncMock())
    monkeypatch.setattr(
        HermesAgent, "_download_json", AsyncMock(side_effect=[model_request, output, {"cleanup_confirmed": True}])
    )
    request = Request(
        {
            "type": "http",
            "path": "/ng-rollout/rollout-a2/v1/responses",
            "headers": [],
            "path_params": {"rollout_id": "rollout-a2"},
        }
    )

    episode = await hermes._run_sandbox_episode(
        request=request,
        body=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}]),
        agent_session_id="session",
        state=state,
    )

    client.post.assert_awaited_once()
    assert client.post.await_args.kwargs["url_path"] == "/ng-rollout/rollout-a2/v1/chat/completions"
    assert client.post.await_args.kwargs["server_name"] == hermes.config.model_server.name
    forwarded = client.post.await_args.kwargs["json"]
    validated = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(forwarded)
    assert "extra_body" not in forwarded
    assert "chat_template_kwargs" not in forwarded
    assert validated.messages == messages
    expected = ({"existing": True} if existing_metadata else {}) | template_kwargs
    assert json.loads(validated.metadata["chat_template_kwargs"]) == expected
    if existing_metadata:
        assert validated.metadata["trace"] == "kept"
        assert existing_metadata["chat_template_kwargs"] == '{"existing": true, "enable_thinking": false}'
    assert episode.response.status == ("failed" if hermes_error else "completed")
    assert episode.response.metadata["turns"] == "1"
    assert episode.response.metadata["partial"] == ("true" if hermes_error else "false")
    if hermes_error:
        assert episode.response.error.message == hermes_error
        assert episode.response.metadata["hermes_error"] == hermes_error
        assert episode.response.output[-1].content[0].text == "working"
    else:
        assert episode.response.error is None
    assert episode.observations.records[0].status == ("failed" if hermes_error else "completed")
    assert episode.observations.records[0].model_calls[0].response_id == "completion"
    assert episode.observations.records[0].model_calls[0].model_ref == hermes.config.model_server
    assert state.runner_session is None

    # Exercise the real vLLM preprocessing boundary: request settings must win
    # over model defaults without discarding other configured template options.
    model = VLLMModel(
        config=VLLMModelConfig(
            name="model",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            base_url="http://unused.invalid/v1",
            api_key="unused",
            model="model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            chat_template_kwargs={"enable_thinking": False, "model_default": "kept"},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    engine_request = model._preprocess_chat_completion_create_params(request, validated.model_dump(exclude_unset=True))
    assert engine_request["chat_template_kwargs"] == {"model_default": "kept"} | expected


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
