# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import itertools
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    NeMoGymLLM,
    NeMoGymSandboxEnvironment,
    NeMoGymTerminus2,
    Terminus2Agent,
    Terminus2AgentConfig,
    _instruction,
)
from responses_api_agents.terminus_2_sandboxed_agent.observability import TerminusObservations


def test_instruction_joins_text_content():
    assert _instruction([{"content": [{"text": "first"}]}, {"content": "second"}]) == "first\n\nsecond"


@pytest.mark.asyncio
async def test_sandbox_environment_adapts_exec_and_is_dir():
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), session_id="session-1")

    result = await environment.exec("pwd", timeout_sec=12, user="root", cwd="/work")

    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.return_code == 0
    assert await environment.is_dir("/workspace")
    assert sandbox_calls == [
        ("pwd", {"timeout_s": 12, "cwd": "/work", "user": "root", "env": None}),
        ('test -d "/workspace"', {"user": None}),
    ]


@pytest.mark.asyncio
async def test_sandbox_environment_uses_sandbox_exec_for_stateful_commands():
    calls = []

    async def sandbox_exec(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="output", stderr=None, return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)
    environment = NeMoGymSandboxEnvironment(sandbox, logs_dir=SimpleNamespace(), session_id="session-1")

    await environment.exec("tmux new-session")

    assert calls == [("tmux new-session", {"timeout_s": None, "cwd": None, "user": None, "env": None})]


def test_agent_implements_required_responses_endpoint():
    assert not getattr(Terminus2Agent, "__abstractmethods__", set())


@pytest.mark.asyncio
async def test_nemo_gym_llm_records_every_responses_request_and_output():
    class Client:
        def __init__(self):
            self.requests = []

        async def create_response(self, **kwargs):
            self.requests.append(kwargs)
            index = len(self.requests)
            return NeMoGymResponse(
                id=f"resp_{index}",
                created_at=0,
                model="policy_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{index}",
                        content=[
                            NeMoGymResponseOutputText(type="output_text", text=f"answer {index}", annotations=[])
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
                usage=NeMoGymResponseUsage(
                    input_tokens=10,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=2),
                    output_tokens=3,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=13,
                ),
            )

    client = Client()
    llm = NeMoGymLLM(
        client=client,
        model_name="policy_model",
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=60,
    )

    first = await llm.call("first")
    second = await llm.call(
        "second",
        message_history=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer 1"}],
        previous_response_id="resp_1",
    )
    third = await llm.call(
        "third",
        message_history=[{"role": "user", "content": "compacted summary"}],
        previous_response_id="resp_2",
    )

    assert first.content == "answer 1"
    assert first.usage.prompt_tokens == 10
    assert second.content == "answer 2"
    assert third.content == "answer 3"
    # The x-ng-* join headers ride along with every request; the body is unchanged.
    assert all(request["extra_headers"]["x-ng-client-call-id"] for request in client.requests)
    assert [{k: v for k, v in request.items() if k != "extra_headers"} for request in client.requests] == [
        {"model": "policy_model", "input": [{"content": "first", "role": "user", "type": "message"}]},
        {
            "model": "policy_model",
            "input": [
                {"content": "first", "role": "user", "type": "message"},
                {"content": "answer 1", "role": "assistant", "type": "message"},
                {"content": "second", "role": "user", "type": "message"},
            ],
        },
        {
            "model": "policy_model",
            "input": [
                {"content": "compacted summary", "role": "user", "type": "message"},
                {"content": "third", "role": "user", "type": "message"},
            ],
        },
    ]
    assert [item.content for item in llm.trajectory if isinstance(item, NeMoGymEasyInputMessage)] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_nemo_gym_llm_keeps_a_timed_out_attempt_separate_from_the_retry():
    """A retried model call is one logical call with two attempts, never one blurred span."""

    class Client:
        def __init__(self):
            self.calls = 0

        async def create_response(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(10)  # cancelled by the LLM's request timeout
            return NeMoGymResponse(
                id="resp_ok",
                created_at=0,
                model="policy_model",
                object="response",
                output=[
                    NeMoGymResponseOutputMessage(
                        id="msg",
                        content=[NeMoGymResponseOutputText(type="output_text", text="answer", annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
                usage=NeMoGymResponseUsage(
                    input_tokens=1,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens=1,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=2,
                ),
            )

    observations = TerminusObservations(
        invocation_id="inv",
        task_id="t",
        rollout_id="t-0",
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    llm = NeMoGymLLM(
        client=Client(),
        model_name="policy_model",
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=0.05,
        observations=observations,
    )

    result = await llm.call("prompt")

    assert result.content == "answer"
    [call] = observations.trajectory.model_calls
    assert [(a.attempt_index, a.status, a.error_type) for a in call.attempts] == [
        (1, "timeout", "TimeoutError"),
        (2, "completed", None),
    ]
    assert call.attempts[0].started_at <= call.attempts[0].completed_at <= call.attempts[1].started_at
    assert call.started_at <= call.attempts[0].started_at and call.attempts[1].completed_at <= call.completed_at
    assert "model_attempt_without_response" in {gap.code for gap in observations.trajectory.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize("dump_trajectory", [False, True])
@pytest.mark.parametrize("debug", [False, True])
async def test_execute_runs_terminus_in_seeded_sandbox(monkeypatch, dump_trajectory, debug):
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_1_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="swebench_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=100,
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        dump_trajectory=dump_trajectory,
        debug=debug,
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=60,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    set_level = MagicMock()
    monkeypatch.setattr(app_module.harbor_logger, "setLevel", set_level)
    server = Terminus2Agent(config=config, server_client=MagicMock(spec=ServerClient))
    sandbox_calls = []

    async def sandbox_exec(command, **kwargs):
        sandbox_calls.append((command, kwargs))
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(exec=sandbox_exec)

    class FakeTerminus:
        session = SimpleNamespace()

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._session = SimpleNamespace(stop=self.stop)
            self._times_spent = [1.0, 3.0]
            self._num_proactive_compactions = 0
            self._num_compactions = 2

        async def stop(self):
            return None

        async def setup(self, environment):
            await environment.exec("tmux setup")

        async def run(self, instruction, environment, context):
            assert instruction == "solve this"
            assert self.kwargs["dump_trajectory"] is dump_trajectory
            await environment.exec("tmux run")
            self.kwargs["llm"]._times_spent.extend([2.0, 4.0])
            self.kwargs["llm"]._num_compactions = 2
            context.n_input_tokens = 4
            context.n_output_tokens = 3
            self.kwargs["llm"].trajectory.append(
                NeMoGymResponseOutputMessage(
                    id="msg_done",
                    content=[NeMoGymResponseOutputText(type="output_text", text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

    class FakeContext:
        n_input_tokens = None
        n_cache_tokens = None
        n_output_tokens = None
        metadata = None

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", FakeTerminus)
    monkeypatch.setattr(app_module, "AgentContext", FakeContext)
    monkeypatch.setattr(Terminus2Agent, "base_url_for_run", lambda *_args, **_kwargs: "http://model")
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    # First read is _execute's start; every later read (phases, the invocation, the total) sees 20.0.
    elapsed_times = itertools.chain([10.0], itertools.repeat(20.0))
    monkeypatch.setattr(app_module, "perf_counter", lambda: next(elapsed_times))

    async def request_json():
        return {"task_id": "task"}

    request = SimpleNamespace(json=request_json, session={app_module.SESSION_ID_KEY: "session-1"})
    response, metrics = await server._execute(
        request,
        NeMoGymResponseCreateParamsNonStreaming(input="solve this"),
        sandbox,
    )

    assert metrics == {
        "terminus2_completed": True,
        "command_exec_times": [1.0, 3.0],
        "model_call_times": [2.0, 4.0],
        "average_command_exec_time": 2.0,
        "average_model_call_time": 3.0,
        "total_command_exec_time": 4.0,
        "total_model_call_time": 6.0,
        "command_exec_time_pct": 40.0,
        "model_call_time_pct": 60.0,
        "terminus2_time_taken": 10.0,
        "model_calls_gt_10min": 0,
        "num_proactive_compactions": 0,
        "num_compactions": 2,
        "error": None,
        "usages": [],
    }
    assert response.output[-1].content[0].text == "done"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 3
    if not debug:
        set_level.assert_called_once_with(logging.WARNING)
    else:
        set_level.assert_not_called()
    assert sandbox_calls == [
        ("mkdir -p /logs/agent", {"timeout_s": None, "cwd": None, "user": "root", "env": None}),
        ("tmux setup", {"timeout_s": None, "cwd": None, "user": None, "env": None}),
        ("tmux run", {"timeout_s": None, "cwd": None, "user": None, "env": None}),
    ]


# --- latency records: what the harness itself measures ------------------------------------------------------------


def _latency_response(index: int, *, text: str = "answer", input_tokens: int = 10) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{index}",
        created_at=0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{index}",
                role="assistant",
                status="completed",
                type="message",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ]
        if text
        else [],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=True,
        usage=NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=2),
            output_tokens=3,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=1),
            total_tokens=input_tokens + 3,
        ),
    )


class _PlannedClient:
    """Fake NeMoGymAsyncOpenAI: ``plan`` lists, per call, 'ok' | 'timeout' | 'empty'."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.requests = []

    async def create_response(self, **kwargs):
        self.requests.append(kwargs)
        action = self.plan.pop(0)
        if action == "timeout":
            await asyncio.sleep(10)
        return _latency_response(len(self.requests), text="" if action == "empty" else "answer")


def _observed_llm(plan, timeout=1.0):
    observations = TerminusObservations(
        invocation_id="inv",
        task_id="t",
        rollout_id="t-0",
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    client = _PlannedClient(plan)
    llm = NeMoGymLLM(
        client=client,
        model_name="policy_model",
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=timeout,
        observations=observations,
    )
    return llm, observations, client


@pytest.mark.asyncio
async def test_logical_call_records_its_own_timing_status_purpose_and_sends_the_join_headers():
    from harbor.llms.base import ContextLengthExceededError

    llm, observations, client = _observed_llm(["ok", "empty"])
    await llm.call("prompt", message_history=[])
    [call] = observations.trajectory.model_calls
    assert call.clock_domain.startswith("node:") and call.status == "ok" and call.model_call_purpose == "agent_step"
    assert call.duration_ms is not None and 0 <= call.duration_ms <= (call.completed_at - call.started_at) * 1000 + 50
    assert call.client_call_id == call.model_call_id and call.invocation_id == "inv"
    headers = client.requests[0]["extra_headers"]
    assert headers["x-ng-client-call-id"] == call.model_call_id and headers["x-ng-client-attempt"] == "1"
    assert headers["x-ng-purpose"] == "agent_step" and headers["x-ng-client-sent-at"].isdigit()
    # Gym answers an over-context request with an empty 200: the call is recorded as the rejection it is.
    with pytest.raises(ContextLengthExceededError):
        await llm.call("prompt", message_history=[])
    rejected = observations.trajectory.model_calls[1]
    assert rejected.status == "incomplete" and rejected.context_overflow is True


@pytest.mark.asyncio
async def test_a_call_whose_attempts_all_fail_is_still_recorded():
    llm, observations, _ = _observed_llm(["timeout"] * 10, timeout=0.01)
    with pytest.raises(TimeoutError):
        await llm.call("prompt", message_history=[])
    [call] = observations.trajectory.model_calls
    assert call.status == "timeout" and len(call.attempts) == 10
    assert all(a.status == "timeout" for a in call.attempts) and call.started_at <= call.completed_at


class _FakeSession:
    def __init__(self, timeout_on=None):
        self.keys = []
        self.timeout_on = timeout_on

    async def send_keys(self, keys, block=False, min_timeout_sec=0.0):
        self.keys.append((keys, block, min_timeout_sec))
        if keys == self.timeout_on:
            raise TimeoutError

    async def get_incremental_output(self):
        return f"screen after {len(self.keys)} commands"


def _fake_terminus(observations):
    return SimpleNamespace(
        _timeout_template="TIMEOUT {timeout_sec} {command} {terminal_state}",
        _limit_output_length=lambda text, max_bytes=10000: text,
        _completed_command_batches=0,
        _times_spent=[],
        _n_episodes=3,
        _nemo_gym_llm=SimpleNamespace(observations=observations),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_on", [None, "sleep 9"])
async def test_execute_commands_matches_harbor_and_records_each_command(timeout_on):
    from harbor.agents.terminus_2 import Terminus2
    from harbor.agents.terminus_2.terminus_2 import Command

    commands = [
        Command(keystrokes="ls\n", duration_sec=0.0),
        Command(keystrokes="sleep 9", duration_sec=0.5),
        Command(keystrokes="pwd\n", duration_sec=0.0),
    ]
    expected = await Terminus2._execute_commands(_fake_terminus(None), commands, _FakeSession(timeout_on))

    observations = TerminusObservations("inv", "t", "t-0", ModelServerRef(type="responses_api_models", name="m"))
    observations.decision_parsed_at = 123.0
    fake = _fake_terminus(observations)
    actual = await NeMoGymTerminus2._execute_commands(fake, commands, _FakeSession(timeout_on))
    assert actual == expected  # same return value and side effects as harbor's own method
    assert fake._completed_command_batches == 1 and len(fake._times_spent) == 1

    tools = observations.trajectory.tool_calls
    assert len(tools) == (2 if timeout_on else 3)
    assert len({t.batch_id for t in tools}) == 1 and [t.batch_index for t in tools] == list(range(1, len(tools) + 1))
    assert all(
        t.requested_at == 123.0 and t.timing_source == "harness" and t.clock_domain.startswith("node:") for t in tools
    )
    assert tools[0].operation == "ls" and tools[1].requested_wait_s == 0.5
    assert all(t.duration_ms is not None and t.started_at <= t.completed_at for t in tools)
    assert tools[0].completed_at <= tools[1].started_at  # per command, not one batch interval copied onto each
    assert all(t.output_chars == len(expected[1]) for t in tools)
    assert [t.status for t in tools] == (["completed", "timeout"] if timeout_on else ["completed"] * 3)
    assert all(t.response_received_at is None for t in tools)
    observations.mark_tool_results_observed()
    assert all(t.response_received_at is not None for t in tools)


def _bare_terminus(llm):
    """A NeMoGymTerminus2 without Harbor's constructor: enough for the wrapped hooks, whose super() calls the
    tests replace on the Harbor base class."""
    terminus = NeMoGymTerminus2.__new__(NeMoGymTerminus2)
    terminus._nemo_gym_llm = llm
    terminus._completed_command_batches = 0
    terminus._is_check_proactive_summarization = False
    terminus._num_proactive_compactions = 0
    return terminus


@pytest.mark.asyncio
async def test_decision_results_and_subagent_purposes_land_on_the_calls(monkeypatch):
    from harbor.agents.terminus_2 import Terminus2

    llm, observations, _ = _observed_llm(["ok", "ok", "ok"])
    terminus = _bare_terminus(llm)

    async def fake_interaction(self, *_args, **_kwargs):
        response = await llm.call("decide", message_history=[])
        observations.decision_response = response
        return ([object(), object()], False, "parsed")

    monkeypatch.setattr(Terminus2, "_handle_llm_interaction", fake_interaction)
    result = await terminus._handle_llm_interaction()
    assert result[1] is False
    [decision] = observations.trajectory.model_calls
    assert decision.parsed_tool_calls == 2 and decision.task_complete is False
    assert decision.processed_at is not None and decision.processed_at >= decision.completed_at
    assert [t.model_calls[0].model_call_id for t in observations.trajectory.turns] == [decision.model_call_id]

    # Harbor's compaction sub-agents are told apart by the filename suffix it passes to _run_subagent.
    async def fake_subagent(self, *_args, **_kwargs):
        await llm.call("questions", message_history=[])

    monkeypatch.setattr(Terminus2, "_run_subagent", fake_subagent)
    await terminus._run_subagent(filename_suffix="questions")
    await llm.call("short summary")  # no message_history: Harbor's short-summary fallback
    purposes = [c.model_call_purpose for c in observations.trajectory.model_calls]
    assert purposes == ["agent_step", "compaction_questions", "compaction_fallback"]


@pytest.mark.asyncio
async def test_overflow_compaction_points_at_the_rejected_call_and_is_timed(monkeypatch):
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import ContextLengthExceededError

    llm, observations, _ = _observed_llm(["empty", "ok", "ok"])
    terminus = _bare_terminus(llm)
    with pytest.raises(ContextLengthExceededError):
        await llm.call("too long", message_history=[])

    async def fake_summarize(self, *_args, **_kwargs):
        await llm.call("summarise", message_history=[])

    monkeypatch.setattr(Terminus2, "_summarize", fake_summarize)
    await terminus._summarize()
    await llm.call("handoff", message_history=[])
    [compaction] = observations.compactions
    rejected, summary, handoff = observations.trajectory.model_calls
    assert compaction.trigger == "overflow" and compaction.before_model_call.model_call_id == rejected.model_call_id
    assert compaction.duration_ms is not None and compaction.clock_domain.startswith("node:")
    assert compaction.tokens_before == 13 and compaction.tokens_after == 10  # last usage total; next step's input
    assert summary.model_call_purpose == "compaction_summary" and handoff.model_call_purpose == "agent_step"
