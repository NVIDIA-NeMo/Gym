# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions: Cline in the task sandbox, behind the seed, activation and close contract."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from omegaconf import OmegaConf

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_SESSION_COOKIE_KEY,
    AgentSeedSessionRequest,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import AgentInvocation, SandboxObservation, ToolCallObservation
from responses_api_agents.cline_agent import app
from responses_api_agents.cline_agent import sandbox as adapter
from responses_api_agents.cline_agent.observability import build_cline_observations


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")


def seed(*, borrowed=True, workdir="/app", tool_accesses=(), descriptor=None):
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        tool_accesses=list(tool_accesses),
        sandbox_access={
            "connection": {"provider_config_ref": "sandbox", "descriptor": descriptor or {"id": "borrowed"}},
            "workdir": workdir,
        }
        if borrowed
        else None,
    )


def agent(**kwargs):
    kwargs.setdefault("model_server", MODEL)
    kwargs.setdefault("model", "policy-model")
    kwargs.setdefault("cline_version", "3.0.55")
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = OmegaConf.create(
        {
            "sandbox": {"local": {}},
            "policy_model": {"responses_api_models": {"vllm_model": {"host": "10.0.0.5", "port": 9000}}},
        }
    )
    instance = app.ClineAgent(
        config=app.ClineAgentConfig(name="cline", host="127.0.0.1", port=8080, entrypoint="app.py", **kwargs),
        server_client=server_client,
    )
    server_client._build_server_base_url = lambda config: f"http://{config['host']}:{config['port']}"
    return instance


def result(stdout="", *, code=0):
    return SimpleNamespace(stdout=stdout, stderr="diagnostic" if code else "", return_code=code, error_type=None)


def request(session_id="session", rollout="rollout"):
    return Request(
        {"type": "http", "session": {AGENT_SESSION_COOKIE_KEY: session_id}, "path_params": {"rollout_id": rollout}}
    )


def stream(*records) -> str:
    stamped = [{"ts": f"2026-11-03T00:00:{index:02d}.000Z", **record} for index, record in enumerate(records)]
    return "\n".join(json.dumps(record) for record in stamped) + "\n"


def tool(kind, call_id, **fields):
    return {"type": "agent_event", "event": {"type": kind, "contentType": "tool", "toolCallId": call_id, **fields}}


def text(value):
    return {"type": "agent_event", "event": {"type": "content_end", "contentType": "text", "text": value}}


def error_event(name, message, error_class="unknown"):
    return {
        "type": "agent_event",
        "event": {"type": "error", "error": {"name": name, "message": message}, "errorClass": error_class},
    }


def run_result(finish, **fields):
    return {"type": "run_result", "finishReason": finish, "durationMs": 1200, **fields}


COMPLETED = stream(
    tool("content_start", "call-1", toolName="run_commands", input={"commands": ["ls"]}),
    tool("content_end", "call-1", toolName="run_commands", output=[{"query": "ls", "result": "a", "success": True}]),
    text("Done."),
    run_result("completed", aggregateUsage={"inputTokens": 100, "outputTokens": 7}),
)


class FakeSession:
    """The SandboxSession surface the adapter uses, with a scripted execute()."""

    def __init__(self, *, artifacts=None, cleanup=None, raises=None):
        self.session_dir = "/tmp/nemo-gym-cline-sessions/id"
        self.workdir = "/app"
        self.sandbox = SimpleNamespace(sandbox_id="sandbox-1")
        self.closing = False
        self.expired = False
        self.artifacts = None
        self.cleanup = None
        self._result = (artifacts, cleanup, raises)
        self.staged = None
        self.close = AsyncMock()

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        self.timeout = timeout
        self.staged = await stage_activation()
        self.artifacts, self.cleanup, raises = self._result
        if raises is not None:
            raise raises
        return self.artifacts


def state_with(session, *, borrowed=True):
    state = adapter.ClineSandboxSession(request=seed(borrowed=borrowed), session=session, runtime="/runtime")
    state.upload_json = AsyncMock()
    state.stage_activation = AsyncMock(return_value="command")
    return state


def cleanup(*, code=0, timed_out=False):
    return {"cleanup_confirmed": True, "return_code": code, "timed_out": timed_out, "error": None}


# Seed


@pytest.mark.parametrize(
    ("config", "unsupported", "reason"),
    [
        (
            {},
            {"tool_accesses": [{"kind": "direct_http", "name": "tools", "base_url": "http://r", "required": True}]},
            "Cline sandbox sessions use Cline's built-in tools; required tool grants are unsupported: tools",
        ),
        ({"cline_version": None}, {}, "Cline sandbox sessions require an exact cline_version, for example 3.0.55"),
        ({"cline_version": "3.0"}, {}, "Cline sandbox sessions require an exact cline_version, for example 3.0.55"),
        ({"cline_version": "9.9.9"}, {}, "Cline 9.9.9 has no pinned runtime checksums; supported: 3.0.55"),
        ({"model_server": None}, {}, "Cline sandbox sessions require a Gym model_server and its model name"),
        (
            {"env": {"A": "b"}, "extra_args": ["--verbose"], "command": "npx cline"},
            {},
            "Cline sandbox sessions do not support these host settings: env, extra_args, command",
        ),
        ({}, {"borrowed": False}, "Cline requires sandbox_access or a configured sandbox_provider"),
        ({}, {"workdir": "/tmp"}, "Cline sandbox workdir must be absolute and separate from its session storage"),
        (
            {},
            {"workdir": "/tmp/nemo-gym-cline-sessions/x"},
            "Cline sandbox workdir must be absolute and separate from its session storage",
        ),
    ],
)
def test_unsupported_seed_is_refused_with_a_final_422(monkeypatch, config, unsupported, reason):
    connect = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    instance = agent(**config)
    assert check_agent_seed_rejected(instance.setup_webserver(), seed(**unsupported)) == reason
    assert not instance._session_records
    connect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_installs_pinned_runtime_inside_the_sandbox(monkeypatch, borrowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": "/app"})

    state = await instance._seed_agent_session_state(seed(borrowed=borrowed))

    commands = [call.args[0] for call in remote.exec.await_args_list]
    digests = adapter.CLINE_RUNTIME_SHA512["3.0.55"]
    assert state.runtime == "/tmp/nemo-gym-cline-runtime-3.0.55"
    assert state.session.session_dir.startswith("/tmp/nemo-gym-cline-sessions/")
    assert state.session.owns_sandbox is not borrowed
    install = commands[-1]
    assert install.endswith(f"{state.runtime} 3.0.55 {digests['x64']} {digests['arm64']}")
    assert install.startswith(f"bash {state.session.session_dir}/install_cline_runtime.sh")
    uploaded = [call.args[1] for call in remote.upload.await_args_list]
    assert uploaded == [
        f"{state.session.session_dir}/install_cline_runtime.sh",
        f"{state.session.session_dir}/harness_runner.sh",
    ]
    assert remote.start.await_count == (0 if borrowed else 1)
    if borrowed:
        constructor.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)
        # The Resources Server prepared the borrowed task; seed only checks that its workdir exists.
        assert not any(command.startswith("mkdir -p -- /app") for command in commands)


@pytest.mark.asyncio
async def test_failed_install_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(side_effect=[result(), result("no network", code=6)]), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    close = AsyncMock()
    monkeypatch.setattr(adapter.ClineSandboxSession, "close", close)
    instance = agent()
    with pytest.raises(RuntimeError, match=r"(?s)Cline sandbox installation failed .*no network"):
        await instance._seed_agent_session_state(seed())
    close.assert_awaited_once()

    remote.exec = AsyncMock(side_effect=[result(), result(code=1)])
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(app.AgentSessionSetupError):
        await instance._seed_agent_session_state(seed())


# Activation


@pytest.mark.asyncio
async def test_activation_stages_isolated_settings_and_maps_a_completed_run():
    instance = agent(system_prompt="Configured", thinking="low")
    session = FakeSession(artifacts=adapter.ClineArtifacts(stdout=COMPLETED, stderr=""), cleanup=cleanup())
    state = state_with(session)
    state.runtime_info = adapter.HarnessProcessInfo(hostname="sandbox", pid=7)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "System"}, {"role": "user", "content": "Question"}],
        metadata={"agent_timeout_sec": "30"},
    )

    response = await instance._sandbox_response(state, instance._validate_session_request(body))

    staged = state.stage_activation.await_args.kwargs
    command = staged["command"]
    assert command[0] == "/runtime/cline"
    assert command[-2:] == ["--", "Configured\n\nSystem\n\nQuestion"]
    conversation = next(record for record in state.observations.records if isinstance(record, AgentInvocation))
    assert [(item.role, item.content) for item in conversation.conversation[:1]] == [
        ("user", "Configured\n\nSystem\n\nQuestion")
    ]
    assert "--timeout" not in command
    assert command[command.index("--config") + 1] == f"{session.session_dir}/config"
    assert command[command.index("--data-dir") + 1] == f"{session.session_dir}/data"
    assert command[command.index("-m") + 1] == "policy-model"
    assert command[command.index("--thinking") + 1] == "low"
    settings = staged["providers"]["providers"]["openai-compatible"]["settings"]
    assert settings["baseUrl"] == "http://10.0.0.5:9000/ng-rollout/rollout/v1"
    assert settings["headers"] == {"x-session-id": "rollout"}
    assert staged["global_settings"] == {"telemetryOptOut": True, "autoUpdateEnabled": False}
    env = staged["env"]
    assert env["CLINE_DATA_DIR"] == f"{session.session_dir}/data"
    assert env["CLINE_NO_AUTO_UPDATE"] == "1"
    # Tool shells keep the task's HOME and XDG directories.
    assert not {"HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"} & set(env)
    assert session.timeout == 30

    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Done."
    assert response.usage.input_tokens == 100 and response.usage.output_tokens == 7
    assert response.metadata["harness_execution"] == "sandbox"
    assert response.metadata["harness_hostname"] == "sandbox"
    records = state.observations.records
    invocation = next(record for record in records if isinstance(record, AgentInvocation))
    assert (invocation.invocation_id, invocation.status, invocation.duration_ms) == ("rollout", "completed", 1200)
    sandbox = next(record for record in records if isinstance(record, SandboxObservation))
    assert (sandbox.outcome, sandbox.sandbox_id, sandbox.exit_code) == ("completed", "sandbox-1", 0)
    tool_call = next(record for record in records if isinstance(record, ToolCallObservation))
    assert (tool_call.status, tool_call.sandbox_id) == ("completed", "sandbox-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("records", "receipt", "status", "failure_kind"),
    [
        # Cline's own error: a failed response the Environment Server classifies.
        (
            [error_event("Error", "Cannot connect to API: Unable to connect"), run_result("error")],
            cleanup(code=1),
            "failed",
            "model_transport_error",
        ),
        ([error_event("Error", "stub boom"), run_result("error")], cleanup(code=1), "failed", "unknown"),
        # A context overflow and an abort leave gradable partial work.
        (
            [error_event("ContextWindowOverflowError", "too large", "context_window_exceeded"), run_result("error")],
            cleanup(code=1),
            "incomplete",
            None,
        ),
        (
            [
                error_event("Error", "Model reached the maximum output token limit before completing the turn"),
                run_result("error"),
            ],
            cleanup(code=1),
            "incomplete",
            None,
        ),
        ([text("partial"), run_result("aborted")], cleanup(code=1), "incomplete", None),
        # The supervisor stopped Cline at its deadline.
        ([text("partial")], cleanup(code=None, timed_out=True), "incomplete", None),
    ],
)
async def test_activation_status_follows_clines_exit_rule(records, receipt, status, failure_kind):
    instance = agent()
    session = FakeSession(artifacts=adapter.ClineArtifacts(stdout=stream(*records), stderr=""), cleanup=receipt)
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == status
    outcome = (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY)
    assert (json.loads(outcome)["failure_kind"] if outcome else None) == failure_kind
    invocation = next(record for record in state.observations.records if isinstance(record, AgentInvocation))
    assert invocation.status == status


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 125, 127])
async def test_run_without_a_result_fails_the_activation(code):
    instance = agent()
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout="", stderr="cline: crashed"), cleanup=cleanup(code=code)
    )
    state = state_with(session)
    with pytest.raises(RuntimeError, match="Cline produced no run result .*crashed"):
        await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert state.observations is not None


@pytest.mark.asyncio
async def test_interrupted_activation_keeps_observations_and_reraises():
    instance = agent()
    session = FakeSession(raises=RuntimeError("provider lost the sandbox"))
    state = state_with(session)
    with pytest.raises(RuntimeError, match="provider lost the sandbox"):
        await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert "agent_artifact_unavailable" in [gap.code for gap in state.observations.gaps]


@pytest.mark.asyncio
async def test_provider_timeout_after_a_confirmed_deadline_is_still_a_timeout():
    instance = agent()
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(text("partial")), stderr=""),
        cleanup=cleanup(code=None, timed_out=True),
        raises=TimeoutError("exec wait expired"),
    )
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == "incomplete"


@pytest.mark.asyncio
async def test_retry_shares_one_activation_and_rejects_another():
    instance = agent()
    state = state_with(SimpleNamespace(closing=False))
    async with instance._locked_agent_session("session") as record:
        record.state = state
    response = MagicMock()
    response.model_copy.return_value = response
    execute = AsyncMock(return_value=response)
    instance._sandbox_response = execute
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    assert await instance.responses(request(), body) is response
    assert await instance.responses(request(), body) is response
    assert execute.await_count == 1
    for changed, rollout in ((NeMoGymResponseCreateParamsNonStreaming(input="Changed"), "rollout"), (body, "other")):
        with pytest.raises(HTTPException) as error:
            await instance.responses(request(rollout=rollout), changed)
        assert error.value.status_code == 409
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_unknown_session_never_falls_back_to_the_host():
    instance = agent()
    instance._run_cline = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await instance.responses(request("unknown"), NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert error.value.status_code == 409
    instance._run_cline.assert_not_awaited()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"input": "Q", "temperature": 0.5}, "do not support request field temperature"),
        ({"input": "Q", "max_output_tokens": 10}, "do not support request field max_output_tokens"),
        ({"input": "Q", "metadata": {"other": "x"}}, "do not support request field metadata"),
        ({"input": "Q", "model": "another-model"}, "selected by the agent configuration"),
        (
            {"input": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            "one text user prompt",
        ),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:x", "detail": "auto"}]}
                ]
            },
            "only supports text input",
        ),
    ],
)
def test_session_request_rejects_controls_cline_cannot_apply(body, message):
    with pytest.raises(HTTPException, match=message) as error:
        agent()._validate_session_request(NeMoGymResponseCreateParamsNonStreaming(**body))
    assert error.value.status_code == 422


def test_session_request_accepts_the_policy_model_and_its_server_name():
    instance = agent()
    for model in ("policy-model", "policy_model", None):
        instance._validate_session_request(NeMoGymResponseCreateParamsNonStreaming(input="Q", model=model))


def test_prompt_beyond_the_argv_limit_is_rejected():
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="x " * (app.MAX_PROMPT_BYTES // 2 + 1))
    with pytest.raises(HTTPException, match="argv limit") as error:
        instance._session_prompt(body)
    assert error.value.status_code == 422


# Close


@pytest.mark.asyncio
async def test_close_without_activation_returns_a_gap_and_keeps_the_borrowed_sandbox():
    instance = agent()
    session = FakeSession()
    state = state_with(session)
    closed = await instance._close_agent_session_state(state)
    session.close.assert_awaited_once()
    assert [gap.code for gap in closed.agent_observations.gaps] == ["observation_capture_failed"]
    assert closed.resources_cookies is None


@pytest.mark.asyncio
async def test_close_returns_observations_of_a_finished_activation():
    instance = agent()
    session = FakeSession(artifacts=adapter.ClineArtifacts(stdout=COMPLETED, stderr=""), cleanup=cleanup())
    state = state_with(session)
    state.task = asyncio.create_task(
        instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    )
    await state.task
    closed = await instance._close_agent_session_state(state)
    assert closed.agent_observations is state.observations
    assert instance._rollout_sandbox_session(state) is session


@pytest.mark.asyncio
async def test_close_cancels_an_activation_still_staging():
    session = FakeSession()
    state = state_with(session)
    started = asyncio.Event()

    async def stuck():
        started.set()
        await asyncio.sleep(60)

    state.task = asyncio.create_task(stuck())
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(state.close(1), 5)
    assert state.task.cancelled()


# Host path and outcome classification


@pytest.mark.asyncio
async def test_sessionless_calls_keep_the_host_cli_path(monkeypatch):
    instance = agent()
    instance._run_cline = AsyncMock(return_value=([], {}, "policy-model"))
    plain = SimpleNamespace(path_params={}, url=SimpleNamespace(path=""))  # what harness_agent's runner passes
    response = await instance.responses(plain, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    instance._run_cline.assert_awaited_once_with("Question", None, None)
    assert response.output[0].type == "message"


@pytest.mark.asyncio
async def test_run_requires_resources_and_refuses_sessions():
    with pytest.raises(HTTPException) as error:
        await agent().run(request(), MagicMock())
    assert error.value.status_code == 409
    plain = Request({"type": "http", "session": {}, "path_params": {}})
    with pytest.raises(HTTPException) as error:
        await agent().run(plain, MagicMock())
    assert error.value.status_code == 422
    agent(resources_server=ResourcesServerRef(type="resources_servers", name="math"))  # the host config stays valid


@pytest.mark.asyncio
async def test_host_install_is_lazy_shared_and_retryable(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def install(version):
        entered.set()
        release.wait(timeout=5)

    installer = MagicMock(side_effect=install)
    monkeypatch.setattr(app, "ensure_cline", installer)
    instance = agent()
    installer.assert_not_called()
    waiters = [asyncio.create_task(instance._ensure_local_cline()) for _ in range(2)]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiters[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        assert not instance._local_setup_task.cancelled()
    finally:
        release.set()
        await asyncio.gather(*waiters, return_exceptions=True)
    assert installer.call_count == 1
    instance = agent()
    installer.side_effect = RuntimeError("install failed")
    with pytest.raises(RuntimeError, match="install failed"):
        await instance._ensure_local_cline()
    installer.side_effect = None
    await instance._ensure_local_cline()
    assert installer.call_count == 3


@pytest.mark.parametrize(
    ("terminal", "kind"),
    [
        ({"error": {"name": "ContextWindowOverflowError", "message": "x"}}, "model_context_limit"),
        ({"error": {"name": "Error", "message": "x"}, "errorClass": "context_window_exceeded"}, "model_context_limit"),
        ({"error": {"name": "Error", "message": "The operation timed out."}}, "model_transport_error"),
        ({"error": {"name": "Error", "message": "Cannot connect to API: refused"}}, "model_transport_error"),
        ({"error": {"name": "Error", "message": "Rate limit exceeded"}}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_classify_outcome_reads_clines_terminal_error(terminal, kind):
    assert agent().classify_outcome(terminal).failure_kind == kind


# Observations


def test_observations_time_tools_and_mark_failed_and_unfinished_calls():
    stdout = (
        stream(
            tool("content_start", "ok", toolName="run_commands"),
            tool("content_end", "ok", toolName="run_commands", output=[{"success": True}]),
            tool("content_start", "bad", toolName="run_commands"),
            tool("content_end", "bad", toolName="run_commands", output=[{"success": True}, {"success": False}]),
            tool("content_start", "err", toolName="editor"),
            tool("content_end", "err", toolName="editor", error="no such file"),
            tool("content_start", "plain", toolName="read_files"),
            tool("content_end", "plain", toolName="read_files", output="contents"),
            tool("content_start", "cut", toolName="run_commands"),
        )
        + "not json\n"
    )
    bundle = build_cline_observations(
        stdout, "rollout", status="incomplete", error_type=None, conversation=[], unfinished_tool_status="timeout"
    )
    tools = {record.tool_call_id: record for record in bundle.records if isinstance(record, ToolCallObservation)}
    assert {call: record.status for call, record in tools.items()} == {
        "ok": "completed",
        "bad": "failed",
        "err": "failed",
        "plain": "unknown",
        "cut": "timeout",
    }
    # The cut-off call gets an output that says it returned none, in the observed conversation only.
    invocation = next(record for record in bundle.records if isinstance(record, AgentInvocation))
    assert [(item.call_id, item.output) for item in invocation.conversation] == [
        ("cut", "[no output: Cline stopped before this tool call returned (timeout)]")
    ]
    assert tools["ok"].duration_ms == 1000 and tools["ok"].timing_source == "harness"
    gaps = [(gap.code, gap.detail) for gap in bundle.gaps]
    assert ("tool_outcome_unavailable", "plain") in gaps
    assert ("tool_timing_unavailable", "cut") in gaps
    assert ("subagent_hierarchy_unavailable", None) in gaps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("records", "expected", "gap"),
    [
        # openai-compatible counts cache reads inside inputTokens; they are a detail, not an addition.
        (
            [
                run_result(
                    "completed", aggregateUsage={"inputTokens": 37432, "cacheReadTokens": 13152, "outputTokens": 9}
                )
            ],
            (37432, 13152, 9, None),
            "reasoning_token_usage_unavailable",
        ),
        # A run the deadline stopped has only running totals, which can lag answered calls: usage stays unknown.
        (
            [
                {
                    "type": "agent_event",
                    "event": {
                        "type": "usage",
                        "totalInputTokens": 50,
                        "totalCacheReadTokens": 0,
                        "totalOutputTokens": 4,
                    },
                }
            ],
            None,
            "model_call_usage_unavailable",
        ),
        ([text("no usage")], None, "model_call_usage_unavailable"),
    ],
)
async def test_session_usage_counts_cache_reads_once(records, expected, gap):
    instance = agent()
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(*records), stderr=""), cleanup=cleanup(timed_out=True)
    )
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    usage = response.usage
    observed = (
        None
        if usage is None
        else (
            usage.input_tokens,
            usage.input_tokens_details.cached_tokens,
            usage.output_tokens,
            usage.output_tokens_details.reasoning_tokens,
        )
    )
    assert observed == expected
    assert gap in [item.code for item in state.observations.gaps]


@pytest.mark.asyncio
async def test_deadline_marks_the_running_tool_call_as_timed_out():
    instance = agent()
    records = [tool("content_start", "call-1", toolName="run_commands", input={"commands": ["sleep 999"]})]
    session = FakeSession(
        artifacts=adapter.ClineArtifacts(stdout=stream(*records), stderr=""),
        cleanup=cleanup(code=None, timed_out=True),
    )
    state = state_with(session)
    response = await instance._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert response.status == "incomplete"
    # The response keeps Cline's own items; only the observed conversation records the missing output.
    assert [item.type for item in response.output] == ["function_call"]
    tool_call = next(record for record in state.observations.records if isinstance(record, ToolCallObservation))
    assert (tool_call.tool_call_id, tool_call.status) == ("call-1", "timeout")
