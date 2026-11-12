# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    HARNESS_RUNNER_SCRIPT,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    OutcomeAction,
    SandboxSession,
    agent_outcome_policy,
    check_agent_seed_rejected,
    shell_assignments,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NgFailureEvent,
    RequestFailureLog,
)
from nemo_gym.sandbox import AsyncSandbox
from responses_api_agents.droid_agent import app, runtime
from responses_api_agents.droid_agent import sandbox as adapter


EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "droid-session"},
    {"type": "message", "role": "user", "id": "u1", "text": "Question"},
    {
        "type": "tool_call",
        "id": "c1",
        "toolId": "Execute",
        "toolName": "Execute",
        "parameters": {"command": "ls"},
        "timestamp": 1000,
    },
    {
        "type": "tool_result",
        "id": "c1",
        "toolId": "Execute",
        "isError": False,
        "value": "a\n[Process exited with code 0]",
        "timestamp": 1250,
    },
    {"type": "message", "role": "assistant", "id": "a1", "text": "Answer"},
    {
        "type": "completion",
        "durationMs": 900,
        "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 4, "thinking_tokens": 1},
    },
]


def seed(*, borrowed=True, workdir="/app", tool_accesses=()):
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        tool_accesses=list(tool_accesses),
        sandbox_access={
            "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "borrowed"}},
            "workdir": workdir,
        }
        if borrowed
        else None,
    )


def agent(monkeypatch, *, model_url="http://10.0.0.5:9000", **kwargs):
    import nemo_gym.agents.base_responses_api_agent as base_agent

    monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _config, _name: {"host": "h", "port": 1})
    client = MagicMock(spec=ServerClient, global_config_dict={"sandbox": {"opensandbox": {}}})
    client._build_server_base_url = lambda _config: model_url
    config = {"model_server": {"type": "responses_api_models", "name": "policy_model"}, "model": "nim/model"} | kwargs
    return app.DroidAgent(
        config=app.DroidAgentConfig(name="droid", host="127.0.0.1", port=8080, entrypoint="app.py", **config),
        server_client=client,
    )


def result(code=0):
    return SimpleNamespace(stdout="", stderr="diagnostic" if code else "", return_code=code, error_type=None)


def state(*, cleanup=None, artifacts=None, request=None):
    return adapter.DroidSandboxSession(
        request=request or seed(),
        runtime="/tmp/nemo-gym-droid-runtime-1.1.0",
        session=SimpleNamespace(
            session_dir="/tmp/nemo-gym-droid-sessions/s",
            workdir="/app",
            cleanup=cleanup or {"timed_out": False, "return_code": 0, "cleanup_confirmed": True},
            artifacts=artifacts,
            closing=False,
            expired=False,
            sandbox=SimpleNamespace(sandbox_id=None),
        ),
    )


def body(**kwargs):
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Question"}], **kwargs
    )


def stdout(events):
    return "\n".join(map(json.dumps, events)) + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_installs_the_pinned_runtime_inside_the_sandbox(monkeypatch, borrowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    install = AsyncMock()
    monkeypatch.setattr(app, "install_sandbox_runtime", install)
    instance = agent(monkeypatch, sandbox_provider="sandbox", sandbox_config={"workdir": "/app"})
    session = await instance._seed_agent_session_state(seed(borrowed=borrowed))
    assert install.await_args.kwargs["version"] == "1.1.0"
    assert install.await_args.kwargs["runtime"] == session.runtime == "/tmp/nemo-gym-droid-runtime-1.1.0"
    directory = session.session.session_dir
    assert directory.startswith("/tmp/nemo-gym-droid-sessions/") and session.factory_home == directory + "/home"
    assert [call.args[1] for call in remote.upload.await_args_list] == [
        f"{directory}/harness_runner.sh",
        f"{directory}/install_droid_runtime.sh",
    ]
    assert session.session.owns_sandbox is not borrowed
    assert remote.start.await_count == (0 if borrowed else 1)
    if borrowed:
        constructor.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)


@pytest.mark.parametrize(
    "unsupported,reason",
    [
        (
            {
                "tool_accesses": [
                    {"kind": "mcp", "name": "tools", "required": True, "connection": {"url": "http://r/mcp"}}
                ]
            },
            "required Resources tool grants",
        ),
        ({"sandbox_access": None}, "sandbox_access or a configured sandbox_provider"),
        (
            {"sandbox_access": {"connection": {"provider_config_ref": "sandbox", "descriptor": {}}, "workdir": "/"}},
            "workdir",
        ),
    ],
)
def test_unservable_seeds_are_refused_as_final(monkeypatch, unsupported, reason):
    request = seed().model_dump() | unsupported
    instance = agent(monkeypatch)
    assert reason in check_agent_seed_rejected(instance.setup_webserver(), AgentSeedSessionRequest(**request))
    assert not instance._session_records


def test_loopback_model_server_is_refused_for_a_remote_sandbox(monkeypatch):
    instance = agent(monkeypatch, model_url="http://127.0.0.1:9000")
    assert "loopback" in check_agent_seed_rejected(instance.setup_webserver(), seed())


@pytest.mark.parametrize("borrowed", [True, False])
@pytest.mark.parametrize("constructor_fails", [False, True])
def test_provider_configuration_errors_are_refused_as_final(monkeypatch, borrowed, constructor_fails):
    constructor = MagicMock()
    constructor.connect = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    instance = agent(monkeypatch, sandbox_provider="sandbox")
    if constructor_fails:

        def create_provider(config):
            raise ValueError("exec.concurrency must be at least 1")

        monkeypatch.setattr(app, "create_provider", create_provider)
        reason = "exec.concurrency"
    else:
        instance.server_client.global_config_dict = {"sandbox": {"unknown_provider": {}}}
        reason = "unknown_provider"
    assert reason in check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=borrowed))
    constructor.assert_not_called()
    constructor.connect.assert_not_awaited()
    assert not instance._session_records


@pytest.mark.asyncio
async def test_connection_failure_stays_retryable_and_closes_the_provider(monkeypatch):
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(side_effect=ConnectionError("refused")))
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    with pytest.raises(ConnectionError):
        await agent(monkeypatch)._seed_agent_session_state(seed())
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["relative", "/app/../tmp", "/tmp", "/tmp/nemo-gym-droid-sessions/x"])
async def test_workdir_overlapping_adapter_storage_is_refused_before_connect(monkeypatch, workdir):
    connect = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    with pytest.raises(HTTPException, match="separate from runtime/session storage"):
        await agent(monkeypatch)._seed_agent_session_state(seed(workdir=workdir))
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_seed_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result(code=1)), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    close = AsyncMock()
    monkeypatch.setattr(adapter.DroidSandboxSession, "close", close)
    with pytest.raises(RuntimeError, match="diagnostic"):
        await agent(monkeypatch)._seed_agent_session_state(seed())
    close.assert_awaited_once()
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(AgentSessionSetupError) as error:
        await agent(monkeypatch)._seed_agent_session_state(seed())
    assert error.value.state.request == seed()


@pytest.mark.asyncio
async def test_activation_routes_droid_through_the_rollout_model_route(monkeypatch):
    instance = agent(monkeypatch, system_prompt="Configured.", context_window=32768)
    session = state()
    session.session.sandbox = SimpleNamespace(sandbox_id="box-1")
    execute = AsyncMock(return_value=adapter.DroidArtifacts(stdout(EVENTS)))
    monkeypatch.setattr(adapter.DroidSandboxSession, "execute", execute)
    response = await instance._sandbox_response(session, body())

    payload = execute.await_args.args[0]
    [model] = payload["settings"]["customModels"]
    capture_key = seed().episode_id.capture_key
    assert model["baseUrl"] == f"http://10.0.0.5:9000/ng-rollout/{capture_key}/v1"
    assert model["extraHeaders"] == {"x-session-id": capture_key}
    assert model["provider"] == "generic-chat-completion-api" and model["apiKey"] == "gym"
    assert (model["maxOutputTokens"], model["maxContextLimit"]) == (8192, 32768)
    assert payload["settings"]["cloudSessionSync"] is False
    assert payload["prompt"] == "Question" and payload["system"] == "Configured.\n\nBe brief."
    assert payload["env"] == {
        "FACTORY_HOME_OVERRIDE": "/tmp/nemo-gym-droid-sessions/s/home",
        "FACTORY_AIRGAP_ENABLED": "1",
    }
    assert payload["command"][:2] == ["/tmp/nemo-gym-droid-runtime-1.1.0/droid", "exec"]
    assert payload["command"][-2:] == ["--file", "/tmp/nemo-gym-droid-sessions/s/prompt.txt"]
    assert "--skip-permissions-unsafe" in payload["command"]
    assert payload["command"][payload["command"].index("--model") + 1] == "custom:nim/model"
    assert payload["command"][payload["command"].index("--cwd") + 1] == "/app"
    assert "--append-system-prompt-file" in payload["command"]

    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    # Droid counts cache reads apart from input_tokens; the response's input total includes them.
    assert response.usage.input_tokens == 14 and response.usage.total_tokens == 16
    assert response.usage.input_tokens_details.cached_tokens == 4
    # Droid's thinking_tokens is not a measurement, so the count stays unknown, also once summed with a known one.
    assert response.usage.output_tokens_details.reasoning_tokens is None
    known = response.usage.model_copy(
        update={"output_tokens_details": NeMoGymResponseOutputTokensDetails(reasoning_tokens=3)}
    )
    assert NeMoGymResponseUsage.sum_from_list([response.usage, known]).output_tokens_details.reasoning_tokens is None
    assert "reasoning_token_usage_unavailable" in {gap.code for gap in session.observations.gaps}
    assert response.metadata == {"harness_execution": "sandbox", "agent_timeout_sec": "1800.0"}
    assert response.error is None

    invocation, tool, sandbox_record = session.observations.records
    assert invocation.invocation_id == capture_key and invocation.harness_session_id == "droid-session"
    assert [item.role for item in invocation.conversation[:2]] == ["system", "user"]
    assert (tool.tool_call_id, tool.status, tool.duration_ms, tool.sandbox_id) == ("c1", "completed", 250, "box-1")
    assert (sandbox_record.role, sandbox_record.outcome, sandbox_record.exit_code) == ("agent", "completed", 0)
    assert sandbox_record.sandbox_id == "box-1"
    assert "model_call_ownership_unavailable" not in {gap.code for gap in session.observations.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events,cleanup,status,reason",
    [
        (EVENTS[:-1], {"timed_out": True, "return_code": 143}, "incomplete", "timeout"),
        # A cut or lost stream leaves nothing to grade, so it fails instead of being verified as the model's zero.
        (EVENTS[:-1], {"timed_out": False, "return_code": 0}, "failed", "result_missing"),
        (EVENTS, {"timed_out": False, "return_code": 2}, "failed", "process_exit_2"),
    ],
)
async def test_outcome_follows_droid_exit_and_completion(monkeypatch, events, cleanup, status, reason):
    instance = agent(monkeypatch, timeout=1200)
    session = state(cleanup=cleanup | {"cleanup_confirmed": True})
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    response = await instance._sandbox_response(session, body(metadata={"agent_timeout_sec": "300"}))
    assert response.status == status
    invocation = session.observations.records[0]
    assert invocation.error_type == reason
    # The deadline that applied, and whether it ended the run, tell an expired deadline from a missing result.
    assert response.metadata["agent_timeout_sec"] == "300.0"
    assert invocation.stop_reason == ("timeout" if reason == "timeout" else None)
    if status == "failed":
        assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}
        # Without an error event, the environment's failure reason names what ended the run.
        assert (response.error.code, response.error.message) == ("server_error", reason)
    else:
        assert response.error is None
    if events is not EVENTS:
        # Without the completion event Droid reports no usage, so none is invented.
        assert response.usage is None
        assert "model_call_usage_unavailable" in {gap.code for gap in session.observations.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ['{"status":404,"title":"Not Found"}', "x" * 5000])
async def test_model_request_error_is_classified_for_the_environment(monkeypatch, detail):
    instance = agent(monkeypatch)
    session = state(cleanup={"timed_out": False, "return_code": 1, "cleanup_confirmed": True})
    error = f"404 {detail}"
    events = [EVENTS[0], {"type": "error", "source": "agent_loop", "message": error}]
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    response = await instance._sandbox_response(session, body())
    assert response.status == "failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": 404,
    }
    # The environment saves Droid's message as the failure reason, cut to the length it keeps.
    message = response.error.message
    if len(error) <= 2000:
        assert message == error
    else:
        assert len(message) == 2000 and message.startswith("404 xxx") and message.endswith(" [truncated]")
    assert session.observations.records[0].error_type == "agent_error"


# Droid 1.1.0's error events for a refused or reset connection, a context overflow, and rejected model requests,
# captured from runs against a scripted model server.
CONTEXT_LIMIT_ERROR = (
    "The existing summary and compaction instructions do not fit the selected model input window. "
    "Choose a compaction model with a larger context window."
)


@pytest.mark.parametrize(
    "message,kind",
    [
        ("400 invalid tool schema", ("model_request_error", 400)),
        ("413 payload too large", ("model_request_error", 413)),
        ("429 rate limited", ("model_request_error", 429)),
        ("503 unavailable", ("model_request_error", 503)),
        ("Connection error.", ("model_transport_error", None)),
        (CONTEXT_LIMIT_ERROR, ("model_context_limit", None)),
        # Only Droid's exact messages count: a tool's own wording or a number that is not a status does not.
        ("Command timed out after 60s", ("unknown", None)),
        ("Connection error. Retrying with the fallback tool", ("unknown", None)),
        ("4000 tokens left", ("unknown", None)),
        ("Error: The request was too large for the model service to accept.", ("unknown", None)),
    ],
)
def test_classifier_reads_droid_error_messages(monkeypatch, message, kind):
    outcome = agent(monkeypatch).classify_outcome({"error": message})
    assert (outcome.failure_kind, outcome.status_code) == kind


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["429 rate limited", "Connection error.", CONTEXT_LIMIT_ERROR])
async def test_terminal_error_reports_unknown_for_an_untested_release(monkeypatch, message):
    events = [*EVENTS[:4], {"type": "error", "source": "agent_loop", "message": message}]
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    session = state(cleanup={"timed_out": False, "return_code": 1, "cleanup_confirmed": True})
    response = await agent(monkeypatch, droid_version="1.2.0")._sandbox_response(session, body())
    assert response.status == "failed" and response.error.message == message
    assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}
    assert agent_outcome_policy(response) == OutcomeAction.RETRY
    invocation = session.observations.records[0]
    assert (invocation.error_type, invocation.stop_reason) == ("agent_error", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["429 rate limited", "Connection error.", CONTEXT_LIMIT_ERROR])
@pytest.mark.parametrize("source", ["tool_result", "assistant", "stderr"])
@pytest.mark.parametrize("return_code", [0, 1])
async def test_error_lookalikes_outside_terminal_events_do_not_classify_the_run(
    monkeypatch, message, source, return_code
):
    events = [event.copy() for event in EVENTS]
    stderr = ""
    if source == "tool_result":
        events[3].pop("value")
        events[3].update(isError=True, error={"message": message})
    elif source == "assistant":
        events[4]["text"] = message
    else:
        stderr = message
    monkeypatch.setattr(
        adapter.DroidSandboxSession,
        "execute",
        AsyncMock(return_value=adapter.DroidArtifacts(stdout(events), stderr=stderr)),
    )
    session = state(cleanup={"timed_out": False, "return_code": return_code, "cleanup_confirmed": True})
    response = await agent(monkeypatch)._sandbox_response(session, body())
    invocation = session.observations.records[0]
    assert invocation.stop_reason is None
    if return_code:
        assert (response.status, response.error.message, invocation.error_type) == (
            "failed",
            "process_exit_1",
            "process_exit_1",
        )
        assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}
        assert agent_outcome_policy(response) == OutcomeAction.RETRY
    else:
        assert (response.status, response.error, invocation.error_type) == ("completed", None, None)
        assert "ng_agent_outcome" not in response.metadata
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY
    if source == "tool_result":
        assert response.output[1].output == message
        assert session.observations.records[1].status == "failed"
    elif source == "assistant":
        assert response.output[2].content[0].text == message


@pytest.mark.asyncio
async def test_context_limit_keeps_partial_work_verifiable(monkeypatch):
    instance = agent(monkeypatch)
    session = state(cleanup={"timed_out": False, "return_code": 1, "cleanup_confirmed": True})
    events = [*EVENTS[:4], {"type": "error", "source": "agent_loop", "message": CONTEXT_LIMIT_ERROR}]
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    response = await instance._sandbox_response(session, body())
    assert (response.status, response.error) == ("incomplete", None)
    assert response.output
    assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "model_context_limit"}
    assert agent_outcome_policy(response) == OutcomeAction.VERIFY
    invocation = session.observations.records[0]
    assert (invocation.status, invocation.stop_reason, invocation.error_type) == (
        "incomplete",
        "context_limit",
        "context_limit",
    )
    # Droid itself exited with an error.
    assert session.observations.records[-1].outcome == "failed"


@pytest.mark.asyncio
async def test_failed_execution_keeps_observations_and_raises(monkeypatch):
    instance = agent(monkeypatch)
    session = state(artifacts=adapter.DroidArtifacts(stdout(EVENTS[:4])))
    monkeypatch.setattr(adapter.DroidSandboxSession, "execute", AsyncMock(side_effect=RuntimeError("exec lost")))
    with pytest.raises(RuntimeError, match="exec lost"):
        await instance._sandbox_response(session, body())
    invocation = session.observations.records[0]
    assert (invocation.status, invocation.error_type) == ("failed", "sandbox_error")
    assert session.observations.records[1].tool_call_id == "c1"


@pytest.mark.asyncio
async def test_sessionless_and_mismatched_activations_never_run(monkeypatch):
    instance = agent(monkeypatch)
    sessionless = SimpleNamespace(cookies={}, path_params={}, session={})
    monkeypatch.setattr(instance, "_agent_session_id_from_request", lambda request: None)
    with pytest.raises(HTTPException) as error:
        await instance.responses(sessionless, body())
    assert error.value.status_code == 409

    session = state()
    monkeypatch.setattr(instance, "_agent_session_id_from_request", lambda request: "session")
    monkeypatch.setattr(instance, "_require_agent_session", lambda session_id: session)
    other_rollout = SimpleNamespace(path_params={"rollout_id": "other"})
    with pytest.raises(HTTPException) as error:
        await instance.responses(other_rollout, body())
    assert error.value.status_code == 409 and session.task is None


@pytest.mark.asyncio
async def test_retry_joins_the_activation_and_a_changed_request_is_refused(monkeypatch):
    instance = agent(monkeypatch)
    session = state()
    monkeypatch.setattr(instance, "_agent_session_id_from_request", lambda request: "session")
    monkeypatch.setattr(instance, "_require_agent_session", lambda session_id: session)
    release = asyncio.Event()

    async def execute(self, payload, **kwargs):
        await release.wait()
        return adapter.DroidArtifacts(stdout(EVENTS))

    monkeypatch.setattr(adapter.DroidSandboxSession, "execute", execute)
    request = SimpleNamespace(path_params={"rollout_id": seed().episode_id.capture_key})
    first = asyncio.create_task(instance.responses(request, body()))
    second = asyncio.create_task(instance.responses(request, body()))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException) as error:
        await instance.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="Different"))
    assert error.value.status_code == 409
    release.set()
    one, two = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert one.id == two.id


@pytest.mark.parametrize(
    "request_fields",
    [
        {"temperature": 0.2},
        {"max_output_tokens": 10},
        {"instructions": "extra"},
        {"tools": [{"type": "function", "name": "f", "parameters": {}, "strict": False}]},
        {"metadata": {"other": "x"}},
        {"model": "another-model"},
    ],
)
def test_unsupported_request_controls_are_rejected(monkeypatch, request_fields):
    with pytest.raises(HTTPException) as error:
        agent(monkeypatch)._validate_request(body(**request_fields))
    assert error.value.status_code == 422


@pytest.mark.parametrize(
    "input",
    [
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}],
    ],
)
def test_history_is_rejected_instead_of_dropped(monkeypatch, input):
    with pytest.raises(HTTPException, match="one user message"):
        agent(monkeypatch)._validate_request(NeMoGymResponseCreateParamsNonStreaming(input=input))


def test_agent_timeout_metadata_is_accepted(monkeypatch):
    agent(monkeypatch)._validate_request(body(metadata={"agent_timeout_sec": "60"}))


@pytest.mark.asyncio
async def test_close_without_activation_reports_a_gap_and_propagates_cleanup_errors(monkeypatch):
    instance = agent(monkeypatch)
    session = state()
    session.session.close = AsyncMock(side_effect=RuntimeError("unconfirmed"))
    session.session.cleanup_confirmed = True
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await instance._close_agent_session_state(session)
    session.session.close.side_effect = None
    response = await instance._close_agent_session_state(session)
    assert response.agent_observations.gaps[0].code == "agent_activation_interrupted"
    assert response.cleanup_confirmed is True


def test_run_route_is_removed(monkeypatch):
    routes = {getattr(route, "path", None) for route in agent(monkeypatch).setup_webserver().router.routes}
    assert "/run" not in routes and "/v1/agent_sessions" in routes


def test_rollout_stop_is_supported(monkeypatch):
    session = state()
    assert agent(monkeypatch)._rollout_sandbox_session(session) is session.session


@pytest.mark.asyncio
async def test_stage_and_collect_use_session_files(monkeypatch):
    uploads = {}

    async def upload_text(sandbox, *, path, text):
        uploads[path] = text

    files = {
        "/s/events.jsonl": stdout(EVENTS),
        "/s/stderr.log": "e" * 20000,
        "/s/runtime.json": '{"hostname": "box", "pid": 7}',
    }

    async def read_text(sandbox, *, path):
        return files[path]

    monkeypatch.setattr(adapter, "upload_text", upload_text)
    monkeypatch.setattr(adapter, "read_text", read_text)
    session = adapter.DroidSandboxSession(
        request=seed(),
        runtime="/r",
        session=SimpleNamespace(session_dir="/s", workdir="/app", sandbox=object(), expired_before_launch=False),
    )
    payload = {"settings": {"customModels": []}, "prompt": "Q", "system": None, "command": ["droid"], "env": {}}
    command = await session.stage_activation(payload)
    assert command.argv == ["bash", "/s/harness_runner.sh", "/s/params.sh"]
    assert set(uploads) == {"/s/settings.json", "/s/prompt.txt", "/s/params.sh"}
    assert "/s/events.jsonl" in uploads["/s/params.sh"]
    artifacts = await session.collect_artifacts()
    assert artifacts.events == files["/s/events.jsonl"] and len(artifacts.stderr) == 16000
    assert session.runtime_info.pid == 7


@pytest.mark.skipif(shutil.which("bash") is None, reason="the harness runner needs bash")
def test_droid_keeps_the_task_home_while_its_state_moves_to_the_session(tmp_path):
    """Run the staged runner input with a fake droid: tool shells must see the task's HOME and workdir."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "droid").write_text(
        "#!/usr/bin/env bash\n"
        'mkdir -p "$FACTORY_HOME_OVERRIDE/.factory/sessions"\n'
        'printf \'{"type":"tool_result","id":"c1","value":"%s %s %s %s"}\\n\' "$HOME" "$PWD" "$FACTORY_AIRGAP_ENABLED" "$*"\n'
    )
    (runtime / "droid").chmod(0o755)
    session_dir, workdir, home = tmp_path / "session", tmp_path / "app", tmp_path / "task-home"
    for directory in (session_dir, workdir, home):
        directory.mkdir()
    params = {
        "cwd": str(workdir),
        "command": [str(runtime / "droid"), "exec", "--file", f"{session_dir}/prompt.txt"],
        "env": {"FACTORY_HOME_OVERRIDE": f"{session_dir}/home", "FACTORY_AIRGAP_ENABLED": "1"},
        "stdin": "/dev/null",
        "stdout": f"{session_dir}/events.jsonl",
        "stderr": f"{session_dir}/stderr.log",
    }
    (session_dir / "params.sh").write_text(shell_assignments(params))
    finished = subprocess.run(
        ["bash", str(HARNESS_RUNNER_SCRIPT), str(session_dir / "params.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(home)},
        timeout=30,
    )
    assert finished.returncode == 0
    [event] = map(json.loads, Path(session_dir, "events.jsonl").read_text().splitlines())
    assert event["value"] == f"{home} {workdir} 1 exec --file {session_dir}/prompt.txt"
    assert not list(home.iterdir()) and not list(workdir.iterdir())
    assert (session_dir / "home" / ".factory" / "sessions").is_dir()


@pytest.mark.asyncio
async def test_rollout_stop_before_launch_is_an_incomplete_response(monkeypatch):
    receipt = {"cleanup_confirmed": True, "return_code": None, "timed_out": True, "error": None}

    async def download(path, destination):
        # Only the supervisor's receipt exists: Droid never started, so it wrote no events.
        if path.endswith("/cleanup.json"):
            Path(destination).write_text(json.dumps(receipt))

    sandbox = AsyncMock(spec=AsyncSandbox, sandbox_id="box-1")
    sandbox.exec.return_value = result()
    sandbox.download.side_effect = download
    session = SandboxSession(
        sandbox=sandbox, session_dir="/tmp/nemo-gym-droid-sessions/s", workdir="/app", harness="Droid"
    )
    droid = adapter.DroidSandboxSession(request=seed(), runtime="/tmp/nemo-gym-droid-runtime-1.1.0", session=session)
    # A stop that reaches the session during seed expires it before the activation launches Droid.
    assert await session.expire(timeout=1, reason="budget")
    response = await agent(monkeypatch)._sandbox_response(droid, body())
    assert response.status == "incomplete" and response.output == []
    assert json.loads(response.metadata["ng_rollout_stop"]) == {"reason": "budget"}
    assert "ng_agent_outcome" not in response.metadata
    # The stop, not the deadline, ended the run.
    assert droid.observations.records[0].stop_reason is None
    await session.close(timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("expired,probe", [(True, "absent"), (True, "present"), (True, ""), (False, "absent")])
async def test_rollout_stop_before_droid_opens_its_stdout_keeps_an_empty_transcript(monkeypatch, expired, probe):
    async def read_text(sandbox, *, path):
        if path.endswith("/stderr.log"):
            return "starting"
        raise FileNotFoundError(path)

    monkeypatch.setattr(adapter, "read_text", read_text)
    checked = SimpleNamespace(stdout=probe + "\n", stderr="", return_code=0, error_type=None)
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=checked))
    session = adapter.DroidSandboxSession(
        request=seed(),
        runtime="/r",
        session=SimpleNamespace(
            session_dir="/s",
            sandbox=sandbox,
            expired=expired,
            expired_before_launch=False,
            read_output_log=AsyncMock(return_value="runner log"),
        ),
    )
    if expired and probe == "absent":
        # Launched, then stopped before the runner opened events.jsonl: nothing to collect, not an error.
        assert await session.collect_artifacts() == adapter.DroidArtifacts(events="", stderr="starting")
    else:
        # Events that may exist, or a run that was not stopped, still need a readable transcript.
        with pytest.raises(RuntimeError, match="returned no events"):
            await session.collect_artifacts()


@pytest.mark.parametrize(
    "probe,bootstrap,reason",
    [
        ("Linux\nx86_64\nmusl\navx2\n", result(), "musl"),
        ("Linux\nriscv64\ngnu\nbaseline\n", result(), "architecture"),
        (
            "",
            SimpleNamespace(
                stdout="", stderr="Droid sandbox sessions require bash: x", return_code=78, error_type=None
            ),
            "require bash",
        ),
    ],
)
def test_sandbox_droid_cannot_run_in_is_refused_as_final(monkeypatch, probe, bootstrap, reason):
    async def exec_(command, **kwargs):
        if "uname" in command:
            return SimpleNamespace(stdout=probe, stderr="", return_code=0, error_type=None)
        return bootstrap if "command -v bash" in command else result()

    remote = AsyncMock(spec=AsyncSandbox, sandbox_id="box")
    remote.exec.side_effect = exec_
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(aclose=AsyncMock()))
    download = AsyncMock()
    monkeypatch.setattr(runtime, "cached_package", download)
    instance = agent(monkeypatch)
    assert reason in check_agent_seed_rejected(instance.setup_webserver(), seed())
    download.assert_not_awaited()
    assert not instance._session_records


@pytest.mark.asyncio
async def test_activation_after_close_started_is_refused(monkeypatch):
    instance = agent(monkeypatch)
    session = state()
    session.session.closing = True
    monkeypatch.setattr(instance, "_agent_session_id_from_request", lambda request: "session")
    monkeypatch.setattr(instance, "_require_agent_session", lambda session_id: session)
    request = SimpleNamespace(path_params={"rollout_id": seed().episode_id.capture_key})
    with pytest.raises(HTTPException, match="closing") as error:
        await instance.responses(request, body())
    assert error.value.status_code == 409 and session.task is None


@pytest.mark.asyncio
async def test_close_cancels_an_activation_waiting_for_a_slot(monkeypatch):
    instance = agent(monkeypatch, concurrency=1)
    session = state()
    session.session.close = AsyncMock()
    session.session.cleanup_confirmed = True
    execute = AsyncMock()
    monkeypatch.setattr(adapter.DroidSandboxSession, "execute", execute)
    await instance.sem.acquire()  # another session's run holds the only slot
    session.task = asyncio.create_task(instance._sandbox_response(session, body()))
    await asyncio.sleep(0)
    closed = await asyncio.wait_for(instance._close_agent_session_state(session), timeout=5)
    execute.assert_not_awaited()
    assert session.task.cancelled()
    invocation = closed.agent_observations.records[0]
    assert (invocation.status, invocation.error_type) == ("incomplete", "cancelled")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,deadline", [(None, 1800), ({"agent_timeout_sec": "60"}, 60), ({"agent_timeout_sec": "99999"}, 1800)]
)
async def test_task_timeout_bounds_the_supervised_run(monkeypatch, metadata, deadline):
    execute = AsyncMock(return_value=adapter.DroidArtifacts(stdout(EVENTS)))
    monkeypatch.setattr(adapter.DroidSandboxSession, "execute", execute)
    await agent(monkeypatch)._sandbox_response(state(), body(metadata=metadata))
    assert execute.await_args.kwargs["timeout"] == deadline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,outcome",
    [
        ("429 rate limited", {"failure_kind": "model_request_error", "status_code": 429}),
        ("Connection error.", {"failure_kind": "model_transport_error"}),
    ],
)
async def test_error_event_fails_a_run_that_exits_zero(monkeypatch, message, outcome):
    events = [EVENTS[0], {"type": "error", "source": "agent_loop", "message": message}]
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    session = state()
    response = await agent(monkeypatch)._sandbox_response(session, body())
    assert response.status == "failed" and session.observations.records[0].error_type == "agent_error"
    assert response.error.message == message
    assert json.loads(response.metadata["ng_agent_outcome"]) == outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 10, "output_tokens": 2},
        {"input_tokens": True, "output_tokens": 2, "cache_read_input_tokens": 0},
        {"input_tokens": 10, "output_tokens": -1, "cache_read_input_tokens": 0},
        {"input_tokens": 10.0, "output_tokens": 2, "cache_read_input_tokens": 0},
        {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 0, "cache_creation_input_tokens": None},
    ],
)
async def test_incomplete_or_malformed_usage_stays_unknown(monkeypatch, usage):
    events = [*EVENTS[:-1], {"type": "completion", "durationMs": 1, "usage": usage}]
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    session = state()
    response = await agent(monkeypatch)._sandbox_response(session, body())
    assert response.status == "completed" and response.usage is None
    codes = {gap.code for gap in session.observations.gaps}
    assert "model_call_usage_unavailable" in codes and "reasoning_token_usage_unavailable" not in codes


def test_non_text_input_is_rejected(monkeypatch):
    image = {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"}
    request = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": [{"type": "input_text", "text": "Look:"}, image]}]
    )
    with pytest.raises(HTTPException, match="requires text messages") as error:
        agent(monkeypatch)._validate_request(request)
    assert error.value.status_code == 422


def test_request_without_an_agent_session_is_refused_by_the_route(monkeypatch):
    from fastapi.testclient import TestClient

    execute = AsyncMock()
    monkeypatch.setattr(adapter.DroidSandboxSession, "execute", execute)
    client = TestClient(agent(monkeypatch).setup_webserver(), raise_server_exceptions=False)
    reply = client.post("/v1/responses", json={"input": "Question"})
    assert reply.status_code == 409 and "agent session" in reply.json()["detail"]
    execute.assert_not_awaited()


@pytest.mark.skipif(shutil.which("bash") is None, reason="the harness runner needs bash")
def test_tool_process_left_holding_stdout_does_not_hold_the_run(tmp_path):
    runtime_dir, session_dir, workdir = tmp_path / "runtime", tmp_path / "session", tmp_path / "app"
    for directory in (runtime_dir, session_dir, workdir):
        directory.mkdir()
    holder = tmp_path / "holder.pid"
    (runtime_dir / "droid").write_text(
        f'#!/usr/bin/env bash\nsleep 30 & echo $! > {holder}\necho \'{{"type": "completion", "usage": {{}}}}\'\n'
    )
    (runtime_dir / "droid").chmod(0o755)
    params = {
        "cwd": str(workdir),
        "command": [str(runtime_dir / "droid"), "exec"],
        "env": {"FACTORY_HOME_OVERRIDE": f"{session_dir}/home"},
        "stdin": "/dev/null",
        "stdout": f"{session_dir}/events.jsonl",
        "stderr": f"{session_dir}/stderr.log",
    }
    (session_dir / "params.sh").write_text(shell_assignments(params))
    try:
        finished = subprocess.run(
            ["bash", str(HARNESS_RUNNER_SCRIPT), str(session_dir / "params.sh")], capture_output=True, timeout=10
        )
        assert finished.returncode == 0
        assert json.loads((session_dir / "events.jsonl").read_text())["type"] == "completion"
    finally:
        if holder.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(holder.read_text()), signal.SIGKILL)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,ttl", [({}, 3000), ({"ttl_s": None}, None)])
async def test_owned_sandbox_has_a_lifetime_and_provider_labels(monkeypatch, configured, ttl):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    instance = agent(monkeypatch, sandbox_provider="sandbox", sandbox_config={"metadata": {"team": "x"}} | configured)
    instance.server_client.global_config_dict = {"sandbox": {"opensandbox": {}, "default_metadata": {"owner": "gym"}}}
    await instance._seed_agent_session_state(seed(borrowed=False))
    spec = remote.start.await_args.args[0]
    assert spec.ttl_s == ttl and spec.workdir == "/app"
    assert spec.metadata == {
        "owner": "gym",
        "team": "x",
        "gym-agent": "droid",
        "gym-session": "session",
        "gym-rollout": seed().episode_id.capture_key,
    }


@pytest.mark.asyncio
async def test_failed_run_without_an_error_event_keeps_droid_stderr(monkeypatch, caplog):
    marker = "FATAL: custom model rejected settings"
    monkeypatch.setattr(
        adapter.DroidSandboxSession,
        "execute",
        AsyncMock(return_value=adapter.DroidArtifacts(events="", stderr=marker)),
    )
    session = state(cleanup={"timed_out": False, "return_code": 1, "cleanup_confirmed": True})
    response = await agent(monkeypatch)._sandbox_response(session, body())
    assert response.status == "failed"
    [gap] = [gap for gap in session.observations.gaps if gap.code == "agent_startup_failed"]
    assert gap.detail == marker and marker in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("typed_cleanup", [False, True])
async def test_failed_connect_keeps_original_failure_when_provider_cleanup_also_fails(monkeypatch, typed_cleanup):
    original = NgFailureEvent(
        boundary="sandbox", failure_kind="transport_timeout", failure_reason="connect timed out", stage="agent"
    )
    cleanup = (
        NgFailureEvent(
            boundary="sandbox", failure_kind="session_release_failed", failure_reason="close failed", stage="cleanup"
        )
        if typed_cleanup
        else RuntimeError("close failed")
    )
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(side_effect=original))
    provider = SimpleNamespace(aclose=AsyncMock(side_effect=cleanup))
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as raised:
            await agent(monkeypatch)._seed_agent_session_state(seed())
    assert raised.value is original
    report = log.reply_report(original)
    assert report.failure_id == original.event_id
    assert report.events[original.event_id].stage == "agent"
    releases = [event for event in report.events.values() if event.failure_kind == "session_release_failed"]
    assert len(releases) == 1 and releases[0].stage == "cleanup"
    if typed_cleanup:
        assert releases[0].event_id == cleanup.event_id
    provider.aclose.assert_awaited_once()


@pytest.mark.parametrize("workdir", [123, ["/app"], None])
def test_owned_workdir_type_is_a_final_refusal(monkeypatch, workdir):
    provider = MagicMock()
    monkeypatch.setattr(app, "create_provider", provider)
    instance = agent(monkeypatch, sandbox_provider="sandbox", sandbox_config={"workdir": workdir})
    assert "workdir" in check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=False))
    provider.assert_not_called()


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("block", [{"unknown_provider": {}}, {"local": 17}])
def test_invalid_provider_blocks_are_final_refusals_before_connect(monkeypatch, borrowed, block):
    constructor = MagicMock()
    constructor.connect = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    instance = agent(monkeypatch, sandbox_provider="sandbox")
    instance.server_client.global_config_dict = {"sandbox": block}
    check_agent_seed_rejected(instance.setup_webserver(), seed(borrowed=borrowed))
    constructor.assert_not_called()
    constructor.connect.assert_not_awaited()
