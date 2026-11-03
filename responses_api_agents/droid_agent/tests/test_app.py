# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
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
    check_agent_seed_rejected,
    shell_assignments,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.droid_agent import app
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
    assert response.usage.input_tokens == 10 and response.usage.total_tokens == 12
    assert response.usage.input_tokens_details.cached_tokens == 4
    assert response.usage.output_tokens_details.reasoning_tokens == 1
    assert response.metadata == {"harness_execution": "sandbox"}

    invocation, tool, sandbox_record = session.observations.records
    assert invocation.invocation_id == capture_key and invocation.harness_session_id == "droid-session"
    assert [item.role for item in invocation.conversation[:2]] == ["system", "user"]
    assert (tool.tool_call_id, tool.status, tool.duration_ms) == ("c1", "completed", 250)
    assert (sandbox_record.role, sandbox_record.outcome, sandbox_record.exit_code) == ("agent", "completed", 0)
    assert "model_call_ownership_unavailable" not in {gap.code for gap in session.observations.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events,cleanup,status,reason",
    [
        (EVENTS[:-1], {"timed_out": True, "return_code": 143}, "incomplete", "timeout"),
        (EVENTS[:-1], {"timed_out": False, "return_code": 0}, "incomplete", "result_missing"),
        (EVENTS, {"timed_out": False, "return_code": 2}, "failed", "process_exit_2"),
    ],
)
async def test_outcome_follows_droid_exit_and_completion(monkeypatch, events, cleanup, status, reason):
    instance = agent(monkeypatch)
    session = state(cleanup=cleanup | {"cleanup_confirmed": True})
    monkeypatch.setattr(
        adapter.DroidSandboxSession, "execute", AsyncMock(return_value=adapter.DroidArtifacts(stdout(events)))
    )
    response = await instance._sandbox_response(session, body())
    assert response.status == status
    assert session.observations.records[0].error_type == reason
    if events is not EVENTS:
        # Without the completion event Droid reports no usage, so none is invented.
        assert response.usage is None
        assert "model_call_usage_unavailable" in {gap.code for gap in session.observations.gaps}


@pytest.mark.asyncio
async def test_model_request_error_is_classified_for_the_environment(monkeypatch):
    instance = agent(monkeypatch)
    session = state(cleanup={"timed_out": False, "return_code": 1, "cleanup_confirmed": True})
    error = '404 {"status":404,"title":"Not Found"}'
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


@pytest.mark.parametrize(
    "message,kind",
    [
        ("429 rate limited", ("model_request_error", 429)),
        ("Connection error.", ("model_transport_error", None)),
        ("tool crashed", ("unknown", None)),
    ],
)
def test_classifier_reads_droid_error_messages(monkeypatch, message, kind):
    outcome = agent(monkeypatch).classify_outcome({"error": message})
    assert (outcome.failure_kind, outcome.status_code) == kind


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
        request=seed(), runtime="/r", session=SimpleNamespace(session_dir="/s", workdir="/app", sandbox=object())
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
