# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn continuation uses ordinary Responses requests and per-activation evidence."""

import asyncio
import json
import shlex
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.agent_activation_types import AgentContinuationRequirements
from nemo_gym.base_responses_api_agent import AGENT_ACTIVATION_HEADER, AgentCloseSessionRequest
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.opencode_agent.continuation import classify_opencode_run
from responses_api_agents.opencode_agent.snapshot import snapshot
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed, setup  # noqa: F401


def install_artifact_runner(sandbox, tmp_path):
    """Model persistence is cumulative; each supervisor writes a distinct checkpoint."""
    database = tmp_path / "session.db"
    with sqlite3.connect(database) as con:
        con.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('opencode-session', null, 0);
        """)
    payloads = []

    async def launch(**kwargs):
        runner_params = next(arg for arg in shlex.split(kwargs["command"]) if arg.endswith("/params.sh"))
        directory = runner_params.removesuffix("/params.sh")
        payload = sandbox.params(directory) | {"directory": directory}
        payloads.append(payload)
        index = len(payloads) - 1
        sandbox.directory = directory
        sandbox.files[f"{directory}/task-environment.ready"] = "ready\n"
        sandbox.files[f"{directory}/stderr.log"] = ""
        info = {
            "role": "assistant",
            "finish": "stop",
            "time": {"completed": 20 + index},
            "tokens": {"input": 10, "output": 2, "reasoning": 1, "cache": {"read": 3}},
        }
        part = {"type": "text", "text": f"Answer {index}"}
        user_id, assistant_id = f"u{index}", f"a{index}"
        with sqlite3.connect(database) as con:
            for message_id, message_info, message_part in (
                (user_id, {"role": "user"}, {"type": "text", "text": payload["prompt"]}),
                (assistant_id, info, part),
            ):
                con.execute(
                    "insert into message values(?,?,?,?)",
                    (
                        message_id,
                        "opencode-session",
                        json.dumps(message_info),
                        len(payloads) * 10,
                    ),
                )
                con.execute(
                    "insert into part values(?,?,?,?,?)",
                    (
                        "part-" + message_id,
                        message_id,
                        "opencode-session",
                        json.dumps(message_part),
                        len(payloads) * 10,
                    ),
                )
            for step_type in ("step-start", "step-finish"):
                con.execute(
                    "insert into part values(?,?,?,?,?)",
                    (
                        step_type + assistant_id,
                        assistant_id,
                        "opencode-session",
                        json.dumps({"type": step_type}),
                        len(payloads) * 10,
                    ),
                )
            ids = [row[0] for row in con.execute("select id from message")]
        sandbox.files[f"{directory}/export.json"] = json.dumps(
            {
                "session_id": "opencode-session",
                "message_ids": ids,
                "activation_message_ids": [user_id, assistant_id],
                "messages": [
                    {"info": {"role": "user"}, "parts": [{"type": "text", "text": payload["prompt"]}]},
                    {"info": info, "parts": [part]},
                ],
                "usage_messages": [info],
            }
        )
        sandbox.files[f"{directory}/stdout.jsonl"] = "\n".join(
            json.dumps(event)
            for event in (
                {"type": "step_start", "part": {}},
                {"type": "reasoning", "part": {"text": f"Private reasoning {index}"}},
                {"type": "text", "part": part},
                {"type": "step_finish", "part": {"reason": "stop"}},
            )
        )
        sandbox.files[f"{directory}/cleanup.json"] = json.dumps(sandbox.result)
        sandbox.files[f"{directory}/runtime.json"] = json.dumps(sandbox.runtime_info)
        return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")

    download = sandbox.download

    async def download_artifacts(source, destination):
        if source.endswith("/opencode.db"):
            Path(destination).write_bytes(database.read_bytes())
        else:
            await download(source, destination)

    sandbox.download = download_artifacts
    sandbox.launch.side_effect = launch
    return payloads


def start(client):
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    created = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
    assert created.status_code == 200, created.text
    return request


def activate(client, index, prompt="Inspect", *, timeout=30, instructions=None):
    body = {"input": prompt, "metadata": {"agent_timeout_sec": str(timeout)}}
    if instructions is not None:
        body["instructions"] = instructions
    return client.post("/v1/responses", headers={AGENT_ACTIVATION_HEADER: str(index)}, json=body)


def close(client, request):
    response = client.post(
        "/v1/agent_sessions/close",
        json={
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def roots(observations):
    return [
        r for r in observations["records"] if r["kind"] == "agent_invocation" and r["parent_invocation_id"] is None
    ]


@pytest.mark.parametrize("activation_count", [1, 2])
@pytest.mark.parametrize("disconnect_failure", [False, True])
def test_negative_cleanup_releases_borrowed_session_and_replays_receipt(
    setup, tmp_path, activation_count, disconnect_failure
):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        request = start(client)
        state = agent._require_agent_session(request.agent_session_id)
        for index in range(1, activation_count):
            assert activate(client, index).status_code == 200
        sandbox.result["cleanup_confirmed"] = False
        assert activate(client, activation_count).status_code == 500
        assert state.session.termination_error is not None
        # The failed launch must not snapshot an unconfirmed process tree.
        assert state.session.artifacts is None
        if disconnect_failure:
            sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
            failed_close = client.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": request.agent_session_id, "episode_id": request.episode_id.model_dump()},
            )
            assert failed_close.status_code == 500
            assert agent._session_records[request.agent_session_id].state is state
        receipt = close(client, request)
        assert receipt["cleanup_confirmed"] is False
        assert close(client, request) == receipt
        assert state.session.closed and not state.session.cleanup_confirmed
        assert agent._session_records[request.agent_session_id].state is None
        assert activate(client, activation_count + 1).status_code == 409
        assert [
            r["activation_id"] for r in roots(receipt["agent_observations"]) if r["status"] == "completed"
        ] == list(range(1, activation_count))
    assert sandbox.disconnect.await_count == (2 if disconnect_failure else 1)
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("prior_success", [False, True])
@pytest.mark.parametrize("disconnect_failure", [False, True])
def test_close_retains_recovered_activation_once(setup, tmp_path, monkeypatch, prior_success, disconnect_failure):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        request = start(client)
        state = agent._require_agent_session(request.agent_session_id)
        earlier = None
        if prior_success:
            assert activate(client, 1).status_code == 200
            earlier = roots(state.observations.model_dump(mode="json"))[0]
        index = 2 if prior_success else 1
        stop = AsyncMock(side_effect=[RuntimeError("temporary receipt unavailable"), dict(sandbox.result)])
        monkeypatch.setattr("nemo_gym.agent_utils.sandbox_session.stop_and_confirm_cleanup", stop)
        capture = AsyncMock(wraps=state.collect_artifacts)
        state.collect_artifacts = capture
        assert activate(client, index).status_code == 500
        stop.assert_awaited_once()
        capture.assert_not_awaited()
        [failed] = [r for r in roots(state.observations.model_dump(mode="json")) if r["activation_id"] == index]
        assert failed["status"] == "failed" and failed["conversation"] == []
        if disconnect_failure:
            sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
            response = client.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": request.agent_session_id, "episode_id": request.episode_id.model_dump()},
            )
            assert response.status_code == 500
        receipt = close(client, request)
        assert receipt["cleanup_confirmed"] is True
        assert close(client, request) == receipt
        assert stop.await_count == 2
        capture.assert_awaited_once()
        invocations = roots(receipt["agent_observations"])
        assert [r["activation_id"] for r in invocations] == list(range(1, index + 1))
        if earlier is not None:
            assert invocations[0] == earlier
        recovered = invocations[-1]
        assert recovered["invocation_id"] == f"{state.invocation_prefix}:opencode-session"
        assert recovered["harness_session_id"] == "opencode-session"
        for field in ("status", "error_type", "stop_reason", "duration_ms"):
            assert recovered[field] == failed[field]
        assert any(
            message.get("role") == "assistant" and message["content"][0]["text"] == f"Answer {index - 1}"
            for message in recovered["conversation"]
        )
        trajectory = receipt["trajectory"]
        assert [turn["activation_id"] for turn in trajectory["turns"]] == list(range(1, index + 1))
        assert trajectory["turns"][-1]["invocation_id"] == recovered["invocation_id"]
        assert (
            len(
                {
                    r["invocation_id"]
                    for r in receipt["agent_observations"]["records"]
                    if r["kind"] == "agent_invocation"
                }
            )
            == index
        )


async def test_close_recovery_preserves_cancelled_activation(setup, tmp_path, monkeypatch):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    request = Request(
        {"type": "http", "session": {}, "headers": [(AGENT_ACTIVATION_HEADER.encode(), b"1")], "path_params": {}}
    )
    body = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    await agent.seed_agent_session(request, body)
    stop = AsyncMock(side_effect=[asyncio.CancelledError(), dict(sandbox.result)])
    monkeypatch.setattr("nemo_gym.agent_utils.sandbox_session.stop_and_confirm_cleanup", stop)
    with pytest.raises(asyncio.CancelledError):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    stop.assert_awaited_once()
    close_body = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    receipt = await agent.close_agent_session(request, close_body)
    [invocation] = roots(receipt.agent_observations.model_dump(mode="json"))
    assert invocation["status"] == "incomplete" and invocation["error_type"] == "cancelled"
    assert invocation["harness_session_id"] == "opencode-session" and invocation["conversation"]
    assert [turn.activation_id for turn in receipt.trajectory.turns] == [1]
    assert await agent.close_agent_session(request, close_body) == receipt
    assert stop.await_count == 2


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("artifact_failure", ["copy", "parse"])
@pytest.mark.parametrize("prior_success", [False, True])
async def test_close_without_recovered_root_preserves_activation(
    setup, tmp_path, monkeypatch, cancelled, artifact_failure, prior_success
):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    request = Request({"type": "http", "session": {}, "headers": [], "path_params": {}})
    body = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    await agent.seed_agent_session(request, body)
    state = agent._require_agent_session(body.agent_session_id)
    if prior_success:
        request = Request({**request.scope, "headers": [(AGENT_ACTIVATION_HEADER.encode(), b"1")]})
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="first task"))
    index = 2 if prior_success else 1
    request = Request({**request.scope, "headers": [(AGENT_ACTIVATION_HEADER.encode(), str(index).encode())]})
    error = asyncio.CancelledError() if cancelled else RuntimeError("temporary receipt unavailable")
    stop = AsyncMock(side_effect=[error, dict(sandbox.result)])
    monkeypatch.setattr("nemo_gym.agent_utils.sandbox_session.stop_and_confirm_cleanup", stop)
    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="next task"))
    retained = state.observations.model_dump(mode="json")
    trajectory = state.trajectory.model_dump(mode="json") if state.trajectory is not None else None
    [original] = [r for r in roots(retained) if r["activation_id"] == index]
    assert original["status"] == ("incomplete" if cancelled else "failed")
    download = sandbox.download

    async def broken_observations(source, destination):
        if source.endswith("/opencode.db"):
            # The export survives; only the observation store is unusable.
            Path(destination).write_bytes(b"" if artifact_failure == "copy" else b"not a SQLite database")
        else:
            await download(source, destination)

    sandbox.download = broken_observations
    close_body = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    receipt = await agent.close_agent_session(request, close_body)
    observations = receipt.agent_observations.model_dump(mode="json")
    assert observations["records"] == retained["records"]
    assert all(gap in observations["gaps"] for gap in retained["gaps"])
    assert {gap["code"] for gap in observations["gaps"]} >= {
        "agent_artifact_unavailable",
        "observation_capture_failed",
    }
    assert observations["gaps"] == [
        *retained["gaps"],
        {"code": "agent_artifact_unavailable", "invocation_id": None, "detail": None},
    ]
    assert (receipt.trajectory.model_dump(mode="json") if receipt.trajectory is not None else None) == trajectory
    assert receipt.cleanup_confirmed is True
    state.retain_activation()
    assert state.observations.model_dump(mode="json") == observations
    assert await agent.close_agent_session(request, close_body) == receipt
    assert stop.await_count == 2


def test_plain_http_delta_replay_and_cumulative_close(setup, tmp_path):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        state = agent._require_agent_session(request.agent_session_id)
        assert state.is_continuation
        assert state.conversation_directory is None and state.activation_id is None
        results = []
        for index, prompt in enumerate(("First version", "Correction: handle empty inputs"), 1):
            response = activate(client, index, prompt)
            assert response.status_code == 200, response.text
            result = response.json()
            results.append(result)
            assert result["object"] == "response" and "response" not in result
            assert result["output"][-1]["content"][0]["text"] == f"Answer {index - 1}"
            assert result["usage"]["total_tokens"] == 16
            assert result["metadata"]["harness_session_id"] == "opencode-session"
            assert result["metadata"]["activation_id"] == str(index)
            invocation = roots(result["_ng_agent_observations"])[0]
            assert invocation["activation_id"] == index
            assert invocation["harness_session_id"] == "opencode-session"
            assert invocation["status"] == "completed"
            assert len(invocation["conversation"]) == 2
            assert (
                invocation["invocation_id"]
                == payloads[-1]["env"]["NEMO_GYM_OPENCODE_ACTIVATION_ID"] + ":opencode-session"
            )
            turns = result["_ng_trajectory"]["turns"]
            assert len(turns) == 1 and turns[0]["activation_id"] == index
            assert turns[0]["invocation_id"] == invocation["invocation_id"]
            assert activate(client, index, prompt).json() == result
            assert sandbox.launch.call_count == index
        assert "--session" not in payloads[0]["command"]
        assert payloads[1]["command"][-2:] == ["--session", "opencode-session"]
        assert payloads[0]["env"]["HOME"] == payloads[1]["env"]["HOME"]
        assert payloads[0]["env"]["XDG_DATA_HOME"] == payloads[1]["env"]["XDG_DATA_HOME"]
        assert set(sandbox.snapshots[1]["previous_message_ids"]) == {"u0", "a0"}
        receipt = close(client, request)
        invocations = roots(receipt["agent_observations"])
        assert [r["activation_id"] for r in invocations] == [1, 2]
        assert len({r["invocation_id"] for r in invocations}) == 2
        assert [r["status"] for r in invocations] == ["completed", "completed"]
        assert len(receipt["trajectory"]["turns"]) == 2
        assert receipt["cleanup_confirmed"] and "activations" not in receipt
        assert close(client, request) == receipt
        assert activate(client, 3).status_code == 409
    assert sandbox.disconnect.await_count == 1 and sandbox.stop.await_count == 0


def test_invalid_delta_and_instructions_leave_activation_available(setup, tmp_path):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        assert activate(client, 1, "").status_code == 422
        assert activate(client, 1, instructions="Keep changes small").status_code == 200
        assert activate(client, 2, instructions="Replace instructions").status_code == 422
        assert activate(client, 2, "A valid correction").status_code == 200
        assert sandbox.launch.call_count == 2
        close(client, request)


@pytest.mark.parametrize("stage", ["prepare", "queue", "upload"])
def test_first_expiry_before_launch_can_start_next_activation(setup, tmp_path, stage):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    execute, upload = sandbox.exec, sandbox.upload

    async def slow_exec(command, **kwargs):
        if stage == "prepare" and command.startswith("mkdir") and "/activation-1" in command:
            await asyncio.sleep(0.2)
        await execute(command, **kwargs)
        return execute.return_value

    async def slow_upload(source, destination):
        if stage == "upload" and destination.endswith("process_supervisor.sh"):
            await asyncio.sleep(0.2)
        await upload(source, destination)

    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        slow_exec.return_value = execute.return_value
        sandbox.exec = slow_exec
        sandbox.upload = slow_upload
        if stage == "queue":
            agent.sem = asyncio.Semaphore(0)
        first = activate(client, 1, timeout=0.01)
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["status"] == "incomplete" and result["output"] == []
        assert result["usage"] is None and roots(result["_ng_agent_observations"])[0]["error_type"] == "timeout"
        sandbox.launch.assert_not_awaited()
        assert activate(client, 1, timeout=0.01).json() == result
        sandbox.exec, sandbox.upload = execute, upload
        agent.sem = asyncio.Semaphore(1)
        second = activate(client, 2, "Start now")
        assert second.status_code == 200 and second.json()["status"] == "completed", second.text
        receipt = close(client, request)
        assert [r["status"] for r in roots(receipt["agent_observations"])] == ["incomplete", "completed"]
        assert receipt["cleanup_confirmed"]


def test_expired_later_activation_keeps_prior_status_and_no_duplicate_turns(setup, tmp_path):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        assert activate(client, 1).status_code == 200
        agent.sem = asyncio.Semaphore(0)
        result = activate(client, 2, "Correction", timeout=0.01)
        assert result.status_code == 200, result.text
        assert result.json()["_ng_trajectory"]["turns"] == []
        receipt = close(client, request)
        assert [r["status"] for r in roots(receipt["agent_observations"])] == ["completed", "incomplete"]
        assert [t["activation_id"] for t in receipt["trajectory"]["turns"]] == [1]
        sandbox.launch.assert_awaited_once()


def test_continuation_requires_standard_header_and_keeps_plain_schema(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        assert client.post("/v1/responses", json={"input": "Task"}).status_code == 422
        assert activate(client, 0).status_code == 409
        assert activate(client, 2).status_code == 409
        assert (
            client.post("/v1/responses", headers={AGENT_ACTIVATION_HEADER: "bad"}, json={"input": "Task"}).status_code
            == 409
        )
        schema = agent.setup_webserver().openapi()
        body = schema["paths"]["/v1/responses"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert body["$ref"].split("/")[-1].removesuffix("-Input") == "NeMoGymResponseCreateParamsNonStreaming"
        close(client, request)
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("finish,reason", [("length", "max_output_tokens"), ("content-filter", "content_filter")])
def test_opencode_budget_uses_response_incomplete_details(finish, reason):
    result = classify_opencode_run(
        {"messages": [{"info": {"role": "assistant", "finish": finish, "time": {"completed": 1}}}]},
        cleanup={"return_code": 0, "error": None, "timed_out": False, "cleanup_confirmed": True},
        previous_session_id=None,
        session_id="opencode-session",
        stdout="",
        continuation=True,
    )
    assert result.status == "incomplete" and result.incomplete_reason == reason


@pytest.mark.parametrize(
    "session_id,previous,error_event,expected",
    [
        ("opencode-session", "opencode-session", False, "incomplete"),
        ("other", "opencode-session", False, "failed"),
        ("opencode-session", None, False, "incomplete"),
        ("opencode-session", "opencode-session", True, "failed"),
    ],
)
def test_interrupted_checkpoint_does_not_hide_opencode_error(session_id, previous, error_event, expected):
    outcome = classify_opencode_run(
        {},
        cleanup={"return_code": -9, "error": None, "timed_out": True, "cleanup_confirmed": True},
        previous_session_id=previous,
        session_id=session_id,
        stdout=json.dumps({"type": "error"}) if error_event else "",
        continuation=True,
    )
    assert outcome.status == expected


@pytest.mark.parametrize("previous", [None, "opencode-session"])
@pytest.mark.parametrize("assistant", [False, True])
@pytest.mark.parametrize("error_event", [False, True])
def test_timeout_opencode_error_overrides_partial_assistant(previous, assistant, error_event):
    outcome = classify_opencode_run(
        {"messages": [{"info": {"role": "assistant"}, "parts": []}] if assistant else []},
        cleanup={"return_code": -9, "error": None, "timed_out": True, "cleanup_confirmed": True},
        previous_session_id=previous,
        session_id="opencode-session",
        stdout=json.dumps({"type": "error"}) if error_event else "",
        continuation=True,
    )
    if error_event:
        assert outcome.status == "failed" and outcome.error_type == "server_error" and outcome.error
    else:
        assert outcome.status == "incomplete" and outcome.error_type == "timeout"


def test_single_turn_timeout_keeps_existing_opencode_error_classification():
    outcome = classify_opencode_run(
        {"messages": [{"info": {"role": "assistant"}, "parts": []}]},
        cleanup={"return_code": -9, "error": None, "timed_out": True, "cleanup_confirmed": True},
        previous_session_id=None,
        session_id="opencode-session",
        stdout=json.dumps({"type": "error"}),
        continuation=False,
    )
    assert outcome.status == "incomplete" and outcome.error_type == "timeout"


@pytest.mark.parametrize("prior_success", [False, True])
@pytest.mark.parametrize("error_event", [False, True])
def test_opencode_error_during_timeout_fails_http_activation(setup, tmp_path, prior_success, error_event):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    launch = sandbox.launch.side_effect

    async def interrupted(**kwargs):
        result = await launch(**kwargs)
        sandbox.files[f"{payloads[-1]['directory']}/stdout.jsonl"] = (
            json.dumps({"type": "error", "error": {"name": "APIError", "data": {"statusCode": 400}}})
            if error_event
            else ""
        )
        return result

    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        request = start(client)
        if prior_success:
            assert activate(client, 1).status_code == 200
        sandbox.launch.side_effect = interrupted
        sandbox.result.update(timed_out=True, return_code=-9)
        response = activate(client, 2 if prior_success else 1)
        if error_event:
            assert response.status_code == 502
            assert "OpenCode emitted an error event" in response.json()["detail"]
            assert (
                '"name": "APIError"' in response.json()["detail"] and '"statusCode": 400' in response.json()["detail"]
            )
        else:
            assert response.status_code == 200 and response.json()["status"] == "incomplete"
        invocation = roots(close(client, request)["agent_observations"])[-1]
        assert invocation["status"] == ("failed" if error_event else "incomplete")
        assert invocation["error_type"] == ("server_error" if error_event else "timeout")


@pytest.mark.parametrize(
    "error_name,expected_status,expected_error_type",
    [("APIError", "failed", "provider_error"), ("ContextOverflowError", "incomplete", "context_limit")],
)
def test_reported_error_retains_continuation_evidence(
    setup, tmp_path, error_name, expected_status, expected_error_type
):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    launch = sandbox.launch.side_effect

    async def report_error(**kwargs):
        result = await launch(**kwargs)
        path = f"{payloads[-1]['directory']}/export.json"
        export = json.loads(sandbox.files[path])
        info = export["messages"][-1]["info"]
        info["finish"] = "error" if error_name == "ContextOverflowError" else None
        info["error"] = {
            "name": error_name,
            "data": {"statusCode": 503, "message": "provider message", "responseBody": "private-body"},
        }
        sandbox.files[path] = json.dumps(export)
        return result

    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        first = activate(client, 1).json()
        assert first["status"] == "completed"
        sandbox.launch.side_effect = report_error
        sandbox.result["return_code"] = 1
        response = activate(client, 2, "Continue")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == expected_status
        assert result["metadata"]["activation_id"] == "2"
        assert result["metadata"]["harness_session_id"] == "opencode-session"
        if error_name == "APIError":
            assert result["error"]["message"] == "HTTP 503: provider message"
            assert json.loads(result["metadata"]["ng_agent_outcome"]) == {
                "failure_kind": "model_request_error",
                "status_code": 503,
            }
        else:
            assert result["error"] is None
        [invocation] = roots(result["_ng_agent_observations"])
        assert invocation["activation_id"] == 2
        assert (invocation["status"], invocation["error_type"]) == (expected_status, expected_error_type)
        assert activate(client, 2, "Continue").json() == result
        assert len(payloads) == 2
        invocations = roots(close(client, request)["agent_observations"])
        assert [(r["activation_id"], r["status"]) for r in invocations] == [(1, "completed"), (2, expected_status)]


def test_collector_and_existing_conformance_keep_two_activation_owners(setup, tmp_path):
    from nemo_gym.base_responses_api_model import ModelCallRecord
    from nemo_gym.harness_capabilities.checker import EvidenceScope, inspect_record
    from nemo_gym.rollout_collection import _build_trajectory_record
    from nemo_gym.rollout_observability import (
        AgentObservationBundle,
        TrajectoryRecord,
        join_agent_model_calls,
        join_assistant_message_calls,
    )

    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        outputs = [activate(client, index, f"delta-{index}").json() for index in (1, 2)]
        receipt = close(client, request)
    calls = []
    for index, output in enumerate(outputs):
        invocation = roots(output["_ng_agent_observations"])[0]
        calls.append(
            ModelCallRecord(
                call_index=index,
                model_call_id=f"capture-{index}",
                response_id=f"reply-{index}",
                model_ref=agent.config.model_server,
                model="fixture",
                dialect="responses",
                status_code=200,
                response_status="completed",
                client_session_id=invocation["invocation_id"],
                client_assistant_message_id=f"a{index}",
                started_at=float(index),
                completed_at=index + 0.5,
                tokens_in=13,
                tokens_out=3,
                tokens_total=16,
                tokens_reasoning=1,
                cached_tokens=3,
                request={"input": [{"role": "user", "content": f"delta-{index + 1}"}]},
                response={
                    "id": f"reply-{index}",
                    "status": "completed",
                    "output": output["output"],
                    "usage": output["usage"],
                },
            )
        )
    bundle = join_agent_model_calls(AgentObservationBundle.model_validate(receipt["agent_observations"]), calls)
    bundle, trajectory = join_assistant_message_calls(
        bundle, TrajectoryRecord.model_validate(receipt["trajectory"]), calls
    )
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "task_id": "task",
        "_ng_rollout_id": request.episode_id.capture_key,
    }
    result = {
        "ng_agent_observations": bundle.model_dump(mode="json"),
        "ng_trajectory": trajectory.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": [c.model_dump(mode="json") for c in calls]},
        "response": outputs[-1],
    }
    canonical = _build_trajectory_record(row, result)
    result["ng_trajectory"] = canonical.model_dump(mode="json")
    assert [inv.activation_id for inv in canonical.invocations] == [1, 2]
    assert [[ref.model_call_id for ref in inv.model_calls] for inv in canonical.invocations] == [
        ["capture-0"],
        ["capture-1"],
    ]
    assert [turn.activation_id for turn in canonical.turns] == [1, 2]
    assert [[ref.model_call_id for ref in turn.model_calls] for turn in canonical.turns] == [
        ["capture-0"],
        ["capture-1"],
    ]
    report = inspect_record(row | result, scope=EvidenceScope(tools=False, verifier=False))
    failed = [check for check in report["checks"] if check["status"] == "fail"]
    assert not failed, failed


def test_permission_yield_is_opencode_continuation_outcome_not_a_task_policy():
    info = {"role": "assistant", "finish": "tool-calls", "time": {"completed": 1}}
    cleanup = {"return_code": 0, "error": None, "timed_out": False, "cleanup_confirmed": True}
    export = {
        "messages": [
            {
                "info": info,
                "parts": [
                    {
                        "type": "tool",
                        "state": {
                            "status": "error",
                            "error": "The user rejected permission to use this specific tool call.",
                        },
                    }
                ],
            }
        ]
    }
    result = classify_opencode_run(
        export, cleanup=cleanup, previous_session_id=None, session_id="opencode-session", stdout="", continuation=True
    )
    assert result.status == "incomplete" and result.error_type == "permission_denied"
    export["messages"][0]["parts"][0]["state"]["error"] = "A different tool failure"
    result = classify_opencode_run(
        export, cleanup=cleanup, previous_session_id=None, session_id="opencode-session", stdout="", continuation=True
    )
    assert result.status == "failed"


def test_ordinary_seed_does_not_add_continuation_capabilities(setup):
    agent, sandbox = setup
    request = seed()
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert not agent._require_agent_session(request.agent_session_id).is_continuation
        assert response.json() == {"agent_session_id": request.agent_session_id, "supports_task_tool_context": False}
        receipt = close(client, request)
        assert "cleanup_confirmed" not in receipt and "trajectory" not in receipt


def test_provider_preparation_timeout_does_not_brick_unstarted_session(setup, tmp_path):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    execute = sandbox.exec.side_effect

    async def timeout_once(command, **kwargs):
        if command.startswith("mkdir") and "/activation-1" in command:
            return SimpleNamespace(return_code=None, stdout="", stderr="", error_type="timeout")
        return await execute(command, **kwargs)

    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        sandbox.exec.side_effect = timeout_once
        first = activate(client, 1)
        assert first.status_code == 200 and first.json()["status"] == "incomplete", first.text
        sandbox.launch.assert_not_awaited()
        assert activate(client, 2).status_code == 200
        receipt = close(client, request)
        assert [r["status"] for r in roots(receipt["agent_observations"])] == ["incomplete", "completed"]


def test_preparation_error_keeps_failed_activation_in_close(setup):
    agent, sandbox = setup
    run = sandbox.exec.side_effect

    async def fail(command, **kwargs):
        if command.startswith("mkdir") and "/activation-1" in command:
            raise OSError("directory service unavailable")
        return await run(command, **kwargs)

    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        request = start(client)
        sandbox.exec.side_effect = fail
        assert activate(client, 1).status_code == 500
        receipt = close(client, request)
        [root] = roots(receipt["agent_observations"])
        assert root["activation_id"] == 1 and root["status"] == "failed"
        assert root["error_type"] == "server_error"
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("retire", [False, True])
async def test_cancelled_preparation_retains_activation(setup, retire):
    agent, sandbox = setup
    entered = asyncio.Event()
    run = sandbox.exec.side_effect

    async def prepare(command, **kwargs):
        if command.startswith("mkdir") and "/activation-1" in command:
            entered.set()
            await asyncio.Future()
        return await run(command, **kwargs)

    request = Request(
        {"type": "http", "session": {}, "headers": [(AGENT_ACTIVATION_HEADER.encode(), b"1")], "path_params": {}}
    )
    body = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    await agent.seed_agent_session(request, body)
    state = agent._require_agent_session(body.agent_session_id)
    sandbox.exec.side_effect = prepare
    running = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(entered.wait(), 2)
    if retire:
        await agent.retire_agent_session(body.agent_session_id)
    else:
        await agent.close_agent_session(
            request, AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
        )
    with pytest.raises(asyncio.CancelledError):
        await running
    [root] = roots(state.observations.model_dump(mode="json"))
    assert root["activation_id"] == 1 and root["status"] == "incomplete" and root["error_type"] == "cancelled"
    if retire:
        assert body.agent_session_id not in agent._session_records
    else:
        assert agent._session_records[body.agent_session_id].state is None
    sandbox.launch.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()


async def test_close_timeout_retry_waits_for_finalization_and_retains_evidence(setup, tmp_path):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    collecting, release = asyncio.Event(), asyncio.Event()
    request = Request(
        {"type": "http", "session": {}, "headers": [(AGENT_ACTIVATION_HEADER.encode(), b"1")], "path_params": {}}
    )
    body = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    await agent.seed_agent_session(request, body)
    state = agent._require_agent_session(body.agent_session_id)
    collect = state.collect_artifacts

    async def delayed_collect(*, timeout):
        collecting.set()
        await release.wait()
        return await collect(timeout=timeout)

    state.collect_artifacts = delayed_collect
    running = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(collecting.wait(), 2)
    close_body = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    agent.config.session_close_timeout_seconds = 0.01
    try:
        with pytest.raises(TimeoutError):
            await agent.close_agent_session(request, close_body)
        assert agent._session_records[body.agent_session_id].state is state
        agent.config.session_close_timeout_seconds = 1
        retry = asyncio.create_task(agent.close_agent_session(request, close_body))
        await asyncio.sleep(0)
        release.set()
        receipt = await retry
        with pytest.raises(asyncio.CancelledError):
            await running
        [root] = roots(receipt.agent_observations.model_dump(mode="json"))
        assert root["activation_id"] == 1 and root["status"] == "incomplete"
        assert root["error_type"] == "cancelled"
        assert any(
            message.get("role") == "assistant" and message["content"][0]["text"] == "Answer 0"
            for message in root["conversation"]
        )
        assert [turn.activation_id for turn in receipt.trajectory.turns] == [1]
        assert await agent.close_agent_session(request, close_body) == receipt
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        agent.config.session_close_timeout_seconds = 1
        await agent.close_agent_session(request, close_body)


@pytest.mark.parametrize("receipt", [False, True])
def test_first_launched_timeout_without_assistant_can_resume(setup, tmp_path, receipt):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    launch = sandbox.launch.side_effect

    async def interrupted(**kwargs):
        result = await launch(**kwargs)
        if len(payloads) == 1:
            directory = payloads[0]["directory"]
            sandbox.files[f"{directory}/export.json"] = json.dumps(
                {"session_id": "opencode-session", "message_ids": [], "activation_message_ids": [], "messages": []}
            )
            sandbox.files[f"{directory}/stdout.jsonl"] = ""
            if not receipt:
                sandbox.files.pop(f"{directory}/task-environment.ready")
            with sqlite3.connect(tmp_path / "session.db") as con:
                con.execute("delete from message")
                con.execute("delete from part")
        return result

    sandbox.launch.side_effect = interrupted
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        sandbox.result.update(timed_out=True, return_code=-9)
        first = activate(client, 1)
        assert first.status_code == 200 and first.json()["status"] == "incomplete", first.text
        sandbox.result.update(timed_out=False, return_code=0)
        assert activate(client, 2).status_code == 200
        assert sandbox.snapshots[1]["harness_session_id"] == "opencode-session"
        assert [r["status"] for r in roots(close(client, request)["agent_observations"])] == [
            "incomplete",
            "completed",
        ]


def test_snapshot_scopes_delta_and_keeps_cumulative_store(tmp_path):
    persistent = tmp_path / "persistent"
    database = persistent / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as con:
        con.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('root', null, 0);
        """)
        for index in range(2):
            info = {"role": "assistant", "tokens": {"input": index + 1}}
            con.execute("insert into message values(?,?,?,?)", (f"m{index}", "root", json.dumps(info), index))
            con.execute(
                "insert into part values(?,?,?,?,?)",
                (
                    f"p{index}",
                    f"m{index}",
                    "root",
                    json.dumps({"type": "text", "text": str(index)}),
                    index,
                ),
            )
    activation = tmp_path / "activation"
    activation.mkdir()
    snapshot(database, activation, harness_session_id="root", previous_message_ids=["m0"])
    export = json.loads((activation / "export.json").read_text())
    assert export["session_id"] == "root"
    assert export["activation_message_ids"] == ["m1"]
    assert [message["id"] for message in export["messages"]] == ["m1"]
    assert [info["tokens"]["input"] for info in export["usage_messages"]] == [2]
    with sqlite3.connect(activation / "observations.db") as con:
        assert con.execute("select count(*) from message").fetchone()[0] == 2
    with pytest.raises(RuntimeError, match="unexpected session"):
        snapshot(database, activation, harness_session_id="wrong", previous_message_ids=[])
