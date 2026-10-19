# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native continuation through agent HTTP routes and persisted checkpoints."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agent_activation_types import AgentContinuationRequirements
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.opencode_agent.continuation import classify_native_outcome
from responses_api_agents.opencode_agent.sandbox_runner import snapshot
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import setup as base_setup  # noqa: F401


@pytest.fixture
def setup(base_setup, tmp_path):
    agent, sandbox = base_setup
    binary = tmp_path / "opencode"
    binary.write_text("trusted test runtime")
    agent.config.local_opencode_binary_path = binary
    return agent, sandbox


def install_artifact_runner(sandbox, tmp_path):
    """Model persistence is cumulative; each supervisor writes a distinct checkpoint."""
    database = tmp_path / "session.db"
    with sqlite3.connect(database) as con:
        con.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('native-session', null, 0);
        """)
    payloads = []

    async def launch(**kwargs):
        path = [path for path in sandbox.files if path.endswith("/input.json")][-1]
        payload = json.loads(sandbox.files[path])
        payloads.append(payload)
        index = len(payloads) - 1
        directory = payload["directory"]
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
                        "native-session",
                        json.dumps(message_info),
                        len(payloads) * 10,
                    ),
                )
                con.execute(
                    "insert into part values(?,?,?,?,?)",
                    (
                        "part-" + message_id,
                        message_id,
                        "native-session",
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
                        "native-session",
                        json.dumps({"type": step_type}),
                        len(payloads) * 10,
                    ),
                )
            ids = [row[0] for row in con.execute("select id from message")]
        sandbox.files[f"{directory}/export.json"] = json.dumps(
            {
                "session_id": "native-session",
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
        if source.endswith("/observations.db"):
            Path(destination).write_bytes(database.read_bytes())
        else:
            await download(source, destination)

    sandbox.download = download_artifacts
    sandbox.launch.side_effect = launch
    return payloads


def test_interactive_http_resumes_same_store_replays_and_closes_cumulative(setup, tmp_path):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    agent.config.reasoning_effort = "high"
    agent.config.model_options = {"reasoning": True, "variants": {"high": {"reasoning": {"effort": "high"}}}}
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        assert created.json()["capabilities"]["mode"] == "native_conversation"
        for activation_id, prompt in enumerate(("Implement a first version", "Now handle empty inputs")):
            body = {
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
                "activation_id": activation_id,
                "timeout_seconds": 30,
                "responses_create_params": {"input": prompt},
            }
            response = client.post("/v1/responses", json=body)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["response"]["output"][-1]["content"][0]["text"] == f"Answer {activation_id}"
            assert result["response"]["usage"]["total_tokens"] == 16
            assert result["response"]["metadata"]["native_conversation_id"] == "native-session"
            assert result["response"]["metadata"]["launch_attempted"] == "true"
            assert result["response"]["metadata"]["native_input_accepted"] == "true"
            assert result["response"]["metadata"]["policy_input_observed"] == "unknown"
            assert result["observation"]["harness_steps"] == 1
            assert result["observation"]["events"][1]["kind"] == "reasoning"
            assert "Private reasoning" not in result["observation"]["raw_log"]
            assert client.post("/v1/responses", json=body).json() == result
            assert sandbox.launch.call_count == activation_id + 1
        config = json.loads(payloads[0]["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["model"] == "nemo_gym/dummy_model"
        assert config["provider"]["nemo_gym"]["npm"] == "@ai-sdk/openai-compatible"
        assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["variants"]["high"] == {
            "reasoning": {"effort": "high"}
        }
        assert payloads[0]["command"][-2:] == ["--variant", "high"]
        assert "--session" not in payloads[0]["command"]
        assert payloads[1]["command"][-2:] == ["--session", "native-session"]
        assert payloads[0]["env"]["HOME"] == payloads[1]["env"]["HOME"]
        assert payloads[0]["env"]["XDG_DATA_HOME"] == payloads[1]["env"]["XDG_DATA_HOME"]
        assert payloads[0]["directory"] != payloads[1]["directory"]
        assert payloads[1]["prompt"] == "Now handle empty inputs"
        assert set(payloads[1]["previous_message_ids"]) == {"u0", "a0"}
        invocations = result["observation"]["agent_observations"]["records"]
        current = [record for record in invocations if record["kind"] == "agent_invocation"][0]
        assert len(current["conversation"]) == 2
        closed = client.post(
            "/v1/agent_sessions/close",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
            },
        )
        assert closed.status_code == 200, closed.text
        receipt = closed.json()
        assert receipt["cleanup_confirmed"] is True
        assert len(receipt["activations"]) == 2
        root = [record for record in receipt["agent_observations"]["records"] if record["kind"] == "agent_invocation"][
            0
        ]
        assert len(root["conversation"]) == 4
        trajectory = receipt["trajectory"]
        assert (trajectory["task_id"], trajectory["rollout_id"]) == ("task", request.episode_id.capture_key)
        turns = trajectory["turns"]
        assert [(turn["source_message_id"], turn["turn_no"], turn["step_count"]) for turn in turns] == [
            ("a0", 1, 1),
            ("a1", 2, 2),
        ]
        assert {turn["invocation_id"] for turn in turns} == {"native-session"}
        assert all(turn["source_model_ref"] == agent.config.model_server.model_dump() for turn in turns)
        # Session affinity is insufficient to assert per-assistant HTTP call ownership.
        assert all(turn["model_calls"] == [] and turn["question"] is None for turn in turns)
        assert sandbox.disconnect.await_count == 1
        assert sandbox.stop.await_count == 0
        assert client.post("/v1/responses", json={**body, "activation_id": 2}).status_code == 409


@pytest.mark.parametrize("provider_error", [False, True])
def test_permission_rejection_preserves_checkpoint_and_resumes_after_feedback(setup, tmp_path, provider_error):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    launch = sandbox.launch.side_effect
    denial = "The user rejected permission to use this specific tool call."

    async def reject_first_tool(**kwargs):
        result = await launch(**kwargs)
        if len(payloads) != 1:
            return result
        path = f"{sandbox.directory}/export.json"
        export = json.loads(sandbox.files[path])
        message = export["messages"][-1]
        message["info"]["finish"] = "tool-calls"
        if provider_error:
            message["info"]["error"] = {"name": "APIError", "message": "Provider unavailable"}
        denied = {
            "type": "tool",
            "callID": "denied-call",
            "tool": "read",
            "state": {
                "status": "error",
                "input": {"filePath": "/"},
                "error": denial,
                "time": {"start": 10, "end": 20},
            },
        }
        message["parts"].append(denied)
        sandbox.files[path] = json.dumps(export)
        with sqlite3.connect(tmp_path / "session.db") as con:
            con.execute("update message set data=? where id='a0'", (json.dumps(message["info"]),))
            con.execute("insert into part values('denied', 'a0', 'native-session', ?, 20)", (json.dumps(denied),))
        sandbox.files[f"{sandbox.directory}/stdout.jsonl"] = "\n".join(
            json.dumps(event)
            for event in (
                {"type": "step_start", "part": {}},
                {"type": "tool_use", "part": denied},
                {"type": "step_finish", "part": {"reason": "tool-calls"}},
            )
        )
        return result

    sandbox.launch.side_effect = reject_first_tool
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        body = {
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
            "activation_id": 0,
            "timeout_seconds": 30,
            "responses_create_params": {"input": "Inspect root"},
        }
        first = client.post("/v1/responses", json=body)
        if provider_error:
            assert first.status_code == 502
            assert "Provider unavailable" in first.json()["detail"]
        else:
            assert first.status_code == 200, first.text
            interrupted = first.json()
            assert interrupted["response"]["status"] == "incomplete"
            assert interrupted["turn_complete"] is False
            assert interrupted["stop_reason"] == "permission_denied"
            assert interrupted["observation"]["events"][1]["result"] == denial
            records = interrupted["observation"]["agent_observations"]["records"]
            assert (
                next(record for record in records if record["kind"] == "agent_invocation")["error_type"]
                == "permission_denied"
            )
            assert next(record for record in records if record["kind"] == "tool_call")["status"] == "failed"
            assert client.post("/v1/responses", json=body).json() == interrupted
            assert sandbox.launch.call_count == 1
            resumed = client.post(
                "/v1/responses",
                json={
                    **body,
                    "activation_id": 1,
                    "timeout_seconds": 30,
                    "responses_create_params": {"input": "Continue inside the repository"},
                },
            )
            assert resumed.status_code == 200, resumed.text
            assert resumed.json()["turn_complete"] is True
            assert resumed.json()["stop_reason"] is None
            assert payloads[1]["native_session_id"] == "native-session"
            assert payloads[1]["previous_message_ids"] == ["u0", "a0"]
        closed = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": request.agent_session_id, "episode_id": request.episode_id.model_dump()},
        )
        assert closed.status_code == 200
        assert closed.json()["cleanup_confirmed"] is True
        assert denial in json.dumps(closed.json()["agent_observations"])
        assert len(closed.json()["activations"]) == (0 if provider_error else 2)


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
    (activation / "input.json").write_text(json.dumps({"native_session_id": "root", "previous_message_ids": ["m0"]}))
    snapshot(activation, session_directory=persistent)
    export = json.loads((activation / "export.json").read_text())
    assert export["session_id"] == "root"
    assert export["activation_message_ids"] == ["m1"]
    assert [message["id"] for message in export["messages"]] == ["m1"]
    assert [info["tokens"]["input"] for info in export["usage_messages"]] == [2]
    with sqlite3.connect(activation / "observations.db") as con:
        assert con.execute("select count(*) from message").fetchone()[0] == 2
    (activation / "input.json").write_text(json.dumps({"native_session_id": "wrong"}))
    with pytest.raises(RuntimeError, match="unexpected session"):
        snapshot(activation, session_directory=persistent)


@pytest.mark.parametrize("after_staging", [115.0, 121.0])
def test_turn_deadline_is_rechecked_after_staging(setup, tmp_path, monkeypatch, after_staging):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    import time

    origin = time.monotonic()
    now = [origin]
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.monotonic", lambda: now[0])
    monkeypatch.setattr("nemo_gym.agent_utils.sandbox_session.monotonic", lambda: now[0])
    upload = sandbox.upload

    async def stage(source, destination):
        await upload(source, destination)
        if destination.endswith("/process_supervisor.py"):
            now[0] = origin + after_staging - 100

    sandbox.upload = stage
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post(
            "/v1/responses",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
                "activation_id": 0,
                "responses_create_params": {"input": "Inspect"},
                "timeout_seconds": 20.0,
            },
        )
        assert response.status_code == 200, response.text
        if after_staging > 120:
            assert response.json()["stop_reason"] == "activation_timeout"
            assert response.json()["response"]["output"] == []
            assert response.json()["observation"]["events"] == []
            sandbox.launch.assert_not_awaited()
        else:
            assert response.json()["turn_complete"] is True
            command = sandbox.launch.await_args.kwargs["command"]
            assert "--timeout 5.0" in command
        closed = client.post(
            "/v1/agent_sessions/close",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
            },
        )
        assert closed.status_code == 200, closed.text
        assert closed.json()["cleanup_confirmed"] is True


def test_expired_resume_does_not_replay_previous_events(setup, tmp_path):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        body = {
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
            "activation_id": 0,
            "timeout_seconds": 30,
            "responses_create_params": {"input": "Inspect"},
        }
        first = client.post("/v1/responses", json=body)
        assert first.status_code == 200, first.text
        second = client.post(
            "/v1/responses",
            json={**body, "activation_id": 1, "timeout_seconds": 0.0000001},
        )
        assert second.status_code == 200, second.text
        assert second.json()["observation"]["events"] == []
        assert second.json()["observation"]["raw_log"] == ""
        assert second.json()["response"]["output"] == []
        assert second.json()["response"]["usage"] is None
        assert second.json()["response"]["metadata"]["launch_attempted"] == "false"
        assert second.json()["response"]["metadata"]["native_input_accepted"] == "false"
        assert second.json()["response"]["metadata"]["native_conversation_id"] == "native-session"
        sandbox.launch.assert_awaited_once()


@pytest.mark.parametrize(
    "session_id,previous,error_event,expected",
    [
        ("native", "native", False, "incomplete"),
        ("other", "native", False, "failed"),
        ("native", None, False, "failed"),
        ("native", "native", True, "failed"),
    ],
)
def test_timeout_before_new_assistant_requires_confirmed_same_session_checkpoint(
    session_id, previous, error_event, expected
):
    stdout = json.dumps({"type": "error", "part": {}}) if error_event else ""
    outcome = classify_native_outcome(
        {},
        cleanup={"return_code": -9, "error": None, "timed_out": True, "cleanup_confirmed": True},
        previous_session_id=previous,
        session_id=session_id,
        stdout=stdout,
        continuation=True,
    )
    assert outcome.status == expected


def test_task_policy_denies_native_tools_without_mutating_global_config(setup):
    from nemo_gym.agent_runtime_policy import AgentRuntimePolicy

    agent, sandbox = setup
    agent.config.opencode_config = {"permission": {"bash": "allow"}}
    config = agent._task_config(
        AgentRuntimePolicy(format="tools.deny.v1", settings={"deny_tools": ["WebFetch", "WebSearch"]})
    )
    assert config["permission"] == {"bash": "allow", "webfetch": "deny", "websearch": "deny"}
    assert agent.config.opencode_config == {"permission": {"bash": "allow"}}


@pytest.mark.parametrize(
    "overrides",
    [
        {"opencode_version": "latest"},
        {"concurrency": 0},
        {"remote_opencode_binary_path": "relative"},
        {"timeout": 0},
        {"max_output_tokens": 1000000},
        {"opencode_config": {"provider": {}}},
        {"env": {"EXAMPLE": "value"}},
    ],
)
def test_static_runtime_settings_are_validated_before_seed(setup, overrides):
    agent, sandbox = setup
    with pytest.raises(ValidationError):
        OpenCodeAgentConfig.model_validate(agent.config.model_dump() | overrides)
    sandbox.exec.assert_not_awaited()


def test_policy_is_validated_at_http_seed_before_sandbox_access(setup):
    agent, sandbox = setup
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()}).model_dump(mode="json")
    request["runtime_policy"] = {"format": "tools.deny.v1", "settings": {"deny_tools": ["WebFetch"]}}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=request)
        assert created.status_code == 200, created.text
    assert sandbox.exec.await_count > 0


async def test_offline_seed_uses_provisioned_python_and_ripgrep_cache(setup, tmp_path):
    agent, sandbox = setup
    agent.config.local_opencode_binary_path = None
    agent.config.remote_opencode_binary_path = "/opt/gym-runtime/opencode"
    agent.config.remote_ripgrep_binary_path = "/opt/gym-runtime/rg"
    agent.config.python_executable = "/opt/gym-runtime/python/bin/python3"
    state = await agent._seed_agent_session_state(seed())
    assert state.python == agent.config.python_executable
    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    assert commands[0].startswith("/opt/gym-runtime/python/bin/python3 -I")
    assert any("cp -- /opt/gym-runtime/opencode" in command for command in commands)
    assert any("/cache/opencode/bin/rg" in command for command in commands)
    assert not any("curl" in command or "apt-get" in command or "apk" in command for command in commands)
    assert not any(path.endswith("install_opencode_runtime.sh") for path in sandbox.files)
    await state.close(1)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "feedback"}]}],
        [{"info": {"role": "user"}, "parts": [{"type": "text", "text": "other"}]}],
    ],
)
def test_launched_input_without_matching_native_user_remains_unknown(messages):
    from responses_api_agents.opencode_agent.continuation import native_input_acceptance

    assert native_input_acceptance({"messages": messages}, prompt="feedback") == "unknown"


@pytest.mark.parametrize("plugin_ready", [True, False])
def test_timeout_after_launch_without_new_user_does_not_claim_feedback_acceptance(setup, tmp_path, plugin_ready):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        body = {
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
            "activation_id": 0,
            "timeout_seconds": 30,
            "responses_create_params": {"input": "Inspect"},
        }
        assert client.post("/v1/responses", json=body).status_code == 200

        async def interrupted_launch(**kwargs):
            path = [path for path in sandbox.files if path.endswith("/input.json")][-1]
            payload = json.loads(sandbox.files[path])
            directory = payload["directory"]
            sandbox.directory = directory
            if plugin_ready:
                sandbox.files[f"{directory}/task-environment.ready"] = "ready\n"
            sandbox.files[f"{directory}/stderr.log"] = ""
            sandbox.files[f"{directory}/cleanup.json"] = json.dumps({**sandbox.result, "timed_out": True})
            sandbox.files[f"{directory}/runtime.json"] = json.dumps(sandbox.runtime_info)
            sandbox.files[f"{directory}/stdout.jsonl"] = ""
            sandbox.files[f"{directory}/export.json"] = json.dumps(
                {
                    "session_id": "native-session",
                    "message_ids": ["u0", "a0"],
                    "activation_message_ids": [],
                    "messages": [],
                    "usage_messages": [],
                }
            )
            return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")

        sandbox.launch.side_effect = interrupted_launch
        second = client.post(
            "/v1/responses",
            json={
                **body,
                "activation_id": 1,
                "timeout_seconds": 30,
                "responses_create_params": {"input": "Handle empty inputs"},
            },
        )
        assert second.status_code == 200, second.text
        result = second.json()
        assert result["stop_reason"] == "activation_timeout"
        assert result["response"]["status"] == "incomplete"
        assert result["response"]["metadata"]["launch_attempted"] == "true"
        assert result["response"]["metadata"]["native_input_accepted"] == "unknown"
        assert result["response"]["metadata"]["policy_input_observed"] == "unknown"
        assert result["observation"]["events"] == []
        closed = client.post(
            "/v1/agent_sessions/close",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
            },
        )
        assert closed.status_code == 200 and closed.json()["cleanup_confirmed"]
