# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume the real adapter contract with persistent artifacts across activations."""

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nemo_gym.interactive_agent_types import AgentContinuationRequirements
from responses_api_agents.opencode_agent.continuation import parse_activation_events
from responses_api_agents.opencode_agent.sandbox_runner import snapshot
from responses_api_agents.opencode_agent.tests.test_native_sessions import seed  # noqa: F401
from responses_api_agents.opencode_agent.tests.test_native_sessions import setup as setup


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
            ids = [row[0] for row in con.execute("select id from message")]
        sandbox.files[f"{directory}/export.json"] = json.dumps(
            {
                "session_id": "native-session",
                "message_ids": ids,
                "activation_message_ids": [user_id, assistant_id],
                "messages": [{"info": info, "parts": [part]}],
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
    agent.config.native_model_id = "openai/gpt-5.5"
    agent.config.native_provider_npm = "@openrouter/ai-sdk-provider"
    agent.config.reasoning_effort = "high"
    agent.config.native_model_options = {"reasoning": True, "variants": {"high": {"reasoning": {"effort": "high"}}}}
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
                "responses_create_params": {"input": prompt},
            }
            response = client.post("/v1/agent_sessions/activate", json=body)
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["response"]["output"][-1]["content"][0]["text"] == f"Answer {activation_id}"
            assert result["response"]["usage"]["total_tokens"] == 16
            assert result["observation"]["harness_steps"] == 1
            assert result["observation"]["events"][1]["kind"] == "reasoning"
            assert "Private reasoning" not in result["observation"]["raw_log"]
            assert client.post("/v1/agent_sessions/activate", json=body).json() == result
            assert sandbox.launch.call_count == activation_id + 1
        config = json.loads(payloads[0]["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["model"] == "nemo_gym/openai/gpt-5.5"
        assert config["provider"]["nemo_gym"]["npm"] == "@openrouter/ai-sdk-provider"
        assert config["provider"]["nemo_gym"]["models"]["openai/gpt-5.5"]["variants"]["high"] == {
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
        assert sandbox.disconnect.await_count == 1
        assert sandbox.stop.await_count == 0
        assert client.post("/v1/agent_sessions/activate", json={**body, "activation_id": 2}).status_code == 409


@pytest.mark.parametrize("finish", ["tool-calls", "content-filter", "length"])
def test_native_terminal_reason_distinguishes_failure_from_model_budget(setup, tmp_path, finish):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    launch = sandbox.launch.side_effect
    denial = "The user rejected permission to use this specific tool call."

    async def interrupted(**kwargs):
        result = await launch(**kwargs)
        path = f"{sandbox.directory}/export.json"
        export = json.loads(sandbox.files[path])
        message = export["messages"][-1]
        message["info"]["finish"] = finish
        if finish == "tool-calls":
            message["parts"].append(
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {"status": "error", "input": {"filePath": "/"}, "error": denial},
                }
            )
        sandbox.files[path] = json.dumps(export)
        with sqlite3.connect(tmp_path / "session.db") as con:
            con.execute("update message set data=? where id='a0'", (json.dumps(message["info"]),))
            if finish == "tool-calls":
                con.execute(
                    "insert into part values('denied', 'a0', 'native-session', ?, 20)",
                    (json.dumps(message["parts"][-1]),),
                )
        sandbox.files[f"{sandbox.directory}/stdout.jsonl"] = json.dumps(
            {"type": "step_finish", "part": {"reason": finish}}
        )
        return result

    sandbox.launch.side_effect = interrupted
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        body = {
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
            "activation_id": 0,
            "responses_create_params": {"input": "Inspect the task"},
        }
        response = client.post("/v1/agent_sessions/activate", json=body)
        if finish == "length":
            assert response.status_code == 200, response.text
            assert response.json()["response"]["status"] == "incomplete"
            assert response.json()["stop_reason"] == "model_budget_exhausted"
        else:
            assert response.status_code == 502, response.text
            assert "terminal assistant result" in response.json()["detail"]
            if finish == "tool-calls":
                assert denial in response.json()["detail"]
            assert client.post("/v1/agent_sessions/activate", json=body).status_code == 502
            assert sandbox.launch.await_count == 1
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": request.agent_session_id, "episode_id": request.episode_id.model_dump()},
        )
        assert close.status_code == 200, close.text
        assert close.json()["cleanup_confirmed"] is True
        records = close.json()["agent_observations"]["records"]
        root = next(record for record in records if record["kind"] == "agent_invocation")
        assert root["status"] == ("incomplete" if finish == "length" else "failed")
        if finish == "tool-calls":
            assert denial in json.dumps(records)


async def test_ripgrep_setup_uses_private_native_cache_and_records_provenance(setup):
    agent, sandbox = setup
    agent.config.prefetched_ripgrep_url = "https://runtime.example/rg.tar.gz"
    agent.config.prefetched_ripgrep_sha256 = "a" * 64
    agent.config.ripgrep_version = "15.1.0"
    execute = sandbox.exec.side_effect

    async def install(command, **kwargs):
        result = await execute(command, **kwargs)
        if "install_ripgrep.py" in command:
            import shlex

            path = shlex.split(command)[-1]
            return SimpleNamespace(
                return_code=0,
                error_type=None,
                stderr="",
                stdout=json.dumps({"path": path, "version": "ripgrep 15.1.0", "source": "prefetched"}),
            )
        return result

    sandbox.exec.side_effect = install
    state = await agent._seed_agent_session_state(seed())
    assert state.ripgrep_info["path"] == f"{state.persistent_directory}/cache/opencode/bin/rg"
    assert state.ripgrep_info["path"].startswith("/tmp/nemo-gym-opencode-sessions/")
    await state.close(1)


@pytest.mark.parametrize(
    "url,digest,version",
    [
        ("file:///rg", "a" * 64, "15.1.0"),
        ("https://example/rg", None, "15.1.0"),
        ("https://example/rg", "a" * 64, None),
    ],
)
async def test_invalid_ripgrep_pin_fails_before_connect(setup, url, digest, version):
    from fastapi import HTTPException

    agent, sandbox = setup
    agent.config.prefetched_ripgrep_url = url
    agent.config.prefetched_ripgrep_sha256 = digest
    agent.config.ripgrep_version = version
    with pytest.raises(HTTPException) as error:
        await agent._seed_agent_session_state(seed())
    assert error.value.status_code == 422
    sandbox.exec.assert_not_awaited()


async def test_resume_requires_confirmed_prior_cleanup(setup):
    agent, sandbox = setup
    state = await agent._seed_agent_session_state(seed())
    state.session.launch_started = True
    with pytest.raises(RuntimeError, match="confirmed previous activation cleanup"):
        await state.prepare_activation(1)
    assert sandbox.launch.call_count == 0


@pytest.mark.parametrize("cap,expected", [(None, None), (32000, "32000"), ("model_limit", "4096")])
def test_native_catalog_and_output_cap_are_independent(setup, tmp_path, cap, expected):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    agent.config.native_model_catalog = "native"
    agent.config.native_model_options = {"reasoning": True}
    agent.config.native_output_token_max = cap
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        response = client.post(
            "/v1/agent_sessions/activate",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
                "activation_id": 0,
                "responses_create_params": {"input": "Check native settings"},
            },
        )
        assert response.status_code == 200, response.text
        env = payloads[0]["env"]
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert config["provider"]["nemo_gym"]["models"]["dummy_model"] == {"reasoning": True}
        assert env.get("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX") == expected
        metadata = response.json()["response"]["metadata"]
        assert metadata["opencode_output_token_max"] == (expected or "native_default")
        assert metadata["opencode_model_catalog"] == "native"
        assert (
            client.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": request.agent_session_id, "episode_id": request.episode_id.model_dump()},
            ).status_code
            == 200
        )


def test_runtime_policy_is_translated_per_session_without_changing_global_config(setup, tmp_path):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    agent.config.opencode_config = {"permission": {"external_directory": {"/workspace/**": "allow"}}}
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    body = request.model_dump(mode="json")
    body["runtime_policy"] = {
        "format": "harbor.agent-kwargs.v1",
        "settings": {"disallowed_tools": " WebFetch, WebSearch , ,"},
    }
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 200
        response = client.post(
            "/v1/agent_sessions/activate",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
                "activation_id": 0,
                "responses_create_params": {"input": "Complete the task"},
            },
        )
        assert response.status_code == 200, response.text
        config = json.loads(payloads[0]["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["permission"] == {
            "external_directory": {"/workspace/**": "allow"},
            "tools": {"webfetch": "deny", "websearch": "deny"},
        }
        closed = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": request.agent_session_id, "episode_id": request.episode_id.model_dump()},
        )
        assert closed.status_code == 200
        assert agent.config.opencode_config == {"permission": {"external_directory": {"/workspace/**": "allow"}}}
        # A following task with no policy inherits the composition, not its predecessor's settings.
        assert agent._native_task_config(None) == agent.config.opencode_config
        assert body["runtime_policy"]["settings"]["disallowed_tools"] == " WebFetch, WebSearch , ,"


@pytest.mark.parametrize(
    "policy",
    [
        {"format": "unknown.v1", "settings": {}},
        {"format": "harbor.agent-kwargs.v1", "settings": {"unknown": True}},
        {"format": "harbor.agent-kwargs.v1", "settings": {"disallowed_tools": ["webfetch"]}},
    ],
)
def test_unsupported_runtime_policy_is_rejected_before_sandbox_setup(setup, policy):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json={**seed().model_dump(mode="json"), "runtime_policy": policy})
        assert response.status_code == 422, response.text
        sandbox.exec.assert_not_awaited()
        assert not agent._session_records


def test_event_projection_preserves_failures_tool_details_and_reasoning():
    raw = "\n".join(
        json.dumps(event)
        for event in (
            {"type": "reasoning", "part": {"text": "hidden"}},
            {
                "type": "tool_use",
                "part": {
                    "tool": "bash",
                    "callID": "call1",
                    "state": {
                        "status": "error",
                        "input": {"command": "false"},
                        "error": "failed",
                        "metadata": {"exit": 7},
                    },
                },
            },
            {"type": "error", "error": {"name": "APIError", "data": {"message": "upstream error"}}},
            {"type": "text", "part": {"text": "visible"}},
        )
    )
    events = parse_activation_events("diagnostic\n[]\n" + raw)
    assert [event.sequence for event in events] == list(range(4))
    assert events[0].kind == "reasoning"
    assert events[1].result == "failed"
    assert events[1].arguments == {"command": "false"}
    assert events[2].metadata["error"]["name"] == "APIError"
    assert events[3].text == "visible"


def test_prefetched_binary_requires_matching_digest_before_connect(setup, tmp_path):
    agent, sandbox = setup
    binary = tmp_path / "opencode"
    binary.write_bytes(b"fixture binary")
    agent.config.local_opencode_binary_path = str(binary)
    agent.config.local_opencode_binary_sha256 = "0" * 64
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert result.status_code == 422
        assert "SHA-256 mismatch" in result.text
    sandbox.exec.assert_not_called()


async def test_prefetched_binary_upload_and_profile_reach_runner(setup, tmp_path):
    agent, sandbox = setup
    binary = tmp_path / "opencode"
    binary.write_bytes(b"fixture binary")
    agent.config.local_opencode_binary_path = str(binary)
    agent.config.local_opencode_binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    original_upload = sandbox.upload
    uploaded_binary = []

    async def upload(source, destination):
        if "opencode-prefetched.gz.part-" in destination:
            uploaded_binary.append(gzip.decompress(Path(source).read_bytes()))
        else:
            await original_upload(source, destination)

    sandbox.upload = upload
    state = await agent._seed_agent_session_state(seed())
    assert uploaded_binary == [b"fixture binary"]
    assert any("Uploaded OpenCode binary SHA-256 mismatch" in call.args[0] for call in sandbox.exec.await_args_list)
    await state.close(1)


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


def test_session_wall_budget_clamps_execution_and_refuses_expired_resume(setup, tmp_path):
    agent, sandbox = setup
    agent.config.timeout = 300
    agent.config.session_execution_timeout_seconds = 60
    payloads = install_artifact_runner(sandbox, tmp_path)
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        body = {
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
            "activation_id": 0,
            "responses_create_params": {"input": "first"},
        }
        result = client.post("/v1/agent_sessions/activate", json=body)
        assert result.status_code == 200, result.text
        command = sandbox.launch.call_args.kwargs["command"]
        import shlex

        args = shlex.split(command)
        assert 0 < float(args[args.index("--timeout") + 1]) <= 60
        state = agent._session_records[request.agent_session_id].state
        state.execution_started_at -= 61
        expired = client.post("/v1/agent_sessions/activate", json={**body, "activation_id": 1})
        assert expired.status_code == 200, expired.text
        receipt = expired.json()
        assert receipt["response"]["status"] == "incomplete"
        assert receipt["stop_reason"] == "session_budget_exhausted"
        assert receipt["response"]["output"] == []
        assert receipt["observation"]["harness_steps"] == 0
        assert len(payloads) == 1
        closed = client.post(
            "/v1/agent_sessions/close",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
            },
        )
        assert closed.json()["cleanup_confirmed"] is True


async def test_prefetched_url_is_pinned_and_installed_outside_task(setup):
    agent, sandbox = setup
    agent.config.prefetched_opencode_binary_url = "http://runtime.example/runtime/sha256/opencode.gz"
    agent.config.prefetched_opencode_binary_sha256 = "a" * 64
    state = await agent._seed_agent_session_state(seed())
    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    assert any(
        "urllib.request.urlopen" in command and "http://runtime.example/runtime/sha256/opencode.gz" in command
        for command in commands
    )
    assert sum("Uploaded OpenCode binary SHA-256 mismatch" in command for command in commands) == 2
    assert all("/workspace/" not in command for command in commands)
    await state.close(1)


@pytest.mark.parametrize(
    "url,digest", [("file:///tmp/runtime.gz", "a" * 64), ("https://runtime.example/runtime.gz", None)]
)
async def test_invalid_prefetched_url_fails_before_sandbox_access(setup, url, digest):
    agent, sandbox = setup
    agent.config.prefetched_opencode_binary_url = url
    agent.config.prefetched_opencode_binary_sha256 = digest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        await agent._seed_agent_session_state(seed())
    assert error.value.status_code == 422
    sandbox.exec.assert_not_called()


@pytest.mark.asyncio
async def test_bootstrapped_python_is_used_for_setup_supervisor_and_snapshot(setup, monkeypatch):
    from unittest.mock import AsyncMock

    from responses_api_agents.opencode_agent import app

    agent, sandbox = setup
    python = "/tmp/nemo-gym-python-pinned/python/bin/python3"
    bootstrap = AsyncMock(return_value=python)
    monkeypatch.setattr(app, "ensure_python", bootstrap)
    agent.config.python_runtime_url = "https://runtime.example/python.tar.gz"
    agent.config.python_runtime_sha256 = "b" * 64
    state = await agent._seed_agent_session_state(seed())
    bootstrap.assert_awaited_once_with(
        sandbox,
        runtime_url=agent.config.python_runtime_url,
        runtime_sha256=agent.config.python_runtime_sha256,
        timeout_s=agent.config.setup_timeout,
    )
    assert any(command.args[0].startswith(python + " -I -c ") for command in sandbox.exec.await_args_list)
    installer = next(command.args[0] for command in sandbox.exec.await_args_list if "bash " in command.args[0])
    assert installer.endswith(python)
    await state.prepare_activation(0)
    command = await state.stage_activation({"prompt": "Continue"})
    assert command.python == python
    assert command.argv[0] == python
    state.session.cleanup = sandbox.result
    await state.snapshot(1)
    assert sandbox.exec.await_args.args[0].startswith(python + " -I ")
    await state.close(1)


def test_completed_database_turn_does_not_fabricate_missing_stdout_events(setup, tmp_path):
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    launch = sandbox.launch.side_effect

    async def omit_terminal_event(**kwargs):
        result = await launch(**kwargs)
        path = f"{sandbox.directory}/stdout.jsonl"
        sandbox.files[path] = "\n".join(sandbox.files[path].splitlines()[:-1])
        return result

    sandbox.launch.side_effect = omit_terminal_event
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        body = {
            "agent_session_id": request.agent_session_id,
            "episode_id": request.episode_id.model_dump(),
            "activation_id": 0,
            "responses_create_params": {"input": "Implement a change"},
        }
        response = client.post("/v1/agent_sessions/activate", json=body)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["response"]["status"] == "completed"
        assert all(event["kind"] != "step_finish" for event in result["observation"]["events"])
        gaps = {gap["code"] for gap in result["observation"]["agent_observations"]["gaps"]}
        assert {"native_event_stream_terminal_missing", "model_usage_reconciliation_unavailable"} <= gaps
        sandbox.launch.side_effect = launch
        assert client.post("/v1/agent_sessions/activate", json={**body, "activation_id": 1}).status_code == 200
        close = client.post(
            "/v1/agent_sessions/close", json={key: body[key] for key in ("agent_session_id", "episode_id")}
        )
        assert close.status_code == 200
        assert "native_event_stream_terminal_missing" in {
            gap["code"] for gap in close.json()["agent_observations"]["gaps"]
        }


def test_native_profile_can_preserve_auxiliary_title_project_and_environment_defaults(setup, tmp_path):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    agent.config.native_env = {"OPENCODE_FAKE_VCS": "git"}
    agent.config.native_session_title = None
    agent.config.native_auxiliary_model = "native_default"
    agent.config.native_load_project_config = True
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        response = client.post(
            "/v1/agent_sessions/activate",
            json={
                "agent_session_id": request.agent_session_id,
                "episode_id": request.episode_id.model_dump(),
                "activation_id": 0,
                "responses_create_params": {"input": "Implement the task"},
            },
        )
        assert response.status_code == 200, response.text
        payload = payloads[0]
        assert "--title" not in payload["command"]
        assert "small_model" not in json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert payload["env"]["OPENCODE_DISABLE_PROJECT_CONFIG"] == "false"
        assert payload["env"]["OPENCODE_FAKE_VCS"] == "git"
        assert (
            client.post(
                "/v1/agent_sessions/close",
                json={
                    "agent_session_id": request.agent_session_id,
                    "episode_id": request.episode_id.model_dump(),
                },
            ).status_code
            == 200
        )


@pytest.mark.parametrize("env", [{"HOME": "/task"}, {"OPENCODE_CONFIG_CONTENT": "{}"}, {"BAD-KEY": "value"}])
async def test_native_environment_cannot_override_runtime_contract(setup, env):
    agent, sandbox = setup
    agent.config.native_env = env
    from fastapi import HTTPException

    with pytest.raises(HTTPException, match="environment"):
        await agent._seed_agent_session_state(seed())
    sandbox.exec.assert_not_awaited()
