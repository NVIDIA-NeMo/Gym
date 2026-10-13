# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.opencode_agent.runtime import OBSERVABILITY_PATCH
from responses_api_agents.opencode_agent.sandbox import format_sandbox_error
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import (
    active_sessions,
    capture_observations,
    close_body,
    seed,
    setup,  # noqa: F401
)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_session_uploads_shared_plugin_only_with_model_capture(setup, enabled, thinking):
    agent, sandbox = setup
    agent.server_client.global_config_dict["observability_enabled"] = enabled
    agent.config.thinking = thinking
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        result.raise_for_status()
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        plugin = Path(sandbox.directory) / OBSERVABILITY_PATCH.name
        assert config.get("plugin", []) == ([plugin.as_uri()] if enabled else [])
        assert (str(plugin) in sandbox.files) is enabled
        if enabled:
            assert sandbox.files[str(plugin)] == OBSERVABILITY_PATCH.read_text()
        assert ("--thinking" in payload["command"]) is thinking
        assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["interleaved"] == {"field": "reasoning_content"}
        assert payload["env"]["OPENCODE_DISABLE_MODELS_FETCH"] == "true"
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()


def test_no_sandbox_or_provider_returns_422_before_connecting(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = None
    with TestClient(agent.setup_webserver()) as client:
        body = seed().model_copy(update={"sandbox_access": None})
        response = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
    assert response.status_code == 422
    assert "sandbox_access or a configured sandbox_provider" in response.text
    assert not active_sessions(agent)
    sandbox.exec.assert_not_awaited()


def test_request_allowlist_rejects_future_fields_but_accepts_defaults(setup):
    class ExtendedRequest(NeMoGymResponseCreateParamsNonStreaming):
        future_control: str | None = None

    agent, sandbox = setup
    assert agent._session_input(ExtendedRequest(input="task", stream=False, background=False)) == ("task", "")
    with pytest.raises(HTTPException, match="future_control") as error:
        agent._session_input(ExtendedRequest(input="task", future_control="unsupported"))
    assert error.value.status_code == 422
    with pytest.raises(HTTPException, match="metadata"):
        agent._session_input(ExtendedRequest(input="task", metadata={"ignored": "value"}))
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize(
    "setting,value", [("env", {"CUSTOM": "value"}), ("extra_args", ["--foo"]), ("command", "custom")]
)
def test_session_rejects_local_only_overrides(setup, setting, value):
    agent, sandbox = setup
    setattr(agent.config, setting, value)
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert "only by local OpenCode" in response.text
    sandbox.exec.assert_not_awaited()


def test_observation_parse_failure_preserves_response(setup, caplog):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        result.raise_for_status()
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
    assert "observation_capture_failed" in {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert "Failed to parse OpenCode observations" in caplog.text


@pytest.mark.parametrize("failure", ["provider_timeout", "missing_export", "snapshot"])
def test_execution_failure_keeps_cleanup_observations_and_diagnostics(setup, tmp_path, failure):
    agent, sandbox = setup
    capture_observations(sandbox, tmp_path)
    execute = sandbox.run_exec
    download = sandbox.download

    async def read(source, destination):
        if source.endswith("/stderr.log"):
            Path(destination).write_text("x" * 18000 + "ROOT CAUSE")
        elif source.endswith("/output.log"):
            Path(destination).write_text("output" * 6000 + "OUTPUT TAIL")
        elif failure == "missing_export" and source.endswith("/export.json"):
            raise FileNotFoundError("missing export")
        else:
            await download(source, destination)

    async def run(command, **kwargs):
        result = await execute(command, **kwargs)
        if failure == "provider_timeout" and "--receipt" in command:
            return SimpleNamespace(return_code=0, error_type="timeout", stdout="", stderr="deadline")
        if failure == "snapshot" and "--snapshot" in command:
            return SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="no root session")
        return result

    sandbox.download = read
    sandbox.exec.side_effect = run
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        session_id = created.json()["agent_session_id"]
        state = active_sessions(agent)[session_id]
        expected = TimeoutError if failure == "provider_timeout" else RuntimeError
        with pytest.raises(expected) as error:
            client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        if failure == "provider_timeout":
            assert state.session.artifacts == sandbox.events
            invocation = next(record for record in state.observations.records if record.kind == "agent_invocation")
            assert invocation.conversation[-1].content[0].text == "Fixed"
        else:
            assert "ROOT CAUSE" in str(error.value)
            assert len(str(error.value)) <= 16000
            if failure == "missing_export":
                assert "returned no valid result" in str(error.value)
                assert "OUTPUT TAIL" in str(error.value)
        assert len(state.stderr) == 16000
        assert state.session.cleanup["cleanup_confirmed"] is True
        assert any(record.kind == "sandbox" for record in state.observations.records)
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        closed.raise_for_status()
        assert closed.json()["agent_observations"]["records"]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_observation_parser_runs_off_event_loop(setup, tmp_path):
    from responses_api_agents.opencode_agent.sandbox import parse_opencode_observations

    agent, sandbox = setup
    capture_observations(sandbox, tmp_path)
    loop_thread = threading.get_ident()
    parser_threads = []
    from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest
    from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import activate

    def parse(*args, **kwargs):
        parser_threads.append(threading.get_ident())
        return parse_opencode_observations(*args, **kwargs)

    with patch("responses_api_agents.opencode_agent.sandbox.parse_opencode_observations", side_effect=parse) as parsed:
        request, session_id, task = await activate(agent, sandbox)
        await task
        parsed.assert_called_once()
        assert len(parser_threads) == 1 and parser_threads[0] != loop_thread
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize("artifact", ["[]", '{"messages":[null]}', '{"messages":{}}'])
def test_malformed_capture_never_replaces_provider_failure(setup, artifact):
    agent, sandbox = setup
    execute = sandbox.run_exec
    sandbox.events = artifact

    async def run(command, **kwargs):
        result = await execute(command, **kwargs)
        if "--receipt" in command:
            return SimpleNamespace(return_code=0, error_type="timeout", stdout="", stderr="provider timeout")
        return result

    sandbox.exec.side_effect = run
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        session_id = created.json()["agent_session_id"]
        state = active_sessions(agent)[session_id]
        with pytest.raises(TimeoutError):
            client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert any(record.kind == "sandbox" for record in state.observations.records)
        client.post("/v1/agent_sessions/close", json=close_body(session_id)).raise_for_status()


@pytest.mark.parametrize("stderr_already_in_error", [False, True])
def test_combined_diagnostics_are_bounded_and_stderr_is_not_duplicated(stderr_already_in_error):
    stderr = "e" * 16000 + "STDERR TAIL"
    message = "OpenCode failed" + (f": {stderr}" if stderr_already_in_error else "")
    result = format_sandbox_error(message, output="o" * 16000 + "OUTPUT TAIL", stderr=stderr)
    assert len(result) <= 16000
    assert result.startswith("OpenCode failed")
    assert "OUTPUT TAIL" in result
    assert result.count("STDERR TAIL") == 1


def test_model_error_502_uses_the_same_combined_diagnostic_bound(setup):
    agent, sandbox = setup
    artifact = json.loads(sandbox.events)
    assistant = next(message["info"] for message in artifact["messages"] if message["info"]["role"] == "assistant")
    assistant["error"] = {"name": "APIError", "message": "e" * 20000 + "MODEL ERROR TAIL"}
    sandbox.events = json.dumps(artifact)
    download = sandbox.download

    async def read(source, destination):
        if source.endswith("/stderr.log"):
            Path(destination).write_text("s" * 20000 + "STDERR TAIL")
        else:
            await download(source, destination)

    sandbox.download = read
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert len(detail) <= 16000
        assert "MODEL ERROR TAIL" in detail
        assert detail.count("STDERR TAIL") == 1
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()
