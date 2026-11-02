# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants in a borrowed-sandbox OpenCode session."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_runtime import MCPToolAccess
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import close_body, seed, setup  # noqa: F401


def read_params(text: str) -> dict:
    """Parse the harness runner's params.sh: scalars, arrays, and the env array back into a mapping."""
    lexer = shlex.shlex(text, posix=True, punctuation_chars="()")
    lexer.whitespace_split = True
    tokens, params = list(lexer), {}
    while tokens:
        name, _, value = tokens.pop(0).partition("=")
        if not value and tokens and tokens[0] == "(":
            end = tokens.index(")")
            params[name], tokens = tokens[1:end], tokens[end + 1 :]
        else:
            params[name] = value
    params["env"] = dict(word.split("=", 1) for word in params["env"])
    return params


def grant(name: str, *, required: bool) -> dict:
    return {
        "kind": "mcp",
        "name": name,
        "required": required,
        "connection": {"url": f"http://resources.example:8000/{name}/mcp", "headers": {"x-session": "token"}},
    }


_UNCONFIRMED = "Required Gym MCP tools were not confirmed before the model call: task_tools"


@pytest.mark.parametrize(
    "scenario", ["connected", "required-failed", "optional-failed", "never-checked", "missing", "malformed", "list"]
)
def test_mcp_grants_are_configured_and_checked(setup, scenario):  # noqa: F811
    agent, sandbox = setup
    agent.config.mcp_tool_timeout_seconds = 900
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True), grant("extra", required=False)]
    statuses = {"task_tools": {"status": "connected"}, "extra": {"status": "connected"}}
    if scenario == "required-failed":
        statuses["task_tools"] = {"status": "failed", "error": "connection refused"}
    elif scenario == "optional-failed":
        statuses["extra"] = {"status": "failed", "error": "connection refused"}
    wait_exit = sandbox.wait_exit

    async def run_plugin_then_exit():
        # The adapter stages a null status; required_mcp.mjs replaces it before the first model call.
        assert json.loads(sandbox.files[f"{sandbox.directory}/mcp-status.json"]) is None
        # Stand-ins for a plugin that did not load (null or missing status) or wrote something else.
        status_path = f"{sandbox.directory}/mcp-status.json"
        if scenario == "missing":
            del sandbox.files[status_path]
        elif scenario == "malformed":
            sandbox.files[status_path] = "{not json"
        elif scenario == "list":
            sandbox.files[status_path] = json.dumps(["task_tools"])
        elif scenario != "never-checked":
            sandbox.files[status_path] = json.dumps(statuses)
        return await wait_exit()

    sandbox.wait_exit = run_plugin_then_exit
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        payload = read_params(sandbox.files[f"{sandbox.directory}/params.sh"])
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["mcp"]["task_tools"] == {
            "type": "remote",
            "url": "http://resources.example:8000/task_tools/mcp",
            "enabled": True,
            "oauth": False,
            "headers": {"x-session": "token"},
            "timeout": 900000,
        }
        assert set(config["mcp"]) == {"task_tools", "extra"}
        assert config["plugin"][-1] == f"file://{sandbox.directory}/required_mcp.mjs"
        assert (
            sandbox.files[f"{sandbox.directory}/required_mcp.mjs"]
            == (Path(__file__).parents[1] / "required_mcp.mjs").read_text()
        )
        assert json.loads(sandbox.files[f"{sandbox.directory}/mcp-grants.json"]) == {
            "required": ["task_tools"],
            "optional": ["extra"],
        }
        if scenario == "required-failed":
            assert result.status_code == 502
            assert result.json()["detail"] == (
                "Required Gym MCP tools could not be initialized: task_tools: failed (connection refused)"
            )
        elif scenario in ("never-checked", "missing", "malformed", "list"):
            # The model answered, but nothing confirms the required tools were there: fail closed.
            assert result.status_code == 502
            assert result.json()["detail"].startswith(_UNCONFIRMED)
        else:
            assert result.status_code == 200, result.text
            assert result.json()["status"] == "completed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    skipped = [gap for gap in closed.json()["agent_observations"]["gaps"] if gap["code"] == "mcp_server_skipped"]
    assert skipped == (
        [{"code": "mcp_server_skipped", "invocation_id": None, "detail": "extra: failed (connection refused)"}]
        if scenario == "optional-failed"
        else []
    )
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert ("mcp_status_unavailable" in gaps) == (scenario in ("never-checked", "missing", "malformed", "list"))
    # The borrowed sandbox belongs to the resources server: close releases it without stopping it.
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("cause", ["timeout", "model-error"])
def test_unconfirmed_status_keeps_an_earlier_failure_cause(setup, cause):  # noqa: F811
    """A run that ends before the plugin records a status keeps its own outcome instead of the MCP error."""
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True)]
    export = json.loads(sandbox.events)
    if cause == "timeout":
        # Killed before the first model call: no assistant message, the status stays null.
        export["messages"] = export["messages"][:1]
        sandbox.result["timed_out"] = True
        sandbox.result["return_code"] = -9
    else:
        export["messages"][-1]["info"]["error"] = {"message": "model rejected request"}
    sandbox.events = json.dumps(export)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 502
        expected = "OpenCode produced no assistant result" if cause == "timeout" else "model rejected request"
        assert expected in result.json()["detail"]
        assert _UNCONFIRMED not in result.json()["detail"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    sandbox.disconnect.assert_awaited_once()


def test_configured_and_request_grants_are_both_kept(setup):  # noqa: F811
    """A grant declared in the agent config and one sent with the session both reach OpenCode."""
    agent, sandbox = setup
    agent.config.tool_accesses = [MCPToolAccess.model_validate(grant("configured", required=False))]
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True)]
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        payload = read_params(sandbox.files[f"{sandbox.directory}/params.sh"])
        assert set(json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])["mcp"]) == {"configured", "task_tools"}
        assert json.loads(sandbox.files[f"{sandbox.directory}/mcp-grants.json"]) == {
            "required": ["task_tools"],
            "optional": ["configured"],
        }
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))


def test_static_mcp_servers_in_opencode_config_are_rejected(setup):  # noqa: F811
    """Session opencode_config allows only permission and tools, so no static MCP server can be overwritten."""
    agent, sandbox = setup
    agent.config.opencode_config = {"mcp": {"task_tools": {"type": "remote", "url": "http://other"}}}
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True)]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=body)
    assert response.status_code == 422
    assert "permission/tools only" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()


def test_required_non_mcp_grant_is_still_rejected(setup):  # noqa: F811
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [
        grant("task_tools", required=True),
        {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True},
    ]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=body)
    assert response.status_code == 422
    assert response.json()["detail"].endswith("required grants it cannot use: tools")
    sandbox.exec.assert_not_awaited()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
@pytest.mark.parametrize(
    "server_status, expected",
    [
        ({"task_tools": {"status": "connected"}, "extra": {"status": "failed", "error": "refused"}}, None),
        ({"task_tools": {"status": "failed", "error": "refused"}}, "task_tools"),
        (None, "task_tools"),
    ],
)
def test_plugin_checks_grants_before_the_model_call(tmp_path, server_status, expected):
    """Run required_mcp.mjs under Node with a stand-in for OpenCode's plugin client."""
    shutil.copy(Path(__file__).parents[1] / "required_mcp.mjs", tmp_path / "required_mcp.mjs")
    (tmp_path / "mcp-grants.json").write_text(json.dumps({"required": ["task_tools"], "optional": ["extra"]}))
    script = tmp_path / "check.mjs"
    script.write_text(
        'import { RequiredMcp } from "./required_mcp.mjs";\n'
        f"const status = {json.dumps(server_status)};\n"
        "const client = { mcp: { status: async () => (status ? { data: status } : { error: new Error('down') }) } };\n"
        "const hooks = await RequiredMcp({ client });\n"
        "for (const call of [1, 2]) {\n"
        "  try { await hooks['chat.params']({}, {}); console.log('ok'); }\n"
        "  catch (error) { console.log(error.message); }\n"
        "}\n"
    )
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True, timeout=60)
    expected_line = "ok" if expected is None else f"Required Gym MCP tools could not be initialized: {expected}"
    assert result.stdout.splitlines() == [expected_line, expected_line]
    recorded = json.loads((tmp_path / "mcp-status.json").read_text())
    assert recorded["task_tools"]["status"] == (
        "connected" if expected is None else ("failed" if server_status else "unavailable")
    )
    assert recorded["extra"]["status"] == ("failed" if expected is None else "unavailable")


@pytest.mark.parametrize("with_grant", [True, False])
@pytest.mark.parametrize("step", ["placeholder", "model-step"])
def test_timeout_during_mcp_connect_keeps_its_cause(setup, step, with_grant):  # noqa: F811
    """OpenCode saves an empty assistant message before it connects MCP; only a real model step needs the status."""
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    if with_grant:
        body["tool_accesses"] = [grant("task_tools", required=True)]
    export = json.loads(sandbox.events)
    if step == "placeholder":
        # Killed while MCP was still connecting: an assistant message with no parts and no finish.
        assistant = {key: value for key, value in export["messages"][-1]["info"].items() if key != "finish"}
        assistant["time"] = {"created": 100}
        assistant.pop("tokens", None)
        export["messages"] = [export["messages"][0], {"info": assistant, "parts": []}]
    sandbox.events = json.dumps(export)
    sandbox.result["timed_out"] = True
    sandbox.result["return_code"] = -9
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        if step == "model-step" and with_grant:
            # The model worked without confirmed required tools: fail closed.
            assert result.status_code == 502 and result.json()["detail"].startswith(_UNCONFIRMED)
        else:
            assert result.status_code == 200, result.text
            assert result.json()["status"] == "incomplete"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only file")
def test_plugin_replaces_a_staged_status_it_cannot_write(tmp_path):
    """The adapter's upload can leave mcp-status.json owned by another user; the plugin must replace, not write into it."""
    shutil.copy(Path(__file__).parents[1] / "required_mcp.mjs", tmp_path / "required_mcp.mjs")
    (tmp_path / "mcp-grants.json").write_text(json.dumps({"required": ["task_tools"], "optional": []}))
    staged = tmp_path / "mcp-status.json"
    staged.write_text("null")
    staged.chmod(0o444)  # Not writable by the plugin's user; the directory still is.
    script = tmp_path / "check.mjs"
    script.write_text(
        'import { RequiredMcp } from "./required_mcp.mjs";\n'
        "const client = { mcp: { status: async () => ({ data: { task_tools: { status: 'connected' } } }) } };\n"
        "const hooks = await RequiredMcp({ client });\n"
        "try { await hooks['chat.params']({}, {}); console.log('ok'); } catch (error) { console.log(error.code); }\n"
    )
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True, timeout=60)
    assert result.stdout.strip() == "ok"
    assert json.loads(staged.read_text()) == {"task_tools": {"status": "connected"}}
    assert sorted(path.name for path in tmp_path.glob("mcp-status.json*")) == ["mcp-status.json"]
