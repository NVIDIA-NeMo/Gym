# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants in a borrowed-sandbox OpenClaw session."""

import json
import re
import shlex
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.openclaw_agent.app import _mcp_probe_problem
from responses_api_agents.openclaw_agent.tests.test_sandbox import close_body, seed, setup  # noqa: F401


def grant(name: str, *, required: bool) -> dict:
    return {
        "kind": "mcp",
        "name": name,
        "required": required,
        "connection": {"url": f"http://resources.example:8000/{name}/mcp", "headers": {"x-session": "token"}},
    }


# What a 502 or gap says: classifications only, never probe output text.
_PROBLEM = {
    "diagnostic": "connection failed (http://resources.example:8000)",
    "exit": "exit 1",
    "not-listed": "not connected",
    "unreadable": "unreadable probe output",
    "timeout": "timeout: probe exceeded 120 s",
}


def probe_result(name: str, failing: set[str], failure: str) -> dict:
    """What the probe of `name` reports: stdout, stderr, exit code and whether its supervisor timed it out."""
    report = {"servers": {name: {"launch": "streamable-http", "tools": 4}}, "tools": [], "diagnostics": []}
    result = {"stderr": "", "return_code": 0, "timed_out": False}
    if name in failing and failure == "exit":
        return {**result, "stdout": "", "stderr": "No MCP server named x", "return_code": 1}
    if name in failing and failure == "timeout":
        return {**result, "stdout": "", "return_code": None, "timed_out": True}
    if name in failing and failure == "diagnostic":
        # OpenClaw 2026.6.11 probe --json exits 0 when the connection fails and reports it as a diagnostic.
        report = {"servers": {}, "tools": [], "diagnostics": [{"serverName": name, "message": "refused"}]}
    if name in failing and failure == "not-listed":
        report["servers"] = {}
    stdout = "probe log line\n" + json.dumps(report, indent=2)
    if name in failing and failure == "unreadable":
        stdout = "not json"
    return {**result, "stdout": stdout}


def read_params(text: str) -> dict:
    """Parse the harness runner's params.sh: scalars, arrays, and the env array back into a mapping."""
    params = {}
    for line in text.splitlines():
        name, _, value = line.partition("=")
        params[name] = shlex.split(value[1:-1]) if value.startswith("(") else shlex.split(value)[0]
    params["env"] = dict(word.split("=", 1) for word in params["env"])
    return params


def install_probe_fake(sandbox, failing: set[str], failure: str) -> list[dict]:
    """Answer each supervised `openclaw mcp probe` launch the way its supervisor and runner would.

    The probe writes stdout.log/stderr.log in its own directory and its supervisor writes cleanup.json there.
    """
    probes = []
    run_exec = sandbox.run_exec

    async def probe_or_run(command, **kwargs):
        match = re.search(r"(/\S*/mcp-probe-\d+)/cleanup\.json", command)
        if match and "process_supervisor" in command:
            directory = match.group(1)
            runner_input = read_params(sandbox.files.pop(f"{directory}/params.sh"))
            name = runner_input["command"][4]
            config = next(text for path, text in sandbox.files.items() if path.endswith("/.openclaw/openclaw.json"))
            probes.append({"name": name, "config": json.loads(config), "input": runner_input, "launch": command})
            result = probe_result(name, failing, failure)
            sandbox.files[f"{directory}/stdout.log"] = result["stdout"]
            sandbox.files[f"{directory}/stderr.log"] = result["stderr"]
            sandbox.files[f"{directory}/cleanup.json"] = json.dumps(
                {
                    "cleanup_confirmed": True,
                    "return_code": result["return_code"],
                    "timed_out": result["timed_out"],
                    "error": None,
                }
            )
            return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")
        argv = shlex.split(command)
        if argv[2:4] == ["mcp", "probe"]:
            # The probe as an unsupervised exec, before supervised probes; kept so these tests also run on it.
            config = next(text for path, text in sandbox.files.items() if path.endswith("/.openclaw/openclaw.json"))
            probes.append({"name": argv[4], "config": json.loads(config), "input": kwargs, "launch": command})
            result = probe_result(argv[4], failing, failure)
            if result["timed_out"]:
                return SimpleNamespace(error_type="timeout", return_code=-1, stdout=None, stderr="timed out")
            return SimpleNamespace(error_type=None, **{k: result[k] for k in ("return_code", "stdout", "stderr")})
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = probe_or_run
    return probes


def skipped(closed) -> list[str]:
    return [
        gap["detail"] for gap in closed.json()["agent_observations"]["gaps"] if gap["code"] == "mcp_server_skipped"
    ]


@pytest.mark.parametrize("failure", list(_PROBLEM))
@pytest.mark.parametrize("unreachable", [None, "task_tools", "extra"])
def test_mcp_grants_are_probed_before_the_agent_runs(setup, unreachable, failure):  # noqa: F811
    agent, sandbox = setup
    agent.config.mcp_tool_timeout_seconds = 900
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True), grant("extra", required=False)]
    probes = install_probe_fake(sandbox, {unreachable} - {None}, failure)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        directory = agent._session_records[session_id].state.session.session_dir
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert [probe["name"] for probe in probes] == ["task_tools", "extra"]
        # The probe runs with plugins off, as the agent run does.
        assert probes[0]["config"]["plugins"] == {"enabled": False}
        assert probes[0]["config"]["mcp"]["servers"]["task_tools"] == {
            "url": "http://resources.example:8000/task_tools/mcp",
            "transport": "streamable-http",
            "headers": {"x-session": "token"},
            "timeout": 900,
        }
        assert probes[0]["input"]["env"]["OPENCLAW_CONFIG_PATH"] == f"{directory}/home/.openclaw/openclaw.json"
        # Each probe runs under its own supervisor and receipt, in its own directory, bounded at 120 s.
        for index, probe in enumerate(probes):
            assert f"--receipt {directory}/mcp-probe-{index}/cleanup.json" in probe["launch"]
            assert "--timeout 120" in probe["launch"]
            assert probe["input"]["command"][2:] == ["mcp", "probe", probe["name"], "--json"]
        if unreachable == "task_tools":
            assert result.status_code == 502
            assert result.json()["detail"] == (
                "Required Gym MCP tools could not be initialized: task_tools: " + _PROBLEM[failure]
            )
            # The model is never called: the agent run does not start.
            sandbox.launch.assert_not_awaited()
        else:
            assert result.status_code == 200, result.text
            assert result.json()["status"] == "completed"
            config = json.loads(sandbox.files[f"{directory}/home/.openclaw/openclaw.json"])
            expected = {"task_tools"} if unreachable == "extra" else {"task_tools", "extra"}
            assert set(config["mcp"]["servers"]) == expected
            assert config["tools"]["allow"] == ["read", "write", "edit", "exec", "process", "bundle-mcp"]
            sandbox.launch.assert_awaited_once()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    assert skipped(closed) == (["extra: " + _PROBLEM[failure]] if unreachable == "extra" else [])
    # The borrowed sandbox belongs to the resources server: close releases it without stopping it.
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("failure", ["diagnostic", "timeout"])
@pytest.mark.parametrize("required_first", [True, False])
def test_required_failure_keeps_the_optional_failure_gap(setup, required_first, failure):  # noqa: F811
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    grants = [grant("task_tools", required=True), grant("extra", required=False)]
    body["tool_accesses"] = grants if required_first else grants[::-1]
    probes = install_probe_fake(sandbox, {"task_tools", "extra"}, failure)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert len(probes) == 2
        assert result.status_code == 502
        assert result.json()["detail"] == (
            "Required Gym MCP tools could not be initialized: task_tools: " + _PROBLEM[failure]
        )
        sandbox.launch.assert_not_awaited()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    codes = [gap["code"] for gap in closed.json()["agent_observations"]["gaps"]]
    assert skipped(closed) == ["extra: " + _PROBLEM[failure]]
    assert codes.count("agent_activation_interrupted") == 1
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


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


@pytest.mark.parametrize("lost", ["task_tools", "extra", "both"])
@pytest.mark.parametrize("required_first", [True, False])
def test_server_that_fails_at_agent_startup_after_its_probe(setup, lost, required_first):  # noqa: F811
    """OpenClaw logs a server that fails at agent startup, runs on without its tools and exits 0."""
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    grants = [grant("task_tools", required=True), grant("extra", required=False)]
    body["tool_accesses"] = grants if required_first else grants[::-1]
    install_probe_fake(sandbox, set(), "diagnostic")
    names = ["task_tools", "extra"] if lost == "both" else [lost]
    wait_exit = sandbox.wait_exit

    async def run_with_lost_servers():
        result = await wait_exit()
        sandbox.files[f"{sandbox.directory}/stderr.log"] = "".join(
            f'\x1b[33m[bundle-mcp]\x1b[39m bundle-mcp: failed to start server "{name}" (http://127.0.0.1:1/mcp)\n'
            for name in names
        )
        return result

    sandbox.wait_exit = run_with_lost_servers
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        sandbox.launch.assert_awaited_once()
        if "task_tools" in names:
            assert result.status_code == 502
            assert result.json()["detail"] == "Required Gym MCP tools did not connect in the agent run: task_tools"
        else:
            assert result.status_code == 200 and result.json()["status"] == "completed", result.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    expected = ["extra: did not connect in the agent run"] if "extra" in names else []
    assert skipped(closed) == expected
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_probe_problem_keeps_kind_and_origin_only(setup):  # noqa: F811
    """A 502 or gap never carries server bodies, credentials, query values or URL userinfo."""
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True)]
    body["tool_accesses"][0]["connection"]["url"] = (
        "http://user:pw@resources.example:8000/task_tools/mcp?token=q"  # pragma: allowlist secret
    )
    message = (
        "Streamable HTTP error: Error POSTing to endpoint (HTTP 401) "
        "http://user:pw@resources.example:8000/task_tools/mcp?token=q-secret "  # pragma: allowlist secret
        'Authorization: Bearer b-secret x-session=token {"error": "body-secret"} plaintext-private-server-response'
    )
    probes = install_probe_fake(sandbox, {"task_tools"}, "diagnostic")
    fake = sandbox.exec.side_effect

    async def leaky_diagnostic(command, **kwargs):
        result = await fake(command, **kwargs)
        for path in [path for path in sandbox.files if path.endswith("/stdout.log") and "/mcp-probe-" in path]:
            report = {"servers": {}, "diagnostics": [{"serverName": "task_tools", "message": message}]}
            sandbox.files[path] = json.dumps(report)
        return result

    sandbox.exec.side_effect = leaky_diagnostic
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
    assert probes and result.status_code == 502
    assert result.json()["detail"] == (
        "Required Gym MCP tools could not be initialized: task_tools: HTTP 401 (http://resources.example:8000)"
    )


@pytest.mark.parametrize(
    "result, expected",
    [
        # A nonzero exit whose output is a plain-text server body.
        (
            SandboxExecResult(stdout="", stderr="HTTP 401\nplain-private-response-body", return_code=1),
            "exit 1: HTTP 401",
        ),
        (SandboxExecResult(stdout="", stderr="plain-private-response-body", return_code=1), "exit 1"),
        # A successful probe whose diagnostic is a plain-text server body.
        (
            SandboxExecResult(
                stdout=json.dumps(
                    {
                        "servers": {},
                        "diagnostics": [
                            {"serverName": "task_tools", "message": "HTTP 401: plaintext-private-server-response"}
                        ],
                    }
                ),
                stderr="",
                return_code=0,
            ),
            "HTTP 401",
        ),
        # Bootstrap failure before OpenClaw starts (no Node), from output.log.
        (
            SandboxExecResult(
                stdout="", stderr="FileNotFoundError: [Errno 2] No such file or directory: '/r/node'", return_code=1
            ),
            "exit 1: executable not found",
        ),
        (SandboxExecResult(stdout="private text", stderr="", return_code=0), "unreadable probe output"),
    ],
)
@pytest.mark.parametrize("url", [None, "http://u:p@resources.example:8000/x/mcp?k=v"])
def test_probe_problem_never_returns_probe_output(result, expected, url):
    origin = " (http://resources.example:8000)" if url and "HTTP" in expected else ""
    problem = _mcp_probe_problem("task_tools", result, url=url)
    assert problem == expected + origin
    for private in ("private", "u:p", "/x/mcp", "k=v", "/r/node"):
        assert private not in problem
