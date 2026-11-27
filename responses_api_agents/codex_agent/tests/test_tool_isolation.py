# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool commands must not be able to kill the harness: tool pid namespace and kill labels."""

import json
import tomllib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from responses_api_agents.codex_agent.sandbox import harness_kill_evidence
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import (  # noqa: F401 - fixtures
    close_body,
    events,
    seed,
    setup,
)
from responses_api_agents.codex_agent.tool_host import (
    TOOL_HOST_ENVIRONMENT_ID,
    environments_toml,
    setup_failure_hint,
    tool_host_argv,
    tool_host_setup_command,
)


SESSION = "/tmp/nemo-gym-codex-sessions/abc"


def test_tool_host_runs_in_its_own_pid_namespace_with_the_same_view():
    argv = tool_host_argv(SESSION)
    assert argv[0] == f"{SESSION}/bin/bwrap"
    separator = argv.index("--")
    options = argv[1:separator]
    for flag in ("--unshare-user", "--unshare-pid", "--as-pid-1", "--die-with-parent"):
        assert flag in options
    assert options[options.index("--dev-bind") : options.index("--dev-bind") + 3] == ["--dev-bind", "/", "/"]
    assert options[options.index("--proc") : options.index("--proc") + 2] == ["--proc", "/proc"]
    # Network, IPC and the filesystem are the harness's own.
    assert not {"--unshare-net", "--unshare-all", "--unshare-ipc", "--ro-bind", "--tmpfs"} & set(options)
    assert argv[separator + 1 :] == [f"{SESSION}/bin/nemo-gym-tool-host", "exec-server", "--listen", "stdio"]
    assert not any(word in argv[separator + 1] for word in ("node", "python"))


def test_environments_toml_routes_every_tool_through_the_tool_host():
    parsed = tomllib.loads(environments_toml(SESSION))
    assert parsed["default"] == TOOL_HOST_ENVIRONMENT_ID
    assert parsed["include_local"] is False
    (environment,) = parsed["environments"]
    assert environment == {
        "id": TOOL_HOST_ENVIRONMENT_ID,
        "program": tool_host_argv(SESSION)[0],
        "args": tool_host_argv(SESSION)[1:],
    }


def test_setup_command_probes_a_fresh_pid_namespace_before_the_episode():
    command = tool_host_setup_command(SESSION, "/tmp/runtime")
    assert "/tmp/runtime/codex/node_modules" in command
    assert f'ln -sfn "$host" {SESSION}/bin/nemo-gym-tool-host' in command
    assert "--setenv NEMO_GYM_TOOL_HOST_PROBE 1" in command
    assert command.rstrip().endswith("--version'")


def test_setup_failures_name_the_fix():
    assert "allow_nested_userns" in setup_failure_hint("bwrap: setting up uid map: Operation not permitted")
    assert setup_failure_hint("unrelated") == ""


def _seed_with_tool_namespace(agent, sandbox, tmp_path, *, setup_result=None):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\n")
    bwrap.chmod(0o755)
    agent.config.tool_pid_namespace = True
    agent.config.tool_pid_namespace_bwrap_path = str(bwrap)
    execute = sandbox.execute

    async def exec_with_setup(command, **kwargs):
        if "NEMO_GYM_TOOL_HOST_PROBE" in command:
            sandbox.setup_commands.append(command)
            return setup_result or SimpleNamespace(
                error_type=None, return_code=0, stdout="codex-cli 0.144.4", stderr=""
            )
        return await execute(command, **kwargs)

    sandbox.setup_commands = []
    sandbox.exec.side_effect = exec_with_setup
    return bwrap


def _activate(agent, *, raise_server_exceptions=True):
    """Seed a session and run one activation; returns the session id, the response and the close response."""
    with TestClient(agent.setup_webserver(), raise_server_exceptions=raise_server_exceptions) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        state = agent._session_records[session_id].state
        gaps = {gap.code: gap.detail for gap in state.observations.gaps} if state.observations else {}
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
    return result, gaps, closed


def test_seed_prepares_the_tool_namespace_and_codex_uses_it(setup, tmp_path):
    agent, sandbox = setup
    bwrap = _seed_with_tool_namespace(agent, sandbox, tmp_path)
    result, _, closed = _activate(agent)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "completed"
    directory = sandbox.directory
    assert sandbox.files[f"{directory}/bin/bwrap"] == bwrap.read_text()
    (setup_command,) = sandbox.setup_commands
    assert setup_command == tool_host_setup_command(directory, "/tmp/nemo-gym-codex-node-22.19.0-0.144.4")
    assert sandbox.files[f"{directory}/home/.codex/environments.toml"] == environments_toml(directory)
    config = tomllib.loads(sandbox.files[f"{directory}/home/.codex/config.toml"])
    assert config["orchestrator"] == {"skills": {"enabled": False}}
    assert closed.status_code == 200, closed.text


def test_without_the_flag_codex_keeps_its_local_environment(setup):
    agent, sandbox = setup
    result, _, _ = _activate(agent)
    assert result.status_code == 200, result.text
    assert f"{sandbox.directory}/home/.codex/environments.toml" not in sandbox.files
    assert f"{sandbox.directory}/bin/bwrap" not in sandbox.files
    assert "orchestrator" not in tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])


def test_seed_fails_loudly_when_the_tool_namespace_cannot_be_created(setup, tmp_path):
    agent, sandbox = setup
    failed = SimpleNamespace(
        error_type=None, return_code=125, stdout="", stderr="bwrap: setting up uid map: Operation not permitted"
    )
    _seed_with_tool_namespace(agent, sandbox, tmp_path, setup_result=failed)
    with TestClient(agent.setup_webserver()) as client:
        with pytest.raises(Exception) as raised:
            client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    causes, error = [], raised.value
    while error is not None:
        causes.append(str(error))
        error = error.__cause__
    assert any("tool pid namespace" in cause and "allow_nested_userns" in cause for cause in causes), causes
    assert not sandbox.launch.called


def test_tool_namespace_bwrap_defaults_to_the_bwrap_provider(setup, tmp_path):
    agent, _ = setup
    pinned, other = tmp_path / "pinned-bwrap", tmp_path / "other-bwrap"
    for path in (pinned, other):
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    assert agent._tool_namespace_bwrap({"bwrap": {"bwrap_path": str(pinned)}}) == pinned
    agent.config.tool_pid_namespace_bwrap_path = str(other)
    assert agent._tool_namespace_bwrap({"bwrap": {"bwrap_path": str(pinned)}}) == other
    agent.config.tool_pid_namespace_bwrap_path = None
    with pytest.raises(HTTPException, match="tool_pid_namespace"):
        agent._tool_namespace_bwrap({"docker": {}})
    # A wrong path is a 422 at seed, not an upload error later.
    pinned.chmod(0o644)
    with pytest.raises(HTTPException, match="not an executable file") as refused:
        agent._tool_namespace_bwrap({"bwrap": {"bwrap_path": str(pinned)}})
    assert refused.value.status_code == 422


@pytest.mark.parametrize(
    "receipt_missing, return_code, expected",
    [
        (True, None, "the process supervisor exited (137) without writing its cleanup receipt"),
        (False, 128 + 9, "the Codex runner or Codex was killed by signal 9"),
        (False, 128 + 15, "the Codex runner or Codex was killed by signal 15"),
        (False, 1, None),
        (False, 0, None),
        (False, None, None),
    ],
)
def test_harness_kill_evidence(receipt_missing, return_code, expected):
    cleanup = (
        None
        if receipt_missing
        else {"cleanup_confirmed": True, "return_code": return_code, "error": None, "timed_out": False}
    )
    evidence = harness_kill_evidence(
        cleanup=cleanup, cleanup_receipt_missing=receipt_missing, launch_return_code=137 if receipt_missing else 0
    )
    assert evidence == expected


def test_the_agent_is_blamed_only_with_positive_evidence(setup, tmp_path):
    agent, sandbox = setup
    # Its tools signalled the supervisor (ignored stop signals), then the runner died by a signal.
    sandbox.events = "\n".join(events().splitlines()[:2])
    sandbox.result["return_code"] = 128 + 9
    sandbox.result["foreign_signals"] = 3
    result, gaps, _ = _activate(agent)
    assert result.status_code == 502
    assert "agent_killed_harness" in gaps and "harness_killed" not in gaps


@pytest.mark.parametrize(
    ("missing", "launch_code", "return_code", "killed"),
    [
        (True, None, None, True),  # the launch's own status is unknown
        (True, 137, None, True),
        (True, 0, None, False),  # the launch shell exited normally: a launch failure, not a kill
        (True, 2, None, False),
        (True, 127, None, False),
        (False, 0, 128 + 15, True),
        (False, 0, 255, False),  # not a signal status
        (False, 0, 128 + 65, False),
    ],
)
def test_only_signal_statuses_count_as_a_kill(missing, launch_code, return_code, killed):
    cleanup = {"cleanup_confirmed": True, "return_code": return_code, "error": None, "timed_out": False}
    evidence = harness_kill_evidence(cleanup=cleanup, cleanup_receipt_missing=missing, launch_return_code=launch_code)
    assert (evidence is not None) is killed


def test_deadline_is_never_a_harness_kill():
    cleanup = {"cleanup_confirmed": True, "return_code": 128 + 15, "error": None, "timed_out": True}
    assert harness_kill_evidence(cleanup=cleanup, cleanup_receipt_missing=False, launch_return_code=0) is None


@pytest.mark.parametrize("isolated", [False, True])
def test_killed_harness_is_labelled_apart_from_other_failures(setup, tmp_path, isolated):
    agent, sandbox = setup
    agent.config.failure_artifacts_dir = str(tmp_path / "failures")
    if isolated:
        _seed_with_tool_namespace(agent, sandbox, tmp_path)
    # Observed 10-15: a tool command's `pkill -f node` ends Codex mid-turn; the runner exits 128+15.
    sandbox.events = "\n".join(events().splitlines()[:2])
    sandbox.result["return_code"] = 128 + 15
    result, gaps, _ = _activate(agent)
    # A signal status alone does not say who sent it (137 is also an OOM kill): the label stays neutral.
    label = "harness_killed"
    assert result.status_code == 502
    assert result.json()["detail"].startswith(
        f"Codex execution failed: {label}: the Codex runner or Codex was killed by signal 15"
    )
    assert gaps[label] == "the Codex runner or Codex was killed by signal 15"
    (saved,) = (tmp_path / "failures").iterdir()
    summary = json.loads((saved / "failure.json").read_text())
    assert summary["classification"] == label
    assert "cleanup.json" in summary["copied"]


def test_ignored_tool_signals_leave_a_successful_episode_with_a_gap(setup):
    agent, sandbox = setup
    sandbox.result["foreign_signals"] = 2
    result, gaps, _ = _activate(agent)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "completed"
    assert gaps["harness_signals_ignored"] == "stop signals from inside the sandbox, ignored: supervisor 2"
    assert "agent_killed_harness" not in gaps
