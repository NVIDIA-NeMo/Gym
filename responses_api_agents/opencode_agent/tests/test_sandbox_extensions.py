# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-staged plugins, pinned binaries and owner exports are opt-in; unset, sessions are unchanged."""

import asyncio
import json
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedRejectedError
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from responses_api_agents.opencode_agent import sandbox_runner
from responses_api_agents.opencode_agent.app import _CHECK_EXPORT_DIR, _CHECK_PLUGIN_FILES, OpenCodeAgentConfig
from responses_api_agents.opencode_agent.runtime import OBSERVABILITY_PATCH
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxSession
from responses_api_agents.opencode_agent.tests.test_continuation import activate, close, install_artifact_runner, start
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import (
    active_sessions,
    capture_observations,
    close_body,
    seed,
    setup,  # noqa: F401
)


PIN = "ab" * 32
RUNTIME = "/tmp/nemo-gym-opencode-runtime-1.17.11"
EXPORT_DIR = "/owner/exports"


def result(code=0, *, stdout="", stderr="", error_type=None):
    return SimpleNamespace(return_code=code, stdout=stdout, stderr=stderr, error_type=error_type)


def intercept(sandbox, marker, reply):
    """Answer commands containing ``marker`` with ``reply``; keep the fake's behavior for the rest."""
    run = sandbox.exec.side_effect

    async def exec_(command, **kwargs):
        if marker in command:
            return await reply(command, **kwargs)
        return await run(command, **kwargs)

    sandbox.exec.side_effect = exec_


def record(sandbox):
    """Record sandbox commands and the disconnect in order."""
    events = []
    run = sandbox.exec.side_effect

    async def exec_(command, **kwargs):
        events.append(command)
        return await run(command, **kwargs)

    sandbox.exec.side_effect = exec_
    sandbox.disconnect.side_effect = lambda: events.append("disconnect")
    return events


def index(events, marker):
    return next(position for position, event in enumerate(events) if marker in event)


async def episode(agent, sandbox, *, close_during_run=False, **request_fields):
    """Seed, activate and close one borrowed session; every wait is bounded."""
    request = Request(
        {"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}}
    )
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.blocked = close_during_run
    body = NeMoGymResponseCreateParamsNonStreaming(input="task", **request_fields)
    task = asyncio.create_task(agent.responses(request, body))
    await asyncio.wait_for(sandbox.started.wait(), 5)
    response = None if close_during_run else await asyncio.wait_for(task, 5)
    closed = await asyncio.wait_for(
        agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id))), 10
    )
    if close_during_run:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    return response, closed


def gap_codes(bundle):
    return [gap.code for gap in bundle.gaps]


def seed_status(agent, *, raise_server_exceptions=True):
    with TestClient(agent.setup_webserver(), raise_server_exceptions=raise_server_exceptions) as client:
        return client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))


async def test_unset_extensions_keep_setup_activation_and_capture_unchanged(setup):
    agent, sandbox = setup
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    directory = sandbox.directory
    install = events[index(events, "install_opencode_runtime.sh")]
    assert shlex.split(install) == [
        "bash",
        f"{directory}/install_opencode_runtime.sh",
        RUNTIME,
        "1.17.11",
        "",
        "",
        "python3",
        "",
    ]
    assert index(events, "install_opencode_runtime.sh") == 1
    for marker in ("os.path.isfile", "--export", "--stage-", "--import", "NG_OPENCODE_SKIP_RIPGREP"):
        assert not any(marker in event for event in events)
    payload = json.loads(sandbox.files[f"{directory}/input.json"])
    assert payload["command"][0] == f"{RUNTIME}/opencode"
    assert payload["harness_session_id"] is None and payload["previous_message_ids"] == []
    assert list(payload["env"]) == [
        "HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_DISABLE_PROJECT_CONFIG",
        "OPENCODE_DISABLE_AUTOUPDATE",
        "OPENCODE_DISABLE_MODELS_FETCH",
        "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX",
    ]
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert "agent" not in config
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"] == {
        "interleaved": {"field": "reasoning_content"},
        "limit": {"context": 32000, "input": 32000, "output": 4096},
    }
    assert set(config) == {"model", "small_model", "enabled_providers", "autoupdate", "share", "provider", "plugin"}
    assert config["plugin"] == [Path(f"{directory}/task_environment.mjs").as_uri()]
    assert response.status == "completed"
    assert "export_dir_write_failed" not in gap_codes(closed.agent_observations)


@pytest.mark.parametrize("capture", [False, True])
async def test_owner_plugins_are_checked_at_seed_and_load_after_gym_plugins_in_order(setup, capture):
    agent, sandbox = setup
    agent.server_client.global_config_dict["observability_enabled"] = capture
    agent.config.plugins = ["/opt/owner/z-last.mjs", "/opt/owner/a first.mjs"]
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    check = index(events, "os.path.isfile")
    assert 0 < check < index(events, "install_opencode_runtime.sh")
    assert shlex.split(events[check])[-2:] == agent.config.plugins
    directory = sandbox.directory
    config = json.loads(json.loads(sandbox.files[f"{directory}/input.json"])["env"]["OPENCODE_CONFIG_CONTENT"])
    gym_plugins = [Path(f"{directory}/{OBSERVABILITY_PATCH.name}").as_uri()] if capture else []
    gym_plugins.append(Path(f"{directory}/task_environment.mjs").as_uri())
    assert config["plugin"] == [*gym_plugins, "file:///opt/owner/z-last.mjs", "file:///opt/owner/a%20first.mjs"]


def test_plugin_check_reports_only_paths_that_are_not_regular_files(tmp_path):
    plugin = tmp_path / "plugin.mjs"
    plugin.write_text("export const Plugin = async () => ({});\n")
    paths = [str(plugin), str(tmp_path), str(tmp_path / "missing.mjs")]
    checked = subprocess.run(
        [sys.executable, "-I", "-c", _CHECK_PLUGIN_FILES, *paths], capture_output=True, text=True, timeout=30
    )
    assert checked.returncode == 3
    assert checked.stdout.split() == paths[1:]
    assert subprocess.run([sys.executable, "-I", "-c", _CHECK_PLUGIN_FILES, str(plugin)], timeout=30).returncode == 0


def test_missing_owner_plugin_fails_seed_with_422_and_releases_the_sandbox(setup):
    agent, sandbox = setup
    agent.config.plugins = ["/opt/owner/present.mjs", "/opt/owner/missing.mjs"]

    async def missing(command, **kwargs):
        return result(3, stdout="/opt/owner/missing.mjs\n")

    intercept(sandbox, "os.path.isfile", missing)
    response = seed_status(agent)
    assert response.status_code == 422, response.text
    assert response.json()["detail"].endswith("/opt/owner/missing.mjs")
    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    assert not any("install_opencode_runtime.sh" in command for command in commands)
    assert "rm -rf" in commands[-1]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    sandbox.launch.assert_not_awaited()
    assert not active_sessions(agent)


def test_plugin_check_outage_stays_retryable(setup):
    agent, sandbox = setup
    agent.config.plugins = ["/opt/owner/plugin.mjs"]

    async def outage(command, **kwargs):
        return result(0, stderr="provider timed out", error_type="timeout")

    intercept(sandbox, "os.path.isfile", outage)
    response = seed_status(agent, raise_server_exceptions=False)
    assert response.status_code == 500
    sandbox.disconnect.assert_awaited_once()
    assert not active_sessions(agent)


@pytest.mark.parametrize(
    "override",
    [
        {"plugins": ["relative/plugin.mjs"]},
        {"plugins": ["/opt/owner/../plugin.mjs"]},
        {"plugins": ["/opt/owner/\x00plugin.mjs"]},
        {"export_dir": "relative/exports"},
        {"export_dir": "/"},
        {"export_dir": "/tmp/nemo-gym-opencode-sessions/exports"},
        {"export_dir": "/app"},
        {"export_dir": "/app/exports"},
    ],
)
def test_invalid_extension_settings_fail_before_connecting(setup, override):
    agent, sandbox = setup
    for key, value in override.items():
        setattr(agent.config, key, value)
    response = seed_status(agent)
    assert response.status_code == 422, response.text
    sandbox.exec.assert_not_awaited()
    assert not sandbox.files


@pytest.mark.parametrize("digest", ["short", "g" * 64, "a" * 65])
def test_malformed_pin_is_rejected_by_configuration(digest):
    with pytest.raises(ValidationError):
        OpenCodeAgentConfig(
            name="opencode", host="localhost", port=1, entrypoint="app.py", sandbox_executable_sha256=digest
        )


@pytest.mark.parametrize("code", [1, 3])
def test_failed_binary_pin_stays_a_retryable_setup_error(setup, code):
    agent, sandbox = setup
    agent.config.sandbox_executable_path = "/mnt/cache/opencode"
    agent.config.sandbox_executable_sha256 = PIN

    async def installer(command, **kwargs):
        return result(code, stderr="ValueError: OpenCode executable digest mismatch")

    intercept(sandbox, "install_opencode_runtime.sh", installer)
    response = seed_status(agent, raise_server_exceptions=False)
    assert response.status_code == 500
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()
    assert not active_sessions(agent)


@pytest.mark.parametrize("close_during_run", [False, True])
async def test_exports_follow_capture_and_precede_session_file_removal(setup, tmp_path, close_during_run):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    capture_observations(sandbox, tmp_path)
    events = record(sandbox)
    response, closed = await episode(agent, sandbox, close_during_run=close_during_run)
    export = index(events, "--export")
    assert index(events, "--snapshot") < export < index(events, "rm -rf") < events.index("disconnect")
    assert len([event for event in events if "--export" in event]) == 1
    argv = shlex.split(events[export])
    directory = sandbox.directory
    assert argv[:7] == [
        "python3",
        "-I",
        f"{directory}/sandbox_runner.py",
        "--export",
        directory,
        directory,
        EXPORT_DIR,
    ]
    # The export deadline keeps a tenth of the 1 s collection budget for the capture to return.
    assert 0 < float(argv[7]) <= 0.9
    assert "export_dir_write_failed" not in gap_codes(closed.agent_observations)
    if not close_during_run:
        assert response.status == "completed"


@pytest.mark.parametrize("failure", ["exit", "provider", "transport"])
async def test_export_failure_is_a_gap_that_keeps_the_result_and_release(setup, failure):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR

    async def failed(command, **kwargs):
        if failure == "transport":
            raise ConnectionError("exec stream lost")
        error_type = "sandbox" if failure == "provider" else None
        return result(1 if failure == "exit" else 0, stderr="child export failed", error_type=error_type)

    intercept(sandbox, "--export", failed)
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Fixed"
    gaps = [gap for gap in closed.agent_observations.gaps if gap.code == "export_dir_write_failed"]
    assert len(gaps) == 1
    assert ("exec stream lost" if failure == "transport" else "child export failed") in gaps[0].detail
    sandbox.disconnect.assert_awaited_once()
    assert not active_sessions(agent)


async def test_hanging_export_is_bounded_by_the_collection_budget(setup):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    never = asyncio.Event()

    async def hang(command, **kwargs):
        await never.wait()

    intercept(sandbox, "--export", hang)
    started = monotonic()
    response, closed = await asyncio.wait_for(episode(agent, sandbox), 10)
    assert monotonic() - started < 5
    assert response.status == "completed"
    gaps = [gap for gap in closed.agent_observations.gaps if gap.code == "export_dir_write_failed"]
    assert len(gaps) == 1 and gaps[0].detail.startswith("TimeoutError")
    sandbox.disconnect.assert_awaited_once()


async def test_session_without_a_launch_writes_no_exports(setup):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    closed = await asyncio.wait_for(
        agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id))), 10
    )
    assert not any("--export" in call.args[0] for call in sandbox.exec.await_args_list)
    assert gap_codes(closed.agent_observations) == ["agent_activation_interrupted"]


def test_continuation_rewrites_exports_after_every_activation(setup, tmp_path):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    payloads = install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        assert activate(client, 1).status_code == 200
        assert activate(client, 2).status_code == 200
        receipt = close(client, request)
    exports = [shlex.split(call.args[0]) for call in sandbox.exec.await_args_list if "--export" in call.args[0]]
    activation_directories = [payload["directory"] for payload in payloads]
    assert [argv[4] for argv in exports] == activation_directories
    assert {argv[5] for argv in exports} == {str(Path(activation_directories[0]).parent)}
    assert "export_dir_write_failed" not in [gap["code"] for gap in receipt["agent_observations"]["gaps"]]


class HostSandbox:
    """Run sandbox commands on this host, with the test interpreter standing in for ``python3``.

    ``streams`` mimics what a provider keeps of the command's output: both streams, stderr merged into stdout
    with only the exit status as stderr (OpenSandbox's background exec), or no stderr at all.
    """

    def __init__(self, streams="separate"):
        self.streams = streams

    async def exec(self, command, *, cwd=None, timeout_s=30):
        command = command.replace("python3 ", shlex.quote(sys.executable) + " ", 1)
        stderr_to = asyncio.subprocess.STDOUT if self.streams == "merged" else asyncio.subprocess.PIPE
        process = await asyncio.create_subprocess_shell(
            command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=stderr_to
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        stderr = (stderr or b"").decode(errors="replace")
        if self.streams == "merged":
            stderr = f"exit status {process.returncode}" if process.returncode else ""
        elif self.streams == "dropped":
            stderr = ""
        return SandboxExecResult(stdout.decode(errors="replace"), stderr, process.returncode)


def host_export_session(tmp_path, *, failing=(), streams="separate"):
    """A finished single-turn session on this host whose stand-in CLI exports each session, failing ``failing``."""
    conversation = tmp_path / "session"
    database = conversation / "data/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    workdir = tmp_path / "task"
    workdir.mkdir()
    shutil.copyfile(sandbox_runner.__file__, conversation / "sandbox_runner.py")
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            create table session(id text, parent_id text, time_created integer);
            insert into session values('ses_root', null, 0), ('ses_child', 'ses_root', 1);
        """)
    cli = tmp_path / "opencode"
    cli.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"if sys.argv[2] in {list(failing)!r}:\n"
        "    sys.exit('Session not found: ' + sys.argv[2])\n"
        "print(json.dumps({'info': {'id': sys.argv[2]}, 'messages': []}))\n"
    )
    cli.chmod(0o755)
    (conversation / "input.json").write_text(
        json.dumps({"command": [str(cli), "run"], "cwd": str(workdir), "env": {}, "activation_id": None})
    )
    (conversation / "stdout.jsonl").write_text('{"type": "step_start"}\n')
    exports = tmp_path / "owner exports"
    state = OpenCodeSandboxSession(
        request=seed(),
        session=SandboxSession(
            sandbox=HostSandbox(streams), session_dir=str(conversation), workdir=str(workdir), harness="OpenCode"
        ),
        runtime=str(tmp_path),
        export_dir=str(exports),
    )
    return state, exports


async def test_adapter_export_command_runs_the_runner_inside_the_sandbox(tmp_path):
    state, exports = host_export_session(tmp_path)
    await asyncio.wait_for(state.write_exports(deadline=monotonic() + 30), 40)
    assert state.take_export_gaps() == []
    assert json.loads((exports / "sessions.json").read_text()) == [
        {"id": "ses_root", "parent_id": None},
        {"id": "ses_child", "parent_id": "ses_root"},
    ]
    for session_id in ("ses_root", "ses_child"):
        assert json.loads((exports / f"{session_id}.json").read_text())["info"]["id"] == session_id
    assert (exports / "ses_root.events.jsonl").read_text() == '{"type": "step_start"}\n'


@pytest.mark.parametrize("streams", ["separate", "merged", "dropped"])
async def test_failed_session_reaches_the_gap_whatever_the_provider_keeps_of_stderr(tmp_path, streams):
    state, _ = host_export_session(tmp_path, failing=["ses_child"], streams=streams)
    await asyncio.wait_for(state.write_exports(deadline=monotonic() + 30), 40)
    (gap,) = state.take_export_gaps()
    assert gap.code == "export_dir_write_failed"
    assert gap.detail.startswith("RuntimeError: exit=1, error=None, exec budget ")
    assert "1 of 2 sessions exported in " in gap.detail
    assert "ses_child: exit 1: Session not found: ses_child" in gap.detail
    if streams == "merged":
        assert gap.detail.endswith("stderr: exit status 1")


EXPORT_STATUS_LINE = json.dumps(
    {
        "sessions": {"ses_root": "ok", "ses_child": "exit 1: Session not found: ses_child"},
        "event_log": "ok",
        "deadline_exhausted": False,
        "error": None,
        "budget_seconds": 43.2,
        "elapsed_seconds": 3.1,
    }
)
EXPORT_FAILED_LINE = "OpenCode session export failed: ses_child: exit 1: Session not found: ses_child"


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (EXPORT_STATUS_LINE + "\n", EXPORT_FAILED_LINE + "\n"),
        (f"{EXPORT_FAILED_LINE}\n{EXPORT_STATUS_LINE}\n", "exit status 1"),
        (EXPORT_STATUS_LINE + "\n", ""),
        (EXPORT_STATUS_LINE + "\n", None),
    ],
    ids=["separate", "merged", "stderr empty", "stderr none"],
)
async def test_export_gap_detail_comes_from_the_status_line(setup, stdout, stderr):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR

    async def failed(command, **kwargs):
        return result(1, stdout=stdout, stderr=stderr)

    intercept(sandbox, "--export", failed)
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    (gap,) = [gap for gap in closed.agent_observations.gaps if gap.code == "export_dir_write_failed"]
    assert "1 of 2 sessions exported in 3.1 of 43.2 s; ses_child: exit 1: Session not found: ses_child" in gap.detail
    if stderr:
        assert gap.detail.endswith(f"stderr: {stderr}")


async def test_export_gap_without_a_status_line_keeps_what_the_provider_returned(setup):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR

    async def killed(command, **kwargs):
        return result(137, stdout="partial output", stderr="")

    intercept(sandbox, "--export", killed)
    _, closed = await episode(agent, sandbox)
    (gap,) = [gap for gap in closed.agent_observations.gaps if gap.code == "export_dir_write_failed"]
    assert "exit=137, error=None" in gap.detail
    assert gap.detail.endswith("no export status; stdout: partial output")


@pytest.mark.parametrize(("budget", "margin"), [(100.0, 5.0), (10.0, 1.0)])
async def test_export_runner_stops_before_its_exec_times_out(budget, margin):
    calls = []

    class Recorder:
        async def exec(self, command, *, cwd=None, timeout_s=None):
            calls.append((shlex.split(command), timeout_s))
            return SandboxExecResult("", "", 0)

    state = OpenCodeSandboxSession(
        request=seed(),
        session=SandboxSession(sandbox=Recorder(), session_dir="/s/one", workdir="/app", harness="OpenCode"),
        runtime="/r",
        export_dir=EXPORT_DIR,
    )
    await state.write_exports(deadline=monotonic() + budget)
    ((argv, exec_timeout),) = calls
    assert budget - 1 < exec_timeout <= budget
    # The runner's own deadline leaves it time to report each session before the exec is abandoned.
    assert float(argv[-1]) == pytest.approx(exec_timeout - margin, abs=0.05)
    assert state.take_export_gaps() == []


def launch_config(sandbox):
    payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
    return payload, json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])


BUILTIN_AGENTS = {"build", "plan", "general", "explore", "compaction", "title", "summary"}


async def test_request_sampling_reaches_every_builtin_agent_and_enables_temperature(setup):
    agent, sandbox = setup
    response, _ = await episode(agent, sandbox, temperature=1.0, top_p=0.95)
    assert response.status == "completed"
    _, config = launch_config(sandbox)
    assert config["agent"] == {name: {"temperature": 1.0, "top_p": 0.95} for name in BUILTIN_AGENTS}
    # OpenCode 1.17.11 sends no temperature for a configured model that does not declare the capability.
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["temperature"] is True


async def test_top_p_alone_leaves_the_temperature_capability_unset(setup):
    agent, sandbox = setup
    await episode(agent, sandbox, top_p=0.5)
    _, config = launch_config(sandbox)
    assert config["agent"] == {name: {"top_p": 0.5} for name in BUILTIN_AGENTS}
    assert "temperature" not in config["provider"]["nemo_gym"]["models"]["dummy_model"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_sampling_is_rejected_without_a_launch(setup, value):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        body = json.dumps({"input": "task", "temperature": value})
        response = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses", content=body, headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert "finite temperature" in response.json()["detail"]
    sandbox.launch.assert_not_awaited()


def test_continuation_activation_applies_its_own_sampling(setup, tmp_path):
    agent, sandbox = setup
    payloads = install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        assert activate(client, 1).status_code == 200
        body = {"input": "Next", "temperature": 0.7, "metadata": {"agent_timeout_sec": "30"}}
        assert client.post("/v1/responses", headers={"x-ng-agent-activation": "2"}, json=body).status_code == 200
        close(client, request)
    configs = [json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"]) for payload in payloads]
    assert "agent" not in configs[0]
    assert configs[1]["agent"]["build"] == {"temperature": 0.7}


@pytest.mark.parametrize("tools", [None, {}])
async def test_null_or_empty_tools_leave_permission_as_the_only_rules(setup, tools):
    agent, sandbox = setup
    permission = {"*": "allow", "webfetch": "deny", "bash": {"*": "allow", "*git push*": "deny"}, "skill": "deny"}
    agent.config.opencode_config = {"tools": tools, "permission": permission}
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    _, config = launch_config(sandbox)
    assert "tools" not in config
    assert json.dumps(config["permission"]) == json.dumps(permission)
    assert agent.config.opencode_config["tools"] == tools


def test_local_config_drops_null_tools(setup):
    agent, _ = setup
    agent.config.opencode_config = {"tools": None, "permission": {"edit": "deny"}}
    assert agent._build_opencode_config()["permission"] == {"edit": "deny"}
    assert "tools" not in agent._build_opencode_config()


async def test_pinned_ripgrep_is_staged_where_opencode_looks_and_replaces_the_package_install(setup):
    agent, sandbox = setup
    agent.config.sandbox_ripgrep_path = "/mnt/cache/rg"
    agent.config.sandbox_ripgrep_sha256 = PIN
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    directory = sandbox.directory
    install = events[index(events, "install_opencode_runtime.sh")]
    assert install.startswith("NG_OPENCODE_SKIP_RIPGREP=1 bash ")
    stage = index(events, "--stage-executable")
    assert index(events, "install_opencode_runtime.sh") < stage < index(events, "--receipt")
    assert shlex.split(events[stage]) == [
        "python3",
        "-I",
        f"{directory}/sandbox_runner.py",
        "--stage-executable",
        "/mnt/cache/rg",
        PIN,
        f"{directory}/cache/opencode/bin/rg",
    ]
    payload, _ = launch_config(sandbox)
    assert payload["env"]["XDG_CACHE_HOME"] == f"{directory}/cache"


async def test_pinned_plugin_dependencies_unpack_into_the_private_config_directory(setup):
    agent, sandbox = setup
    agent.config.sandbox_plugin_dependencies_path = "/mnt/cache/opencode-plugin-dependencies.tar.gz"
    agent.config.sandbox_plugin_dependencies_sha256 = PIN
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    directory = sandbox.directory
    argv = shlex.split(events[index(events, "--stage-archive")])
    assert argv[3:] == [
        "--stage-archive",
        agent.config.sandbox_plugin_dependencies_path,
        PIN,
        f"{directory}/config/opencode",
    ]
    payload, _ = launch_config(sandbox)
    assert payload["env"]["XDG_CONFIG_HOME"] == f"{directory}/config"
    # Without a pinned rg the installer keeps its best-effort ripgrep install.
    assert not any(event.startswith("NG_OPENCODE_SKIP_RIPGREP") for event in events)


@pytest.mark.parametrize("kind", ["ripgrep", "plugin_dependencies"])
@pytest.mark.parametrize(("code", "status"), [(3, 422), (1, 500)])
def test_only_rejected_bundle_bytes_are_a_deterministic_setup_error(setup, kind, code, status):
    agent, sandbox = setup
    setattr(agent.config, f"sandbox_{kind}_path", "/mnt/cache/item")
    setattr(agent.config, f"sandbox_{kind}_sha256", PIN)

    async def staged(command, **kwargs):
        return result(code, stderr="Rejected: SHA-256 mismatch: expected a, found b")

    intercept(sandbox, "--stage-", staged)
    response = seed_status(agent, raise_server_exceptions=False)
    assert response.status_code == status
    if status == 422:
        assert "SHA-256 mismatch: expected a, found b" in response.json()["detail"]
    sandbox.disconnect.assert_awaited_once()
    assert not active_sessions(agent)


@pytest.mark.parametrize(
    "override",
    [
        {"sandbox_ripgrep_path": "/mnt/rg"},
        {"sandbox_ripgrep_sha256": PIN},
        {"sandbox_plugin_dependencies_path": "deps.tar", "sandbox_plugin_dependencies_sha256": PIN},
        {"session_import_path": "relative/export.json"},
        {"session_import_path": "/staged/../export.json"},
    ],
)
def test_invalid_bundle_and_import_settings_fail_before_connecting(setup, override):
    agent, sandbox = setup
    for key, value in override.items():
        setattr(agent.config, key, value)
    assert seed_status(agent).status_code == 422
    sandbox.exec.assert_not_awaited()


IMPORT_PATH = "/owner/prefix-export.json"


async def test_imported_session_is_continued_without_its_prefix(setup, tmp_path):
    agent, sandbox = setup
    agent.config.session_import_path = IMPORT_PATH
    imported = {"session_id": "ses_prefix", "message_ids": ["prefix-user", "prefix-reply"]}

    async def import_(command, **kwargs):
        argv = shlex.split(command)
        params = json.loads(sandbox.files[argv[4]])
        assert argv[3] == "--import" and argv[5] == IMPORT_PATH
        assert params["cwd"] == kwargs["cwd"] == "/app"
        assert params["command"] == [f"{RUNTIME}/opencode"]
        directory = argv[2].rsplit("/", 1)[0]
        assert params["env"]["XDG_DATA_HOME"] == f"{directory}/data"
        assert "OPENCODE_CONFIG_CONTENT" not in params["env"]
        return result(stdout=json.dumps(imported))

    intercept(sandbox, "--import", import_)
    # The stored transcript holds the imported prefix before this activation's turn.
    database = tmp_path / "observations.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('ses_prefix', null, 0);
        """)
        export = json.loads(sandbox.events)
        rows = [("prefix-user", {"role": "user"}, {"type": "text", "text": "Earlier task"})]
        rows.append(("prefix-reply", {**export["messages"][1]["info"]}, {"type": "text", "text": "Earlier answer"}))
        for index_, message in enumerate(export["messages"]):
            for part in message["parts"]:
                rows.append((f"activation-{index_}", message["info"], part))
        for position, (message_id, info, part) in enumerate(rows):
            connection.execute(
                "insert or ignore into message values (?, 'ses_prefix', ?, ?)",
                (message_id, json.dumps(info), position),
            )
            connection.execute(
                "insert into part values (?, ?, 'ses_prefix', ?, ?)",
                (f"part-{position}", message_id, json.dumps(part), position),
            )
    # The runner's snapshot leaves previous messages out of the exported turns (see test_sandbox_runner).
    sandbox.events = json.dumps(
        {
            **export,
            "session_id": "ses_prefix",
            "message_ids": ["prefix-user", "prefix-reply", "activation-0", "activation-1"],
            "activation_message_ids": ["activation-0", "activation-1"],
            "usage_messages": [export["messages"][1]["info"]],
        }
    )
    original = sandbox.download

    async def download(source, destination):
        if source.endswith("/observations.db"):
            Path(destination).write_bytes(database.read_bytes())
        else:
            await original(source, destination)

    sandbox.download = download
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    assert index(events, "--import") < index(events, "--receipt")
    payload, _ = launch_config(sandbox)
    assert payload["command"][-2:] == ["--session", "ses_prefix"]
    assert payload["harness_session_id"] == "ses_prefix"
    assert payload["previous_message_ids"] == ["prefix-user", "prefix-reply"]
    assert response.status == "completed"
    assert "Earlier answer" not in json.dumps(response.model_dump(mode="json"))
    assert response.output[-1].content[0].text == "Fixed"
    invocation = next(record for record in closed.agent_observations.records if record.kind == "agent_invocation")
    transcript = json.dumps([item.model_dump(mode="json") for item in invocation.conversation])
    assert "Fixed" in transcript
    assert "Earlier answer" not in transcript and "Earlier task" not in transcript


async def test_absent_import_file_leaves_a_fresh_session(setup):
    agent, sandbox = setup
    agent.config.session_import_path = IMPORT_PATH

    async def nothing_staged(command, **kwargs):
        return result(stdout="null\n")

    intercept(sandbox, "--import", nothing_staged)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    payload, _ = launch_config(sandbox)
    assert "--session" not in payload["command"]
    assert payload["harness_session_id"] is None and payload["previous_message_ids"] == []


@pytest.mark.parametrize(("code", "status"), [(3, 422), (1, 500)])
def test_failed_import_is_a_deterministic_setup_error_only_when_rejected(setup, code, status):
    agent, sandbox = setup
    agent.config.session_import_path = IMPORT_PATH

    async def failed(command, **kwargs):
        return result(code, stderr="Rejected: opencode import exited 1: invalid export")

    intercept(sandbox, "--import", failed)
    response = seed_status(agent, raise_server_exceptions=False)
    assert response.status_code == status
    if status == 422:
        assert "invalid export" in response.json()["detail"]
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()
    assert not active_sessions(agent)


def test_export_failure_is_reported_by_the_activation_that_had_it(setup, tmp_path):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    install_artifact_runner(sandbox, tmp_path)
    exports = []

    async def first_fails(command, **kwargs):
        exports.append(command)
        return result(1 if len(exports) == 1 else 0, stderr="disk full")

    intercept(sandbox, "--export", first_fails)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        first, second = activate(client, 1).json(), activate(client, 2).json()
        receipt = close(client, request)

    def codes(observations):
        return [gap["code"] for gap in observations["gaps"]]

    assert "export_dir_write_failed" in codes(first["_ng_agent_observations"])
    assert "export_dir_write_failed" not in codes(second["_ng_agent_observations"])
    assert codes(receipt["agent_observations"]).count("export_dir_write_failed") == 1


@pytest.mark.parametrize(
    ("export_dir", "status"), [("/srv", 422), ("/srv/task", 422), ("/srv/task/out", 422), ("/srv/exports", 200)]
)
def test_export_dir_must_be_neither_the_workdir_inside_it_nor_above_it(setup, export_dir, status):
    agent, sandbox = setup
    agent.config.export_dir = export_dir
    body = seed()
    body.sandbox_access.workdir = "/srv/task"
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
    assert response.status_code == status, response.text
    if status == 422:
        assert "export_dir must be neither the task workdir" in response.json()["detail"]
        sandbox.exec.assert_not_awaited()
    else:
        # The prepare step checks the resolved paths too, before any session file exists.
        argv = shlex.split(sandbox.exec.await_args_list[0].args[0])
        check = argv.index(_CHECK_EXPORT_DIR)
        assert argv[check - 3 : check] == ["python3", "-I", "-c"]
        assert argv[check + 1 : check + 4] == ["/srv/task", "/srv/exports", "&&"]
        assert argv[check + 4 : check + 6] == ["mkdir", "-p"]


def test_resolved_export_dir_check_follows_links_in_the_sandbox(tmp_path):
    workdir = tmp_path / "task"
    (workdir / "out").mkdir(parents=True)
    inside = tmp_path / "inside"
    inside.symlink_to(workdir / "out", target_is_directory=True)
    above = tmp_path / "above"
    above.symlink_to(tmp_path, target_is_directory=True)
    cases = {inside: False, inside / "not-yet": False, above: False, workdir: False, tmp_path / "exports": True}
    for export_dir, accepted in cases.items():
        checked = subprocess.run(
            [sys.executable, "-I", "-c", _CHECK_EXPORT_DIR, str(workdir), str(export_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (checked.returncode == 0) is accepted, (export_dir, checked.stderr)
        if not accepted:
            assert "OpenCode export_dir overlaps the task workdir" in checked.stderr


async def test_configured_interpreter_and_path_prefix_reach_every_sandbox_command(setup, tmp_path):
    agent, sandbox = setup
    python, prefix = "/opt/gym runtime/python/bin/python3", "/opt/gym runtime/bin"
    agent.config.sandbox_python_executable = python
    agent.config.sandbox_path_prefix = [prefix]
    agent.config.plugins = ["/opt/owner/plugin.mjs"]
    agent.config.sandbox_ripgrep_path = "/mnt/cache/rg"
    agent.config.sandbox_ripgrep_sha256 = PIN
    agent.config.sandbox_plugin_dependencies_path = "/mnt/cache/deps.tar.gz"
    agent.config.sandbox_plugin_dependencies_sha256 = PIN
    agent.config.session_import_path = IMPORT_PATH
    agent.config.export_dir = EXPORT_DIR

    async def nothing_staged(command, **kwargs):
        return result(stdout="null\n")

    intercept(sandbox, "--import", nothing_staged)
    capture_observations(sandbox, tmp_path)
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    prepare = shlex.split(events[0])
    assert prepare[prepare.index(_CHECK_EXPORT_DIR) - 3] == python
    assert shlex.split(events[index(events, "install_opencode_runtime.sh")])[-2:] == [python, prefix]
    for marker in ("os.path.isfile", "--stage-executable", "--stage-archive", "--import", "--export", "--snapshot"):
        assert shlex.split(events[index(events, marker)])[0] == python, marker
    assert f"exec {shlex.quote(python)} -I " in events[index(events, "--receipt")]
    directory = sandbox.directory
    for name in ("input.json", "import.json"):
        assert json.loads(sandbox.files[f"{directory}/{name}"])["path_prefix"] == [prefix], name
    assert "export_dir_write_failed" not in gap_codes(closed.agent_observations)


async def test_unset_export_dir_adds_no_prepare_check(setup):
    agent, sandbox = setup
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    prepare = sandbox.exec.await_args_list[0].args[0]
    assert prepare.count("python3 -I -c") == 1
    assert _CHECK_EXPORT_DIR not in shlex.split(prepare)


def refused(agent, caplog):
    """Seed through the real route, as an Environment Server does; return the refusal, which must be final."""
    body = seed()
    detail = check_agent_seed_rejected(agent.setup_webserver(), body)
    # AgentSeedRejectedError, not just any 422: the base agent logs each refusal once.
    assert f"Refused agent session seed {body.agent_session_id}: {detail}" in caplog.text
    assert not active_sessions(agent)
    return detail


SETTING_REFUSALS = {
    "plugin path": (
        {"plugins": ["relative/plugin.mjs"]},
        "OpenCode plugins must be absolute, normalized paths inside the sandbox",
    ),
    "export_dir path": (
        {"export_dir": "relative/exports"},
        "OpenCode export_dir must be an absolute sandbox path outside adapter runtime/session storage",
    ),
    "export_dir in the workdir": (
        {"export_dir": "/app/exports"},
        "OpenCode export_dir must be neither the task workdir, inside it nor above it",
    ),
    "import path": (
        {"session_import_path": "/staged/../export.json"},
        "OpenCode session_import_path must be an absolute, normalized sandbox path",
    ),
    "digest without a file": (
        {"sandbox_ripgrep_sha256": PIN},
        "sandbox_ripgrep_sha256 pins a supplied file; set sandbox_ripgrep_path",
    ),
    "file without a digest": (
        {"sandbox_plugin_dependencies_path": "/mnt/deps.tar"},
        "A supplied file requires sandbox_plugin_dependencies_sha256",
    ),
    "relative pin": (
        {"sandbox_ripgrep_path": "mnt/rg", "sandbox_ripgrep_sha256": PIN},
        "sandbox_ripgrep_path must be an absolute, normalized sandbox path",
    ),
}


@pytest.mark.parametrize(("override", "reason"), SETTING_REFUSALS.values(), ids=SETTING_REFUSALS.keys())
def test_unusable_owner_setting_refuses_the_seed_before_connecting(setup, caplog, override, reason):
    agent, sandbox = setup
    for key, value in override.items():
        setattr(agent.config, key, value)
    assert refused(agent, caplog) == reason
    sandbox.exec.assert_not_awaited()
    sandbox.disconnect.assert_not_awaited()


INPUT_REFUSALS = {
    "plugin file missing": (
        {"plugins": ["/opt/owner/missing.mjs"]},
        "os.path.isfile",
        result(3, stdout="/opt/owner/missing.mjs\n"),
        "OpenCode plugin files are missing from the sandbox: /opt/owner/missing.mjs",
    ),
    "pinned ripgrep": (
        {"sandbox_ripgrep_path": "/mnt/cache/rg", "sandbox_ripgrep_sha256": PIN},
        "--stage-executable",
        result(3, stderr="Rejected: /mnt/cache/rg --version exited 2: no"),
        "Pinned ripgrep executable rejected",
    ),
    "pinned plugin dependencies": (
        {"sandbox_plugin_dependencies_path": "/mnt/cache/deps.tar", "sandbox_plugin_dependencies_sha256": PIN},
        "--stage-archive",
        result(3, stderr="Rejected: archive member 'd/l/l2' passes through the link 'd/l'"),
        "Pinned plugin dependency archive rejected",
    ),
    "session import": (
        {"session_import_path": IMPORT_PATH},
        "--import",
        result(3, stderr="Rejected: opencode import exited 1: invalid export"),
        "OpenCode could not import session_import_path",
    ),
}


@pytest.mark.parametrize(("override", "marker", "reply", "reason"), INPUT_REFUSALS.values(), ids=INPUT_REFUSALS.keys())
def test_rejected_owner_input_refuses_the_seed_and_releases_the_sandbox(
    setup, caplog, override, marker, reply, reason
):
    agent, sandbox = setup
    for key, value in override.items():
        setattr(agent.config, key, value)

    async def rejected(command, **kwargs):
        return reply

    intercept(sandbox, marker, rejected)
    detail = refused(agent, caplog)
    assert detail.startswith(reason)
    assert (reply.stdout or reply.stderr).strip() in detail
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


async def test_refusal_is_a_value_error_for_callers_outside_a_request(setup):
    agent, _ = setup
    agent.config.export_dir = "/app"
    with pytest.raises(ValueError, match="export_dir must be neither the task workdir") as refusal:
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    assert isinstance(refusal.value, AgentSeedRejectedError) and refusal.value.status_code == 422
