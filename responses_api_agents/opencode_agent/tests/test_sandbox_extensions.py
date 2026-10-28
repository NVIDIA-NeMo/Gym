# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-staged plugins, pinned binaries and owner exports are opt-in; unset, sessions are unchanged."""

import asyncio
import hashlib
import io
import json
import shlex
import sqlite3
import subprocess
import tarfile
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
from responses_api_agents.opencode_agent.app import (
    _CHECK_EXPORT_DIR,
    _CHECK_PLUGIN_FILES,
    OpenCodeAgentConfig,
    _sandbox_prepare_command,
)
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
# Distinctive text of each sandbox step, to find its command.
PLUGIN_CHECK = 'for path in "$@"'
EXPORT_SESSION = '"$1" export "$3"'
EVENT_LOG = ".events.jsonl"
PINNED_COPY = "sha256sum"
VERSION_PROBE = "--version"
UNPACK = "tar -x"
IMPORT_CHECK = "test -f"
IMPORT = "/opencode import "


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


def positional(command):
    """The arguments that a step's first ``set --`` line gives its script."""
    line = next(line for line in command.splitlines() if line.startswith("set -- "))
    return shlex.split(line)[2:]


def environment(command):
    """The variables that the OpenCode environment prelude of ``command`` exports."""
    line = next(line for line in command.splitlines() if line.startswith("export "))
    return dict(word.split("=", 1) for word in shlex.split(line)[1:])


def session_store(path, *, roots, messages=()):
    """An OpenCode session store holding the root sessions ``roots`` and the messages ``messages``."""
    with sqlite3.connect(path) as connection:
        connection.execute("create table session(id text, parent_id text, time_created integer)")
        connection.execute("create table message(id text, session_id text, data text, time_created integer)")
        connection.executemany("insert into session values (?, null, ?)", [(root, n) for n, root in enumerate(roots)])
        connection.executemany(
            "insert into message values (?, ?, '{}', ?)",
            [(message, roots[0], n) for n, message in enumerate(messages)],
        )
    return path


def plugin_dependencies(sandbox, tmp_path, *members):
    """A dependency archive that the fake serves where the adapter downloads its checked copy; returns its SHA-256."""
    archive = tmp_path / "plugin-dependencies.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in members or [("package.json", b"{}")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    original = sandbox.download

    async def download(source, destination):
        if source.endswith("/.pinned-opencode"):
            Path(destination).write_bytes(archive.read_bytes())
        else:
            await original(source, destination)

    sandbox.download = download
    return hashlib.sha256(archive.read_bytes()).hexdigest()


async def test_unset_extensions_keep_setup_activation_and_capture_unchanged(setup):
    agent, sandbox = setup
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    directory = sandbox.directory
    install = events[index(events, "install_opencode_runtime.sh")]
    assert shlex.split(install) == ["bash", f"{directory}/install_opencode_runtime.sh", RUNTIME, "1.17.11", "", "", ""]
    assert index(events, "install_opencode_runtime.sh") == 1
    for marker in (PLUGIN_CHECK, EXPORT_SESSION, EVENT_LOG, PINNED_COPY, IMPORT_CHECK, "NG_OPENCODE_SKIP_RIPGREP"):
        assert not any(marker in event for event in events)
    assert _CHECK_EXPORT_DIR not in events[0]
    params, config = launch_config(sandbox)
    assert params["command"][0] == f"{RUNTIME}/opencode" and "--session" not in params["command"]
    assert sandbox.snapshots == [{"harness_session_id": None, "previous_message_ids": []}]
    assert list(params["env"]) == [
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
    check = index(events, PLUGIN_CHECK)
    assert 0 < check < index(events, "install_opencode_runtime.sh")
    assert positional(events[check]) == agent.config.plugins
    directory = sandbox.directory
    _, config = launch_config(sandbox)
    gym_plugins = [Path(f"{directory}/{OBSERVABILITY_PATCH.name}").as_uri()] if capture else []
    gym_plugins.append(Path(f"{directory}/task_environment.mjs").as_uri())
    assert config["plugin"] == [*gym_plugins, "file:///opt/owner/z-last.mjs", "file:///opt/owner/a%20first.mjs"]


def test_plugin_check_reports_only_paths_that_are_not_regular_files(tmp_path):
    plugin = tmp_path / "plugin.mjs"
    plugin.write_text("export const Plugin = async () => ({});\n")
    paths = [str(plugin), str(tmp_path), str(tmp_path / "missing.mjs")]
    checked = subprocess.run(
        ["/bin/sh", "-c", _CHECK_PLUGIN_FILES, "sh", *paths], capture_output=True, text=True, timeout=30
    )
    assert checked.returncode == 3
    assert checked.stdout.split() == paths[1:]
    assert subprocess.run(["/bin/sh", "-c", _CHECK_PLUGIN_FILES, "sh", str(plugin)], timeout=30).returncode == 0


def test_missing_owner_plugin_fails_seed_with_422_and_releases_the_sandbox(setup):
    agent, sandbox = setup
    agent.config.plugins = ["/opt/owner/present.mjs", "/opt/owner/missing.mjs"]

    async def missing(command, **kwargs):
        return result(3, stdout="/opt/owner/missing.mjs\n")

    intercept(sandbox, PLUGIN_CHECK, missing)
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

    intercept(sandbox, PLUGIN_CHECK, outage)
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
    export, event_log = index(events, EXPORT_SESSION), index(events, EVENT_LOG)
    snapshot = index(events, "opencode.db-wal")
    assert snapshot < index(events, "/sessions.json") < export < event_log < index(events, "rm -rf")
    assert index(events, "rm -rf") < events.index("disconnect")
    assert len([event for event in events if EXPORT_SESSION in event]) == 1
    directory = sandbox.directory
    assert positional(events[export]) == [f"{RUNTIME}/opencode", EXPORT_DIR, "root", f"{directory}/export-root"]
    # The run's private store, from the task workdir, without the run's configuration.
    assert "cd /app || exit 125\nunset OPENCODE_CONFIG_CONTENT\n" in events[export]
    env = environment(events[export])
    assert env["XDG_DATA_HOME"] == f"{directory}/data" and env["PWD"] == "/app"
    assert "OPENCODE_CONFIG_CONTENT" not in env
    assert positional(events[event_log]) == [f"{directory}/stdout.jsonl", f"{EXPORT_DIR}/root.events.jsonl"]
    assert json.loads(sandbox.files[f"{directory}/sessions.json"]) == [{"id": "root", "parent_id": None}]
    assert "export_dir_write_failed" not in gap_codes(closed.agent_observations)
    if not close_during_run:
        assert response.status == "completed"


@pytest.mark.parametrize("failure", ["exit", "provider", "transport"])
async def test_export_failure_is_a_gap_that_keeps_the_result_and_release(setup, tmp_path, failure):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    capture_observations(sandbox, tmp_path)

    async def failed(command, **kwargs):
        if failure == "transport":
            raise ConnectionError("exec stream lost")
        if failure == "provider":
            return result(0, stderr="child export failed", error_type="sandbox")
        return result(1, stdout="exit 1: child export failed")

    intercept(sandbox, EXPORT_SESSION, failed)
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Fixed"
    gaps = [gap for gap in closed.agent_observations.gaps if gap.code == "export_dir_write_failed"]
    assert len(gaps) == 1
    assert ("exec stream lost" if failure == "transport" else "child export failed") in gaps[0].detail
    sandbox.disconnect.assert_awaited_once()
    assert not active_sessions(agent)


async def test_hanging_export_is_bounded_by_the_collection_budget(setup, tmp_path):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    capture_observations(sandbox, tmp_path)
    never = asyncio.Event()

    async def hang(command, **kwargs):
        await never.wait()

    intercept(sandbox, EXPORT_SESSION, hang)
    started = monotonic()
    response, closed = await asyncio.wait_for(episode(agent, sandbox), 10)
    assert monotonic() - started < 5
    assert response.status == "completed"
    (gap,) = [gap for gap in closed.agent_observations.gaps if gap.code == "export_dir_write_failed"]
    assert gap.detail.startswith("0 of 1 sessions exported in ") and ", deadline exhausted; " in gap.detail
    assert "root: export deadline exceeded" in gap.detail
    sandbox.disconnect.assert_awaited_once()


async def test_session_without_a_launch_writes_no_exports(setup):
    agent, sandbox = setup
    agent.config.export_dir = EXPORT_DIR
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    closed = await asyncio.wait_for(
        agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id))), 10
    )
    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    assert not any(EXPORT_SESSION in command or "/sessions.json" in command for command in commands)
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
    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    exports = [positional(command) for command in commands if EXPORT_SESSION in command]
    activation_directories = [payload["directory"] for payload in payloads]
    assert [argv[3] for argv in exports] == [
        f"{directory}/export-opencode-session" for directory in activation_directories
    ]
    # One conversation store across activations; one event log per activation.
    conversation = str(Path(activation_directories[0]).parent)
    assert {environment(command)["XDG_DATA_HOME"] for command in commands if EXPORT_SESSION in command} == {
        f"{conversation}/data"
    }
    assert [positional(command)[1] for command in commands if EVENT_LOG in command] == [
        f"{EXPORT_DIR}/opencode-session.activation-1.events.jsonl",
        f"{EXPORT_DIR}/opencode-session.activation-2.events.jsonl",
    ]
    assert "export_dir_write_failed" not in [gap["code"] for gap in receipt["agent_observations"]["gaps"]]


async def test_each_export_step_gets_what_is_left_of_the_budget(tmp_path):
    timeouts = []

    class Recorder:
        async def exec(self, command, *, cwd=None, timeout_s=None):
            timeouts.append(timeout_s)
            return SandboxExecResult("", "", 0)

        async def upload(self, source, destination):
            pass

    session = SandboxSession(sandbox=Recorder(), session_dir="/s/one", workdir="/app", harness="OpenCode")
    session.cleanup = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
    state = OpenCodeSandboxSession(request=seed(), session=session, runtime="/r", export_dir=EXPORT_DIR)
    session_store(tmp_path / "opencode.db", roots=["root"])
    budget = 10.0
    await state.write_exports(tmp_path, deadline=monotonic() + budget)
    assert state.take_export_gaps() == []
    # Prepare, publish the inventory, the session, the event log: each until the same deadline.
    assert len(timeouts) == 4
    assert budget - 1 < timeouts[0] <= budget and timeouts == sorted(timeouts, reverse=True)


def launch_config(sandbox):
    params = sandbox.params()
    return params, json.loads(params["env"]["OPENCODE_CONFIG_CONTENT"])


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


async def test_request_temperature_marks_the_native_model_entry(setup):
    agent, sandbox = setup
    agent.config.sandbox_model_id = "gpt-5.5"
    response, _ = await episode(agent, sandbox, temperature=0.6)
    assert response.status == "completed"
    _, config = launch_config(sandbox)
    assert set(config["provider"]["nemo_gym"]["models"]) == {"gpt-5.5"}
    assert config["provider"]["nemo_gym"]["models"]["gpt-5.5"]["temperature"] is True
    assert config["agent"] == {name: {"temperature": 0.6} for name in BUILTIN_AGENTS}


@pytest.mark.parametrize("sampling", [{"temperature": 0.6}, {"top_p": 0.9}])
def test_request_sampling_is_refused_for_an_opencode_version_without_a_known_agent_list(setup, sampling):
    agent, sandbox = setup
    agent.config.opencode_version = "1.15.13"
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        refused = client.post(path, json={"input": "task", **sampling})
        assert refused.status_code == 422
        assert f"applies request {next(iter(sampling))} only with OpenCode 1.17.11" in refused.json()["detail"]
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
    _, config = launch_config(sandbox)
    assert "agent" not in config


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
    stage = index(events, VERSION_PROBE)
    assert index(events, "install_opencode_runtime.sh") < stage < index(events, "--receipt")
    assert positional(events[stage]) == [
        "/mnt/cache/rg",
        PIN,
        f"{directory}/cache/opencode/bin/.pinned-rg",
        f"{directory}/cache/opencode/bin/rg",
    ]
    params, _ = launch_config(sandbox)
    assert params["env"]["XDG_CACHE_HOME"] == f"{directory}/cache"


async def test_pinned_plugin_dependencies_unpack_into_the_private_config_directory(setup, tmp_path):
    agent, sandbox = setup
    agent.config.sandbox_plugin_dependencies_path = "/mnt/cache/opencode-plugin-dependencies.tar.gz"
    agent.config.sandbox_plugin_dependencies_sha256 = plugin_dependencies(sandbox, tmp_path)
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    directory = sandbox.directory
    copy, unpack = index(events, PINNED_COPY), index(events, UNPACK)
    assert index(events, "install_opencode_runtime.sh") < copy < unpack < index(events, "--receipt")
    assert positional(events[copy]) == [
        agent.config.sandbox_plugin_dependencies_path,
        agent.config.sandbox_plugin_dependencies_sha256,
        f"{directory}/config/.pinned-opencode",
    ]
    assert positional(events[unpack]) == [f"{directory}/config/.pinned-opencode", f"{directory}/config/opencode"]
    params, _ = launch_config(sandbox)
    assert params["env"]["XDG_CONFIG_HOME"] == f"{directory}/config"
    # Without a pinned rg the installer keeps its best-effort ripgrep install.
    assert not any(event.startswith("NG_OPENCODE_SKIP_RIPGREP") for event in events)


def test_a_dependency_archive_that_leaves_its_folder_refuses_the_seed(setup, tmp_path):
    agent, sandbox = setup
    agent.config.sandbox_plugin_dependencies_path = "/mnt/cache/opencode-plugin-dependencies.tar.gz"
    agent.config.sandbox_plugin_dependencies_sha256 = plugin_dependencies(sandbox, tmp_path, ("../escape", b"x"))
    response = seed_status(agent)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == (
        "Pinned plugin dependency archive rejected: archive member '../escape' leaves the target directory"
    )
    assert not any(UNPACK in call.args[0] for call in sandbox.exec.await_args_list)
    sandbox.disconnect.assert_awaited_once()
    assert not active_sessions(agent)


@pytest.mark.parametrize("kind", ["ripgrep", "plugin_dependencies"])
@pytest.mark.parametrize(("code", "status"), [(3, 422), (1, 500)])
def test_only_rejected_bundle_bytes_are_a_deterministic_setup_error(setup, kind, code, status):
    agent, sandbox = setup
    setattr(agent.config, f"sandbox_{kind}_path", "/mnt/cache/item")
    setattr(agent.config, f"sandbox_{kind}_sha256", PIN)

    async def staged(command, **kwargs):
        return result(code, stderr="SHA-256 mismatch: expected a, found b")

    intercept(sandbox, PINNED_COPY, staged)
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

    async def import_(command, **kwargs):
        directory = command.split("mkdir -p -- ", 1)[1].split("/home", 1)[0]
        assert command.endswith(f"exec {RUNTIME}/opencode import {IMPORT_PATH} </dev/null")
        assert "cd /app || exit 125\nunset OPENCODE_CONFIG_CONTENT\n" in command
        env = environment(command)
        assert env["XDG_DATA_HOME"] == f"{directory}/data" and env["PWD"] == "/app"
        assert "OPENCODE_CONFIG_CONTENT" not in env
        return result()

    intercept(sandbox, IMPORT, import_)
    # The store holds only the imported prefix until the run adds this activation's turn.
    imported = session_store(tmp_path / "imported.db", roots=["ses_prefix"], messages=["prefix-user", "prefix-reply"])
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
    # The host snapshot leaves previous messages out of the exported turns (see test_snapshot).
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
        if source.endswith("/opencode.db"):
            Path(destination).write_bytes((database if sandbox.started.is_set() else imported).read_bytes())
        else:
            await original(source, destination)

    sandbox.download = download
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    assert index(events, IMPORT_CHECK) < index(events, IMPORT) < index(events, "--receipt")
    params, _ = launch_config(sandbox)
    assert params["command"][-2:] == ["--session", "ses_prefix"]
    assert sandbox.snapshots == [
        {"harness_session_id": "ses_prefix", "previous_message_ids": ["prefix-user", "prefix-reply"]}
    ]
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
        return result(1)

    intercept(sandbox, IMPORT_CHECK, nothing_staged)
    events = record(sandbox)
    response, _ = await episode(agent, sandbox)
    assert response.status == "completed"
    assert not any(IMPORT in event for event in events)
    params, _ = launch_config(sandbox)
    assert "--session" not in params["command"]
    assert sandbox.snapshots == [{"harness_session_id": None, "previous_message_ids": []}]


@pytest.mark.parametrize(("code", "status"), [(1, 422), (125, 500), (127, 500), (137, 500)])
def test_failed_import_is_a_deterministic_setup_error_only_when_opencode_refused_it(setup, code, status):
    agent, sandbox = setup
    agent.config.session_import_path = IMPORT_PATH

    async def failed(command, **kwargs):
        return result(code, stderr="Error: invalid export")

    intercept(sandbox, IMPORT, failed)
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
        return result(1, stdout="exit 1: disk full") if len(exports) == 1 else result()

    intercept(sandbox, EXPORT_SESSION, first_fails)
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
        prepare = sandbox.exec.await_args_list[0].args[0]
        check = prepare.index(f"set -- /srv/exports\n{_CHECK_EXPORT_DIR}")
        assert prepare.index("OpenCode runtime/session storage overlaps") < check < prepare.index("mkdir -p ")


def test_resolved_export_dir_check_follows_links_in_the_sandbox(tmp_path):
    workdir = tmp_path / "task"
    (workdir / "out").mkdir(parents=True)
    inside = tmp_path / "inside"
    inside.symlink_to(workdir / "out", target_is_directory=True)
    above = tmp_path / "above"
    above.symlink_to(tmp_path, target_is_directory=True)
    cases = {inside: False, inside / "not-yet": False, above: False, workdir: False, tmp_path / "exports": True}
    for number, (export_dir, accepted) in enumerate(cases.items()):
        directory = tmp_path / "sessions" / str(number)
        command = _sandbox_prepare_command(
            str(workdir), str(directory), str(tmp_path / "runtime"), export_dir=str(export_dir)
        )
        checked = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, timeout=30)
        assert (checked.returncode == 0) is accepted, (export_dir, checked.stderr)
        # The check runs before the session directory exists.
        assert directory.is_dir() is accepted
        if not accepted:
            assert "OpenCode export_dir overlaps the task workdir" in checked.stderr


async def test_path_prefix_reaches_the_installer_and_every_opencode_process(setup, tmp_path):
    agent, sandbox = setup
    prefix = "/opt/gym runtime/bin"
    agent.config.sandbox_path_prefix = [prefix]
    agent.config.session_import_path = IMPORT_PATH
    agent.config.export_dir = EXPORT_DIR
    capture_observations(sandbox, tmp_path)
    events = record(sandbox)
    response, closed = await episode(agent, sandbox)
    assert response.status == "completed"
    assert shlex.split(events[index(events, "install_opencode_runtime.sh")])[-1] == prefix
    assert f"PATH={shlex.quote(prefix)}:$PATH\n" in launch_config(sandbox)[0]["setup"]
    for marker in (IMPORT, EXPORT_SESSION):
        assert f"PATH={shlex.quote(prefix)}:$PATH\n" in events[index(events, marker)], marker
    assert not any("python" in event for event in events)
    assert "export_dir_write_failed" not in gap_codes(closed.agent_observations)


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
        PLUGIN_CHECK,
        result(3, stdout="/opt/owner/missing.mjs\n"),
        "OpenCode plugin files are missing from the sandbox: /opt/owner/missing.mjs",
    ),
    "pinned ripgrep": (
        {"sandbox_ripgrep_path": "/mnt/cache/rg", "sandbox_ripgrep_sha256": PIN},
        VERSION_PROBE,
        result(3, stderr="/mnt/cache/rg --version exited 2: no"),
        "Pinned ripgrep executable rejected",
    ),
    "pinned plugin dependencies": (
        {"sandbox_plugin_dependencies_path": "/mnt/cache/deps.tar", "sandbox_plugin_dependencies_sha256": PIN},
        PINNED_COPY,
        result(3, stderr="SHA-256 mismatch: expected a, found b in /mnt/cache/deps.tar"),
        "Pinned plugin dependency archive rejected",
    ),
    "session import": (
        {"session_import_path": IMPORT_PATH},
        IMPORT,
        result(1, stderr="Error: invalid export"),
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
