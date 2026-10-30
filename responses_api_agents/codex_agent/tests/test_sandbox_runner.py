# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Codex's staged activation through the real bash runner, Node launcher and supervisor."""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agent_utils.sandbox_session import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SandboxSession
from nemo_gym.agent_utils.supervisor_client import SUPERVISOR_SCRIPT
from responses_api_agents.codex_agent.sandbox import LAUNCHER_FILE, RAISE_OOM_SCORE_ADJ, CodexSandboxSession
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import seed


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("node") is None, reason="Linux process supervision and Node are required"
)


class LocalFiles:
    async def upload(self, source, destination):
        shutil.copyfile(source, destination)


def stage(tmp_path, command, *, env=None, timeout=3, model_base_url=None, **activation_fields):
    """Stage ``command`` as the Codex CLI exactly as an activation does, and return the runner argv."""
    runtime = tmp_path / "runtime"
    (runtime / "node/bin").mkdir(parents=True)
    (runtime / "node/bin/node").symlink_to(shutil.which("node"))
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_FILE)
    shutil.copyfile(Path(__file__).parents[1] / LAUNCHER_FILE, tmp_path / LAUNCHER_FILE)
    state = CodexSandboxSession(
        request=seed(),
        session=SandboxSession(
            sandbox=LocalFiles(), session_dir=str(tmp_path), workdir=str(tmp_path), harness="Codex"
        ),
        runtime=str(runtime),
    )
    payload = {
        "directory": str(tmp_path),
        "command": command,
        "model_base_url": model_base_url,
        "invocation_id": "invocation",
        "timeout": timeout,
        "prompt": "task input",
        "cwd": str(tmp_path),
        "env": env or {},
        **activation_fields,
    }
    return asyncio.run(state.stage_activation(payload)).argv


def launch(tmp_path, command, timeout=3, **options):
    argv = stage(tmp_path, command, timeout=timeout, **options)
    return subprocess.Popen(
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", str(timeout), "--cleanup-timeout", "0.5"]
        + ["--stop-file", str(tmp_path / "stop.request"), "--receipt", str(tmp_path / "cleanup.json")]
        + ["--", *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def result(tmp_path, process):
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        return json.loads((tmp_path / "cleanup.json").read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def wait_for(path):
    for _ in range(500):
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} never appeared")


def test_capture_stdin_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_PROVIDER_API_KEY", "must-not-inherit")
    # Would stop Node from starting the launcher at all.
    monkeypatch.setenv("NODE_OPTIONS", "--require /nonexistent/preload.js")
    task_environment = {"HOME": str(tmp_path / "task-home"), "XDG_DATA_HOME": str(tmp_path / "task-data")}
    for key, value in task_environment.items():
        monkeypatch.setenv(key, value)
    script = (
        'printf \'{"prompt": "%s", "key": "%s", "node_options": "%s", "home": "%s", "data": "%s", "codex": "%s"}\\n\' '
        '"$(cat)" "${SANDBOX_PROVIDER_API_KEY-unset}" "${NODE_OPTIONS-unset}" "$HOME" "$XDG_DATA_HOME" "$CODEX_HOME"'
    )
    process = launch(tmp_path, ["bash", "-c", script], env={"CODEX_HOME": "/codex-home"})
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0, (tmp_path / "stderr.log").read_text()
    recorded_at, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert recorded_at > 0
    assert event == {
        "prompt": "task input",
        "key": "unset",
        "node_options": "unset",
        "home": task_environment["HOME"],
        "data": task_environment["XDG_DATA_HOME"],
        "codex": "/codex-home",
    }
    assert set(json.loads((tmp_path / "runtime.json").read_text())) == {"hostname", "pid"}


@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
def test_task_descendants_survive_until_owner_teardown(tmp_path, ending):
    try:
        # A completed tool leaves its detached service, which no longer descends from Codex at the stop.
        tool = "setsid sleep 60 </dev/null >/dev/null 2>&1 & echo $! > child.pid"
        script = f"bash -c '{tool}'; touch ready; " + ("sleep 60" if ending != "natural" else "true")
        process = launch(tmp_path, ["bash", "-c", script])
        if ending == "cancel":
            wait_for(tmp_path / "ready")
            (tmp_path / "stop.request").touch()
        summary = result(tmp_path, process)
        assert summary["cleanup_confirmed"] is True
        assert summary["timed_out"] is (ending == "timeout")
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_stop_reaches_codex_through_the_launcher(tmp_path):
    script = 'trap \'echo "{\\"type\\": \\"stopped\\"}"; exit 0\' TERM; touch ready; while :; do sleep 0.05; done'
    process = launch(tmp_path, ["bash", "-c", script])
    wait_for(tmp_path / "ready")
    (tmp_path / "stop.request").touch()
    summary = result(tmp_path, process)
    assert summary["cleanup_confirmed"] is True
    # Codex handled TERM and exited by itself, so its last events are kept.
    assert summary["return_code"] == 0
    assert json.loads((tmp_path / "events.jsonl").read_text())[1] == {"type": "stopped"}


@pytest.mark.parametrize(
    ("command", "status"),
    [
        (["bash", "-c", "exit 7"], 7),
        (["bash", "-c", "kill -TERM $$"], 128 + signal.SIGTERM),
        (["/missing/codex"], 127),
    ],
)
def test_codex_status_reaches_the_receipt_as_a_shell_status(tmp_path, command, status):
    summary = result(tmp_path, launch(tmp_path, command))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == status


def test_stop_marker_prevents_process_launch(tmp_path):
    (tmp_path / "stop.request").touch()
    summary = result(tmp_path, launch(tmp_path, ["bash", "-c", "touch should-not-exist"]))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] is None
    assert not (tmp_path / "should-not-exist").exists()


def test_inherited_environment_is_opt_in_and_keeps_explicit_values(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_VARIABLE", "from-task-image")
    monkeypatch.setenv("HOME", "/root")
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=2048")
    script = (
        'printf \'{"task": "%s", "home": "%s", "codex": "%s", "path": "%s", "node_options": "%s"}\\n\' '
        '"$TASK_VARIABLE" "$HOME" "$CODEX_HOME" "$PATH" "$NODE_OPTIONS"'
    )
    process = launch(tmp_path, ["bash", "-c", script], inherit_env=True, env={"CODEX_HOME": "/tmp/session/.codex"})
    assert result(tmp_path, process)["cleanup_confirmed"]
    _, event = json.loads((tmp_path / "events.jsonl").read_text())
    # The task's whole environment, Node options included, as a plain codex exec in the task sees it.
    assert event == {
        "task": "from-task-image",
        "home": "/root",
        "codex": "/tmp/session/.codex",
        "path": os.environ["PATH"],
        "node_options": "--max-old-space-size=2048",
    }


@pytest.mark.parametrize(("start", "expected"), [("0", "1000"), ("500", "1000"), ("-997", "0"), ("-1000", "0")])
def test_oom_score_setup_raises_the_score_and_never_lowers_it(tmp_path, start, expected):
    path = tmp_path / "oom_score_adj"
    path.write_text(start + "\n")
    setup = RAISE_OOM_SCORE_ADJ.replace("/proc/self/oom_score_adj", str(path))
    ran = subprocess.run(["bash", "-c", setup], capture_output=True, text=True, timeout=10)
    assert (ran.returncode, ran.stderr) == (0, f"oom_score_adj: {start} -> {expected}\n")
    assert path.read_text().strip() == expected
    missing = subprocess.run(
        ["bash", "-c", setup.replace(str(path), str(tmp_path / "missing"))], capture_output=True, text=True, timeout=10
    )
    assert (missing.returncode, missing.stderr) == (0, "oom_score_adj: unchanged\n")


@pytest.mark.parametrize("raise_score", [False, True])
def test_codex_and_its_tools_inherit_the_raised_oom_score(tmp_path, raise_score):
    start = int(Path("/proc/self/oom_score_adj").read_text())
    # Stands in for Codex: a tool it starts reports the score it inherited.
    process = launch(tmp_path, ["bash", "-c", "cat /proc/self/oom_score_adj >score"], raise_oom_score_adj=raise_score)
    assert result(tmp_path, process)["return_code"] == 0
    assert int((tmp_path / "score").read_text()) == ((0 if start < 0 else 1000) if raise_score else start)
    # The runner reports the change in the supervisor's log, not in Codex's stderr.
    assert "oom_score_adj:" not in (tmp_path / "stderr.log").read_text()
