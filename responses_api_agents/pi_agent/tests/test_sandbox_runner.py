# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Pi's staged activation through the real bash runner and supervisor, not a mocked receipt."""

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SUPERVISOR_SCRIPT, SandboxSession
from responses_api_agents.pi_agent.sandbox import PiSandboxSession
from responses_api_agents.pi_agent.tests.test_sandbox import seed


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision is required")


class LocalFiles:
    async def upload(self, source, destination):
        shutil.copyfile(source, destination)


# Generous enough for a loaded runner to start the harness chain; a test that needs the deadline fires it on
# purpose through the expire file after the harness is ready, rather than racing the harness start.
SUPERVISOR_TIMEOUT_SECONDS = 60
CLEANUP_TIMEOUT_SECONDS = 5


def wait_for(path):
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.01)


def launch(tmp_path, script, timeout=SUPERVISOR_TIMEOUT_SECONDS, env=None):
    """Stage ``script`` as Pi's CLI exactly as an activation does, then start it under the supervisor."""
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_FILE)
    state = PiSandboxSession(
        request=seed(),
        session=SandboxSession(sandbox=LocalFiles(), session_dir=str(tmp_path), workdir=str(tmp_path), harness="Pi"),
        runtime="/unused",
    )
    command = asyncio.run(state.stage_activation(command=["bash", "-c", script], env=env or {}, prompt="task input"))
    return subprocess.Popen(
        ["bash", str(SUPERVISOR_SCRIPT), "--timeout", str(timeout), "--cleanup-timeout", str(CLEANUP_TIMEOUT_SECONDS)]
        + ["--stop-file", str(tmp_path / "stop.request"), "--receipt", str(tmp_path / "cleanup.json")]
        + ["--expire-file", str(tmp_path / "expire.request"), "--", *command.argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def result(tmp_path, process):
    try:
        stdout, stderr = process.communicate(timeout=SUPERVISOR_TIMEOUT_SECONDS)
        assert process.returncode == 0, (stdout, stderr)
        return json.loads((tmp_path / "cleanup.json").read_text())
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def stop(tmp_path, process):
    """Stop a supervisor whose test failed before reading its result, so it cannot outlive the test."""
    if process is None or process.poll() is not None:
        return
    (tmp_path / "stop.request").touch()
    try:
        process.wait(timeout=CLEANUP_TIMEOUT_SECONDS + 10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def test_capture_stdin_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_TEST_INHERITED", "inherited")
    monkeypatch.setenv("PI_TEST_OVERRIDE", "old")
    script = (
        "printf '\\377\\n[]\\n'; "
        'printf \'{"prompt": "%s", "inherited": "%s", "override": "%s"}\' '
        '"$(cat)" "$PI_TEST_INHERITED" "$PI_TEST_OVERRIDE"'
    )
    summary = result(tmp_path, launch(tmp_path, script, env={"PI_TEST_OVERRIDE": "new"}))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 0
    recorded_at, event = json.loads((tmp_path / "events.jsonl").read_text())
    assert recorded_at > 0
    assert event == {"prompt": "task input", "inherited": "inherited", "override": "new"}
    assert set(json.loads((tmp_path / "runtime.json").read_text())) == {"hostname", "pid"}


@pytest.mark.parametrize("ending", ["natural", "timeout", "cancel"])
def test_task_descendants_survive_until_owner_teardown(tmp_path, ending):
    process = None
    try:
        # A completed tool leaves its detached service, which no longer descends from the CLI at the stop.
        # The detached child publishes its own PID only after setsid has left the harness group.
        tool = (
            "setsid bash -c 'echo $$ > child.pid; exec sleep 60' </dev/null >/dev/null 2>&1 & "
            "while [ ! -s child.pid ]; do sleep 0.01; done"
        )
        script = f"bash -c {shlex.quote(tool)}; touch ready; " + ("sleep 60" if ending != "natural" else "true")
        process = launch(tmp_path, script)
        if ending != "natural":
            wait_for(tmp_path / "ready")
            (tmp_path / ("stop.request" if ending == "cancel" else "expire.request")).touch()
        summary = result(tmp_path, process)
        assert summary["cleanup_confirmed"] is True
        assert summary["timed_out"] is (ending == "timeout")
        pid = int((tmp_path / "child.pid").read_text())
        assert Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    finally:
        stop(tmp_path, process)
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cli_failure_is_not_success(tmp_path):
    summary = result(tmp_path, launch(tmp_path, "exit 7"))
    assert summary["cleanup_confirmed"] is True
    assert summary["return_code"] == 7
