# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the uploaded bash harness runner as the sandbox runs it."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agents import HARNESS_RUNNER_SCRIPT, shell_assignments


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def _params(tmp_path: Path, **values) -> Path:
    (tmp_path / "prompt.txt").write_text("the task\n")
    params = tmp_path / "params.sh"
    defaults = {
        "cwd": str(tmp_path),
        "env": {},
        "stdin": str(tmp_path / "prompt.txt"),
        "stdout": str(tmp_path / "stdout.log"),
        "stderr": str(tmp_path / "stderr.log"),
    }
    params.write_text(shell_assignments(defaults | values))
    return params


def _run(params: Path, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HARNESS_RUNNER_SCRIPT), str(params)], capture_output=True, text=True, timeout=20, **kwargs
    )


def test_runs_the_command_with_files_env_and_cwd(tmp_path: Path) -> None:
    work = tmp_path / "work's dir"
    work.mkdir()
    script = 'cat; pwd; echo "$GYM_VALUE|${DROPPED_TOKEN-unset}|${KEPT-unset}"; echo diagnostics >&2; exit 3'
    params = _params(
        tmp_path,
        cwd=str(work),
        command=["bash", "-c", script],
        env={"GYM_VALUE": "it's $(set)"},
        unset_env=["DROPPED_*"],
    )
    result = _run(params, env=os.environ | {"DROPPED_TOKEN": "secret", "KEPT": "yes"})
    assert result.returncode == 3, result.stderr
    assert (tmp_path / "stdout.log").read_text() == f"the task\n{work}\nit's $(set)|unset|yes\n"
    assert (tmp_path / "stderr.log").read_text() == "diagnostics\n"
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert set(runtime) == {"hostname", "pid"}


def test_setup_runs_before_the_harness_environment_applies(tmp_path: Path) -> None:
    params = _params(
        tmp_path,
        setup=f'echo "$HOME" > {tmp_path}/task-home',
        command=["bash", "-c", 'echo "$HOME"'],
        env={"HOME": "/private/home"},
    )
    result = _run(params, env=os.environ | {"HOME": "/task/home"})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "task-home").read_text() == "/task/home\n"
    assert (tmp_path / "stdout.log").read_text() == "/private/home\n"


def test_events_keep_json_objects_with_the_time_they_were_seen(tmp_path: Path) -> None:
    script = 'printf \'{"type": "a"}\\nnot json\\n\\n{"type": "b"}\\n{"type": "partial"}\'; sleep 0.2'
    params = _params(tmp_path, command=["bash", "-c", script], events=str(tmp_path / "events.jsonl"))
    before = time.time()
    result = _run(params)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event for _, event in events] == [{"type": "a"}, {"type": "b"}, {"type": "partial"}]
    assert all(before - 1 <= seen <= time.time() for seen, _ in events)


def test_events_keep_numeric_times_and_crlf_lines_without_epochrealtime(tmp_path: Path) -> None:
    # Bash < 5 has no EPOCHREALTIME and bash < 4.2 no printf strftime; both still stamp numbers.
    script = 'printf \'{"type": "a"}\\r\\n{"type": "partial"}\'; sleep 0.2'
    params = _params(tmp_path, command=["bash", "-c", script], events=str(tmp_path / "events.jsonl"))
    result = subprocess.run(
        ["bash", "-c", 'unset EPOCHREALTIME; . "$0" "$1"', str(HARNESS_RUNNER_SCRIPT), str(params)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event for _, event in events] == [{"type": "a"}, {"type": "partial"}]
    assert all(isinstance(seen, (int, float)) and seen > 0 for seen, _ in events)


def test_events_mode_stamps_long_lines_in_linear_time(tmp_path: Path) -> None:
    # Pi's agent_end event carries the whole conversation; a quadratic strip would outlast the deadline.
    code = "import json; print(json.dumps({'text': 'a' * 2_000_000})); print('b' * 1_000_000)"
    params = _params(tmp_path, command=[sys.executable, "-c", code], events=str(tmp_path / "events.jsonl"))
    started = time.monotonic()
    result = _run(params, env=os.environ | {"LC_ALL": "C.UTF-8"})
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 10
    (event,) = [json.loads(line)[1] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert event == {"text": "a" * 2_000_000}


def test_inherited_errexit_does_not_stop_the_runner(tmp_path: Path) -> None:
    (tmp_path / "bash_env").write_text("set -euo pipefail\n")
    script = 'echo "not json"; echo \'{"type": "ok"}\'; exit 3'
    params = _params(tmp_path, command=["bash", "-c", script], events=str(tmp_path / "events.jsonl"))
    result = _run(params, env=os.environ | {"BASH_ENV": str(tmp_path / "bash_env")})
    assert result.returncode == 3, result.stderr
    events = [json.loads(line)[1] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == [{"type": "ok"}]


@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize(
    "failure, return_codes",
    [("setup", {125}), ("cwd", {125}), ("stdin", {126}), ("command", {126, 127})],
)
def test_runner_failures_are_told_apart_from_the_command_status(
    tmp_path: Path, failure: str, return_codes: set[int], events: bool
) -> None:
    # A command that cannot start must not fall through into events mode or run anyway.
    values = {"command": ["touch", str(tmp_path / "ran")]}
    if failure == "setup":
        values["setup"] = "exit 1"
    elif failure == "cwd":
        values["cwd"] = str(tmp_path / "missing")
    elif failure == "stdin":
        values["stdin"] = str(tmp_path / "missing")
    else:
        values["command"] = [str(tmp_path / "missing")]
    if events:
        values["events"] = str(tmp_path / "events.jsonl")
    params = _params(tmp_path, **values)
    (tmp_path / "stdout.log").write_text("earlier output\n")
    result = _run(params)
    assert result.returncode in return_codes, result.stderr
    assert not (tmp_path / "ran").exists()
    # Earlier output survives unless the runner got as far as opening stdout.
    if failure in ("setup", "cwd") or (failure == "stdin" and not events):
        assert (tmp_path / "stdout.log").read_text() == "earlier output\n"


def test_inherited_runner_names_neither_configure_the_runner_nor_reach_the_cli(tmp_path: Path) -> None:
    params = _params(tmp_path, command=["bash", "-c", 'echo "${stdout-unset}|${setup-unset}"'])
    inherited = {"setup": f"touch {tmp_path}/evaluated", "events": str(tmp_path / "events.jsonl"), "stdout": "leak"}
    result = _run(params, env=os.environ | inherited)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stdout.log").read_text() == "unset|unset\n"
    assert not (tmp_path / "evaluated").exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_events_mode_keeps_runner_descriptors_out_of_the_cli(tmp_path: Path) -> None:
    script = 'echo \'{"type": "injected"}\' >&4; echo \'{"type": "ok"}\''
    params = _params(tmp_path, command=["bash", "-c", script], events=str(tmp_path / "events.jsonl"))
    result = _run(params)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line)[1] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == [{"type": "ok"}]
    assert "Bad file descriptor" in (tmp_path / "stderr.log").read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process semantics")
def test_events_mode_forwards_term_and_ignores_a_detached_stdout_holder(tmp_path: Path) -> None:
    # A detached tool keeps stdout open; the runner must still finish when the CLI does.
    script = (
        "setsid sleep 30 & echo $! > holder.pid; "
        'trap \'echo "{\\"type\\": \\"stopped\\"}"; exit 0\' TERM; echo \'{"type": "ready"}\'; '
        "while :; do sleep 0.05; done"
    )
    params = _params(tmp_path, command=["bash", "-c", script], events=str(tmp_path / "events.jsonl"))
    process = subprocess.Popen(["bash", str(HARNESS_RUNNER_SCRIPT), str(params)], stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while "ready" not in ((tmp_path / "events.jsonl").read_text() if (tmp_path / "events.jsonl").exists() else ""):
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0, process.stderr.read()
        events = [json.loads(line)[1] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert events == [{"type": "ready"}, {"type": "stopped"}]
    finally:
        if process.poll() is None:
            process.kill()
        if (tmp_path / "holder.pid").exists():
            try:
                os.kill(int((tmp_path / "holder.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
