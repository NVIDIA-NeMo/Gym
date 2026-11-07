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

from nemo_gym.agents.utils.sandbox_session import HARNESS_RUNNER_SCRIPT, shell_assignments


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


# -- the model relay ---------------------------------------------------------------------------------------------

RELAY_CHECK = r"""
import json, os, socket, sys
status = json.load(open(sys.argv[1]))
socket.create_connection(("127.0.0.1", status["port"]), timeout=5).close()
comm = open("/proc/%d/comm" % status["pid"]).read().strip()
cmdline = open("/proc/%d/cmdline" % status["pid"]).read().replace("\0", " ")
print(os.environ.get("NEMO_GYM_MODEL_RELAY"), comm, "python" in cmdline.lower())
"""

RELAY_RESTART = r"""
import json, os, signal, sys, time
path = sys.argv[1]
first = json.load(open(path))
os.kill(first["pid"], signal.SIGKILL)
deadline = time.time() + 10
while time.time() < deadline:
    try:
        now = json.load(open(path))
    except (OSError, ValueError):
        now = {}
    if now.get("pid") not in (None, first["pid"]) and now.get("port") == first["port"]:
        print("restarted on", now["port"])
        break
    time.sleep(0.1)
"""


# A stand-in relay for the runner's tests, so they need no JavaScript runtime: it listens on the port it is given and
# writes the relay's status layout, which is all the runner reads.
STUB_RELAY = r"""
import argparse, os, signal, socket
parser = argparse.ArgumentParser()
for name in ("--port", "--endpoint", "--session-dir", "--status"):
    parser.add_argument(name)
args = parser.parse_args()
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", int(args.port)))
server.listen()
stat = open("/proc/self/stat").read()
status = {"pid": os.getpid(), "start": int(stat[stat.rindex(")") + 2 :].split()[19]), "port": server.getsockname()[1]}
with open(args.status + ".tmp", "w") as handle:
    handle.write("{" + ", ".join('"%s": %d' % item for item in status.items()) + "}")
os.replace(args.status + ".tmp", args.status)
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
while True:
    server.accept()[0].close()
"""


def _relay_params(tmp_path: Path, script: str, **values) -> tuple[Path, Path]:
    (tmp_path / "relay").mkdir()
    status = tmp_path / "relay" / "status.json"
    stub = tmp_path / "stub_relay.py"
    stub.write_text(STUB_RELAY)
    (tmp_path / "endpoint.json").write_text(json.dumps({"gen": 1, "upstream": "http://127.0.0.1:9/v1"}))
    defaults = {
        "command": [sys.executable, "-c", script, str(status)],
        "MODEL_RELAY_PORT": "0",
        "MODEL_RELAY_ENDPOINT": str(tmp_path / "endpoint.json"),
        "MODEL_RELAY_STATUS": str(status),
        "MODEL_RELAY_COMMAND": [sys.executable, str(stub)],
    }
    return _params(tmp_path, **(defaults | values)), status


def _process_ends(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][0]
        except (OSError, IndexError):
            return True
        if state in "ZX":
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the relay watchdog reads /proc")
def test_the_runner_starts_the_model_relay_under_a_neutral_name(tmp_path: Path) -> None:
    params, status = _relay_params(tmp_path, RELAY_CHECK)

    result = _run(params)

    assert result.returncode == 0, (result.stderr, (tmp_path / "stderr.log").read_text())
    assert (tmp_path / "stdout.log").read_text().split() == ["1", "gym-model-relay", "False"]
    assert _process_ends(json.loads(status.read_text())["pid"]), "the relay ends with the command"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the relay watchdog reads /proc")
def test_a_relay_that_dies_is_restarted_on_the_same_port(tmp_path: Path) -> None:
    params, _ = _relay_params(tmp_path, RELAY_RESTART)

    result = _run(params)

    assert result.returncode == 0, (result.stderr, (tmp_path / "stderr.log").read_text())
    assert (tmp_path / "stdout.log").read_text().startswith("restarted on ")
    assert "restarting it on port" in (tmp_path / "relay" / "relay.log").read_text()


@pytest.mark.skipif(not Path("/proc/uptime").exists(), reason="the runner times the relay start from /proc/uptime")
def test_a_freeze_while_the_relay_starts_does_not_use_up_its_wait(tmp_path: Path) -> None:
    # A checkpoint can stop the runner while it starts the relay, for longer than the wait. The relay must still
    # come up once the runner runs again, not leave the command on the fallback environment.
    script = 'import os; print(os.environ.get("NEMO_GYM_MODEL_RELAY"), os.environ.get("MODEL_BASE_URL"))'
    entered = tmp_path / "entered"
    slow_python = tmp_path / "slow-python"
    slow_python.write_text(f'#!/bin/sh\n: >{entered}\nsleep 1\nexec {sys.executable} "$@"\n')
    slow_python.chmod(0o755)
    params, _ = _relay_params(
        tmp_path,
        script,
        MODEL_RELAY_COMMAND=[str(slow_python), str(tmp_path / "stub_relay.py")],
        MODEL_RELAY_FALLBACK_ENV={"MODEL_BASE_URL": "http://model.example/v1"},
    )
    runner = subprocess.Popen(
        ["bash", str(HARNESS_RUNNER_SCRIPT), str(params)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not entered.exists():
            assert time.monotonic() < deadline, "the runner never started the relay"
            time.sleep(0.01)
        os.killpg(runner.pid, signal.SIGSTOP)
        time.sleep(11)
        os.killpg(runner.pid, signal.SIGCONT)
        _, stderr = runner.communicate(timeout=20)
    finally:
        if runner.poll() is None:
            os.killpg(runner.pid, signal.SIGKILL)
            runner.wait()

    assert runner.returncode == 0, stderr
    assert (tmp_path / "stdout.log").read_text().split() == ["1", "None"]
    assert not (tmp_path / "relay" / "unavailable").exists()


RELAY_COMMAND_CHECK = r"""
import json, os, socket, sys
status = json.load(open(sys.argv[1]))
socket.create_connection(("127.0.0.1", status["port"]), timeout=5).close()
comm = open("/proc/%d/comm" % status["pid"]).read().strip()
relay_env = open("/proc/%d/environ" % status["pid"], "rb").read().split(b"\0")
print(os.environ.get("NEMO_GYM_MODEL_RELAY"), comm, b"RELAY_ONLY=1" in relay_env, os.environ.get("RELAY_ONLY"))
"""


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the relay watchdog reads /proc")
def test_a_relay_command_runs_under_a_neutral_name_with_an_environment_of_its_own(tmp_path: Path) -> None:
    # A harness binary that runs the relay (OpenCode as Bun) needs a switch that must not reach the harness itself.
    params, status = _relay_params(tmp_path, RELAY_COMMAND_CHECK, MODEL_RELAY_COMMAND_ENV={"RELAY_ONLY": "1"})

    result = _run(params)

    assert result.returncode == 0, (result.stderr, (tmp_path / "stderr.log").read_text())
    assert (tmp_path / "stdout.log").read_text().split() == ["1", "gym-model-relay", "True", "None"]
    assert _process_ends(json.loads(status.read_text())["pid"]), "the relay ends with the command"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the relay watchdog reads /proc")
def test_a_relay_command_that_cannot_run_leaves_the_command_on_the_fallback_environment(tmp_path: Path) -> None:
    script = 'import os; print(os.environ.get("NEMO_GYM_MODEL_RELAY"), os.environ.get("MODEL_BASE_URL"))'
    params, _ = _relay_params(
        tmp_path,
        script,
        MODEL_RELAY_COMMAND=[str(tmp_path / "missing" / "opencode"), "model_relay.mjs"],
        MODEL_RELAY_FALLBACK_ENV={"MODEL_BASE_URL": "http://model.example/v1"},
    )

    result = _run(params)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stdout.log").read_text().split() == ["None", "http://model.example/v1"]
    assert (tmp_path / "relay" / "unavailable").exists()


JS_RUNTIME = os.environ.get("NEMO_GYM_TEST_JS_RUNTIME") or shutil.which("node") or shutil.which("bun")

JS_RELAY_CHECK = r"""
import json, os, socket, sys
status = json.load(open(sys.argv[1]))
socket.create_connection(("127.0.0.1", status["port"]), timeout=5).close()
argv0 = open("/proc/%d/cmdline" % status["pid"]).read().split("\0")[0]
comm = open("/proc/%d/comm" % status["pid"]).read().strip()
print(os.environ.get("NEMO_GYM_MODEL_RELAY"), os.path.basename(argv0), comm)
"""


@pytest.mark.skipif(JS_RUNTIME is None, reason="no JavaScript runtime (node or bun) on this machine")
@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="the relay watchdog reads /proc")
def test_the_runner_starts_the_javascript_relay(tmp_path: Path) -> None:
    params, status = _relay_params(
        tmp_path,
        JS_RELAY_CHECK,
        MODEL_RELAY_COMMAND=[JS_RUNTIME, str(HARNESS_RUNNER_SCRIPT.with_name("model_relay.mjs"))],
    )

    result = _run(params)

    assert result.returncode == 0, (result.stderr, (tmp_path / "stderr.log").read_text())
    relayed, argv0, comm = (tmp_path / "stdout.log").read_text().split()
    # Kills that match the runtime's name miss the relay. Bun keeps the link's name as the process name; Node names
    # its main thread, so only the command line shows the link there.
    assert relayed == "1" and argv0 == "gym-model-relay" and comm != Path(JS_RUNTIME).name
    assert _process_ends(json.loads(status.read_text())["pid"]), "the relay ends with the command"
