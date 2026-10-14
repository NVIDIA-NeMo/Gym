# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the uploaded script in an isolated interpreter, without Gym imports."""

import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


SUPERVISOR = Path(__file__).parents[2] / "nemo_gym/agent_utils/process_supervisor.py"


@pytest.mark.parametrize("harness", ["pi", "codex", "opencode", "openclaw"])
def test_capture_error_stops_direct_cli_before_wrapper_exits(tmp_path: Path, harness: str) -> None:
    runner = SUPERVISOR.parents[2] / f"responses_api_agents/{harness}_agent/sandbox_runner.py"
    driver = """
import json, os, pathlib, runpy, subprocess, sys
module = runpy.run_path(sys.argv[1])
children = []
class InterruptedWait(subprocess.Popen):
    interrupted = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        children.append(self)
    def wait(self, *args, **kwargs):
        if not self.interrupted:
            self.interrupted = True
            raise OSError('capture interrupted')
        return super().wait(*args, **kwargs)
subprocess.Popen = InterruptedWait
root = sys.argv[2]
try:
    try:
        module['run']({'directory': root, 'cwd': root, 'env': {}, 'prompt': 'task',
                       'command': [sys.executable, '-c', 'import os,time; os.close(1); time.sleep(60)']})
    except OSError as error:
        assert str(error) == 'capture interrupted', error
    else:
        raise AssertionError('capture failure did not propagate')
    assert children and all(child.poll() is not None for child in children), 'CLI is still running'
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", driver, str(runner), str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("timeout, cleanup", [(1, 0.1), (2700, 10), (21600, 100)])
def test_exec_timeout_reserves_all_cleanup_phases(timeout: float, cleanup: float) -> None:
    supervisor = runpy.run_path(str(SUPERVISOR))
    assert supervisor["exec_timeout"](timeout=timeout, cleanup_timeout=cleanup) == (
        timeout + supervisor["CLEANUP_PHASE_COUNT"] * cleanup + 30
    )


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_supervisor_rejects_invalid_deadlines(tmp_path: Path, timeout: str) -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(SUPERVISOR), "--timeout", timeout, "--receipt", str(tmp_path / "cleanup.json")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "finite and positive" in result.stderr
    assert not (tmp_path / "cleanup.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owner subreaper and /proc are required")
@pytest.mark.parametrize("ending", ["normal", "crash", "timeout", "cancel", "grace", "stubborn", "killed"])
@pytest.mark.parametrize("detached", [False, True])
def test_only_detached_services_survive_harness_group_termination(tmp_path: Path, ending: str, detached: bool) -> None:
    # The isolated parent emulates the sandbox owner and reaps every test process.
    # A completed tool leaves a service adopted before interruption. Only
    # explicitly detached services survive normal and interrupted exits.
    driver = r"""
import ctypes, json, os, pathlib, signal, subprocess, sys, time, urllib.request
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
supervisor, root, ending, detached = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4] == 'True'
service = root / 'service.py'
service.write_text("from http.server import HTTPServer, BaseHTTPRequestHandler\n"
                  "from pathlib import Path\n"
                  "class Handler(BaseHTTPRequestHandler):\n"
                  "    def do_GET(self):\n"
                  "        self.send_response(200); self.end_headers(); self.wfile.write(b'task solved')\n"
                  "server = HTTPServer(('127.0.0.1', 0), Handler)\n"
                  "Path('port').write_text(str(server.server_port))\n"
                  "server.serve_forever()\n")
tool = root / 'tool.py'
tool.write_text("import pathlib,subprocess,sys\n"
                "child = subprocess.Popen([sys.executable, 'service.py'], stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL, start_new_session=" + str(detached) + ")\n"
                "pathlib.Path('service.pid').write_text(str(child.pid))\n")
worker = root / 'worker.py'
worker.write_text("import os, pathlib, signal, subprocess, sys, time\n"
                 "def checkpoint(*_):\n"
                 "    pathlib.Path('checkpoint').write_text('saved before exit')\n"
                 "    raise SystemExit(0)\n"
                 "ending = sys.argv[1]\n"
                 "if ending in ('timeout', 'cancel', 'grace'): signal.signal(signal.SIGTERM, checkpoint)\n"
                 "if ending == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                 "pathlib.Path('worker.pid').write_text(str(os.getpid()))\n"
                 "subprocess.run([sys.executable, 'tool.py'], check=True)\n"
                 "deadline = time.monotonic() + 5\n"
                 "while not pathlib.Path('port').exists():\n"
                 "    assert time.monotonic() < deadline\n"
                 "    time.sleep(0.01)\n"
                 "pathlib.Path('ready').touch()\n"
                 "signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})\n"
                 "if ending == 'crash': raise SystemExit(7)\n"
                 "if ending == 'killed': os.kill(os.getpid(), signal.SIGKILL)\n"
                 "if ending != 'normal': time.sleep(60)\n")
launcher = [sys.executable, '-I']
timed_out = ending in ('timeout', 'grace', 'stubborn')
if timed_out:
    launcher += ['-c', "import os, signal, sys; signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM});"
                 " os.execv(sys.executable, [sys.executable, '-I', *sys.argv[1:]])"]
process = None
try:
    process = subprocess.Popen([*launcher, supervisor, '--timeout', '0.8' if timed_out else '10',
        '--cleanup-timeout', '0.5', '--receipt', str(root / 'cleanup.json'), '--',
        sys.executable, str(worker), ending], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ending == 'cancel':
        deadline = time.monotonic() + 5
        while not (root / 'ready').exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, (stdout, stderr)
    receipt = json.loads((root / 'cleanup.json').read_text())
    assert receipt['cleanup_confirmed'] is True, receipt
    assert receipt['timed_out'] is timed_out, receipt
    expected = {'normal': 0, 'crash': 7, 'grace': 0, 'stubborn': -signal.SIGKILL, 'killed': -signal.SIGKILL}.get(ending, 0)
    assert receipt['return_code'] == expected, receipt
    assert receipt['error'] is None, receipt
    if ending == 'grace':
        assert (root / 'checkpoint').read_text() == 'saved before exit'
    try:
        os.killpg(int((root / 'worker.pid').read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('supervised process group still present, including zombies')
    port = int((root / 'port').read_text())
    if detached:
        with urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=2) as response:
            assert response.read() == b'task solved'
    else:
        assert not pathlib.Path('/proc/' + (root / 'service.pid').read_text()).exists()
        try:
            urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=0.2)
        except OSError:
            pass
        else:
            raise AssertionError('same-group service survived')
finally:
    if process is not None and process.poll() is None:
        process.kill(); process.wait(timeout=5)
    children = pathlib.Path('/proc/self/task/%d/children' % os.getpid())
    for pid in children.read_text().split():
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        while True:
            os.waitpid(-1, 0)
    except ChildProcessError:
        pass
if (root / 'service.pid').exists():
    try:
        os.kill(int((root / 'service.pid').read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('Sandbox teardown left the task service alive')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", driver, str(SUPERVISOR), str(tmp_path), ending, str(detached)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=25,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
@pytest.mark.parametrize("cooperative", [True, False])
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("ending", ["timeout", "cancel"])
def test_foreground_tool_cannot_mutate_task_after_confirmed_receipt(
    tmp_path: Path, cooperative: bool, detached: bool, ending: str
) -> None:
    runner = SUPERVISOR.parents[2] / "responses_api_agents/pi_agent/sandbox_runner.py"
    # A separate owner reaps any escaped process even when this regression fails.
    driver = r"""
import ctypes, json, os, pathlib, signal, subprocess, sys, time
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
supervisor, runner, root, cooperative = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4] == 'True'
detached, ending = sys.argv[5] == 'True', sys.argv[6]
writer = "import pathlib,time\nwhile not pathlib.Path('release').exists(): time.sleep(.01)\npathlib.Path('mutated').touch()"
# A second detached generation exercises repeated adoption after the tool dies.
tool = ("import pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c'," + repr(writer) + "],start_new_session=True)\n"
        "pathlib.Path('writer.pid').write_text(str(child.pid))\nchild.wait()\n")
cli = ("import pathlib,signal,subprocess,sys\n"
       + ("" if cooperative else "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n")
       + "child=subprocess.Popen([sys.executable,'-c'," + repr(tool) + "], start_new_session=" + str(detached) + ")\n"
       + "pathlib.Path('tool.pid').write_text(str(child.pid))\nchild.wait()\n")
params = {'directory': str(root), 'cwd': str(root), 'env': {}, 'prompt': 'task',
          'command': [sys.executable, '-c', cli]}
(root / 'input.json').write_text(json.dumps(params))
process = None
try:
    process = subprocess.Popen([sys.executable, '-I', supervisor, '--timeout', '1' if ending == 'timeout' else '10', '--cleanup-timeout', '.3',
        '--receipt', str(root / 'cleanup.json'), '--', sys.executable, '-I', runner, str(root / 'input.json')],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 5
    while not (root / 'writer.pid').exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(.01)
    if ending == 'cancel': process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)
    receipt = json.loads((root / 'cleanup.json').read_text())
    assert receipt['cleanup_confirmed'] and receipt['timed_out'] == (ending == 'timeout'), receipt
    assert (root / 'tool.pid').exists(), 'foreground tool never started'
    (root / 'release').touch()
    time.sleep(.2)
    assert not (root / 'mutated').exists(), 'foreground tool wrote after termination receipt'
    assert not pathlib.Path('/proc/' + (root / 'tool.pid').read_text()).exists(), 'tool was not reaped'
    assert not pathlib.Path('/proc/' + (root / 'writer.pid').read_text()).exists(), 'writer was not reaped'
finally:
    if process is not None and process.poll() is None:
        process.kill(); process.wait(timeout=5)
    for pid in pathlib.Path('/proc/self/task/%d/children' % os.getpid()).read_text().split():
        try: os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError: pass
    try:
        while True: os.waitpid(-1, 0)
    except ChildProcessError: pass
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            driver,
            str(SUPERVISOR),
            str(runner),
            str(tmp_path),
            str(cooperative),
            str(detached),
            ending,
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
def test_missing_binary_preserves_launch_error(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SUPERVISOR),
            "--timeout",
            "1",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            str(tmp_path / "missing-worker"),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=5,
    )
    receipt = json.loads((tmp_path / "cleanup.json").read_text())
    assert result.returncode == 1
    assert receipt["cleanup_confirmed"] is False
    assert receipt["return_code"] is None
    assert "Harness launch failed" in receipt["error"] and "missing-worker" in receipt["error"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
def test_provider_group_kill_leaves_supervisor_alive_to_reap_worker(tmp_path: Path) -> None:
    # Keep the shell as the provider group leader; the supervisor must leave that group.
    process = subprocess.Popen(
        [
            "sh",
            "-c",
            '"$@" & wait',
            "provider",
            sys.executable,
            "-I",
            str(SUPERVISOR),
            # A deadline that fired before the group kill would skip the case under test.
            # The test stops the supervisor itself once the group is gone.
            "--timeout",
            "60",
            "--cleanup-timeout",
            "0.2",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            sys.executable,
            "-c",
            # Publish both PIDs by rename so the test never reads a created but still empty file.
            "import json, os, pathlib, time; pathlib.Path('pids.tmp').write_text(json.dumps([os.getppid(), os.getpid()]));"
            " os.replace('pids.tmp', 'pids.json'); time.sleep(60)",
        ],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pids = []
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "pids.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        # The worker exists, so the supervisor has already left the provider group.
        pids = json.loads((tmp_path / "pids.json").read_text())
        supervisor_pid, worker_pid = pids
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        # SIGKILL takes effect before any later signal, so a supervisor killed with the group cannot handle this one.
        os.kill(supervisor_pid, signal.SIGTERM)
        process.communicate(timeout=10)
        receipt = json.loads((tmp_path / "cleanup.json").read_text())
        assert receipt["timed_out"] is False
        assert receipt["cleanup_confirmed"] is True
        assert receipt["error"] is None
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for pid in reversed(pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
def test_sigterm_during_spawn_does_not_lose_child_handle(tmp_path):
    # Deliver SIGTERM after the real child exists but before Popen returns to _supervise().
    # Isolate signal handlers from pytest, and always reap the test child.
    driver = """
import json, os, runpy, signal, subprocess, sys
runner = runpy.run_path(sys.argv[1])
spawn = subprocess.Popen
children = []
def interrupted_spawn(*args, **kwargs):
    child = spawn(*args, **kwargs)
    children.append(child)
    os.kill(os.getpid(), signal.SIGTERM)
    return child
subprocess.Popen = interrupted_spawn
try:
    summary = runner['_supervise'](
        [sys.executable, '-c', 'import time; time.sleep(60)'], timeout=5, cleanup_timeout=0.2,
    )
    summary['child_alive'] = any(child.poll() is None for child in children)
    print(json.dumps(summary))
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, str(SUPERVISOR), str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["timed_out"] is False
    assert summary["cleanup_confirmed"] is True
    assert summary["error"] is None
    assert summary["child_alive"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
def test_lost_spawn_handle_requires_sandbox_teardown(tmp_path):
    # A lost handle cannot confirm termination; the owner must destroy the sandbox.
    driver = """
import json, os, runpy, subprocess, sys
runner = runpy.run_path(sys.argv[1])
spawn = subprocess.Popen
children = []
def lost_handle(*args, **kwargs):
    child = spawn(*args, **kwargs)
    children.append(child)
    raise OSError('launch handle lost after process creation')
subprocess.Popen = lost_handle
try:
    summary = runner['_supervise'](
        [sys.executable, '-c', 'import time; time.sleep(60)'], timeout=5, cleanup_timeout=0.2,
    )
    try:
        os.kill(children[0].pid, 0)
        summary['child_alive'] = True
    except ProcessLookupError:
        summary['child_alive'] = False
    print(json.dumps(summary))
finally:
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, str(SUPERVISOR), str(tmp_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["return_code"] is None
    assert summary["error"] == "Harness launch failed: launch handle lost after process creation"
    assert summary["cleanup_confirmed"] is False
    assert summary["child_alive"] is True


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc zombie state")
def test_zombie_group_owned_by_other_parent_is_terminated(tmp_path: Path) -> None:
    # Keep our own child unreaped. The helper cannot reap another parent's zombie,
    # but it must not confuse that zombie with a process still able to write.
    child = subprocess.Popen([sys.executable, "-I", "-c", "pass"], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        stat = Path(f"/proc/{child.pid}/stat")
        while stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        os.killpg(child.pid, 0)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import runpy,sys; runpy.run_path(sys.argv[1])['_reap_process_group'](int(sys.argv[2]), 0.1)",
                str(SUPERVISOR),
                str(child.pid),
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
        )
        assert completed.returncode == 0, completed.stderr
        assert stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        child.wait(timeout=5)


def test_unreadable_group_state_does_not_confirm_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = runpy.run_path(str(SUPERVISOR))
    monkeypatch.setattr(os, "killpg", lambda *_: None)
    monkeypatch.setattr(os, "waitpid", lambda *_: (0, 0))
    monkeypatch.setattr(Path, "iterdir", lambda _: iter([Path("/proc/123")]))

    def unreadable(*_: object) -> str:
        raise PermissionError("Cannot read process state")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(PermissionError, match="Cannot read process state"):
        supervisor["_reap_process_group"](123, 0.1)


@pytest.mark.parametrize("state, alive", [("Z", False), ("X", False), ("S", True)])
def test_group_scan_ignores_vanished_and_unrelated_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, alive: bool
) -> None:
    scan = runpy.run_path(str(SUPERVISOR))["_group_has_live_processes"]
    # A process may disappear between listing /proc and reading its stat file.
    for name in ("self", "100", "101", "102"):
        (tmp_path / name).mkdir()
    (tmp_path / "101/stat").write_text("101 (unrelated) R 1 999 0")
    (tmp_path / "102/stat").write_text(f"102 (name with ) parentheses) {state} 1 123 0")
    monkeypatch.setitem(scan.__globals__, "Path", lambda _: tmp_path)

    assert scan(123) is alive


def test_forced_cleanup_drains_successive_adoptions_but_preserves_existing_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reap = runpy.run_path(str(SUPERVISOR))["_reap_process_group"]
    children = {101, 202, 204}
    killed: list[int] = []
    reaped: list[int] = []

    def absent_group(*_: object) -> None:
        raise ProcessLookupError

    def kill(pid: int, sig: int) -> None:
        assert sig == signal.SIGKILL
        killed.append(pid)
        if pid == 204:
            children.remove(pid)
            raise ProcessLookupError  # Exited before the signal arrived.

    def waitpid(pid: int, options: int) -> tuple[int, int]:
        assert options == os.WNOHANG
        if pid == -123:
            raise ChildProcessError
        children.remove(pid)
        reaped.append(pid)
        if pid == 202:
            children.add(303)  # Killing this tool adopts another detached generation.
        return pid, 0

    monkeypatch.setitem(
        reap.__globals__, "os", SimpleNamespace(killpg=absent_group, kill=kill, waitpid=waitpid, WNOHANG=os.WNOHANG)
    )
    monkeypatch.setitem(reap.__globals__, "_child_pids", lambda: set(children))

    reap(123, 1, preserved_children={101})

    assert set(killed) == {202, 204, 303}
    assert reaped == [202, 303]
    assert children == {101}


@pytest.mark.parametrize("remaining", ["live_group", "unreaped_group", "adopted_tool"])
def test_cleanup_deadline_rejects_each_kind_of_unfinished_process(
    monkeypatch: pytest.MonkeyPatch, remaining: str
) -> None:
    reap = runpy.run_path(str(SUPERVISOR))["_reap_process_group"]
    clock = iter([0.0, 0.5, 1.0])
    sleeps: list[float] = []

    def killpg(*_: object) -> None:
        if remaining != "live_group":
            raise ProcessLookupError

    def waitpid(pid: int, _: int) -> tuple[int, int]:
        if pid == -123 and remaining != "unreaped_group":
            raise ChildProcessError
        return 0, 0  # Still present after a nonblocking reap attempt.

    monkeypatch.setitem(
        reap.__globals__,
        "os",
        SimpleNamespace(killpg=killpg, kill=lambda *_: None, waitpid=waitpid, WNOHANG=os.WNOHANG),
    )
    monkeypatch.setitem(reap.__globals__, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=sleeps.append))
    monkeypatch.setitem(reap.__globals__, "_group_has_live_processes", lambda _: remaining == "live_group")
    monkeypatch.setitem(reap.__globals__, "_child_pids", lambda: {101, 202} if remaining == "adopted_tool" else {101})

    with pytest.raises(TimeoutError, match="have not terminated"):
        reap(123, 1, preserved_children={101})
    assert sleeps == [0.01]


@pytest.mark.parametrize("boundary", ["stop_marker", "unsupported_platform", "subreaper_failure"])
def test_supervisor_does_not_launch_without_startup_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    supervise = runpy.run_path(str(SUPERVISOR))["_supervise"]
    launched: list[list[str]] = []
    stop = tmp_path / "stop"
    if boundary == "stop_marker":
        stop.touch()
    monkeypatch.setitem(
        supervise.__globals__, "signal", SimpleNamespace(signal=lambda *_: None, SIGTERM=15, SIG_IGN=1)
    )
    monkeypatch.setitem(
        supervise.__globals__,
        "sys",
        SimpleNamespace(platform="darwin" if boundary == "unsupported_platform" else "linux"),
    )
    monkeypatch.setitem(supervise.__globals__, "os", SimpleNamespace(getpgrp=lambda: 1, getpid=lambda: 1))
    monkeypatch.setitem(
        supervise.__globals__,
        "ctypes",
        SimpleNamespace(
            CDLL=lambda *_, **__: SimpleNamespace(prctl=lambda *_: -1 if boundary == "subreaper_failure" else 0),
            get_errno=lambda: 1,
        ),
    )
    monkeypatch.setitem(
        supervise.__globals__, "subprocess", SimpleNamespace(Popen=lambda command, **_: launched.append(command))
    )

    receipt = supervise(["worker"], timeout=1, stop_path=stop)

    assert launched == []
    assert receipt["cleanup_confirmed"] is True  # No process was started.
    assert receipt["return_code"] is None
    assert receipt["timed_out"] is False
    if boundary == "stop_marker":
        assert receipt["error"] is None
    elif boundary == "unsupported_platform":
        assert receipt["error"] == "Sandbox process supervision requires Linux"
    else:
        assert "Cannot supervise harness process group" in receipt["error"]


def test_supervisor_requires_a_command_before_writing_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main = runpy.run_path(str(SUPERVISOR))["main"]
    receipt = tmp_path / "cleanup.json"
    monkeypatch.setattr(sys, "argv", [str(SUPERVISOR), "--timeout", "1", "--receipt", str(receipt), "--"])

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "a harness command is required" in capsys.readouterr().err
    assert not receipt.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
def test_missing_pre_stop_snapshot_does_not_confirm_termination(tmp_path: Path) -> None:
    driver = """
import json,runpy,sys
supervise = runpy.run_path(sys.argv[1])['_supervise']
def unavailable():
    raise OSError('children unavailable')
supervise.__globals__['_child_pids'] = unavailable
receipt = supervise([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=.1, cleanup_timeout=.1)
assert receipt['cleanup_confirmed'] is False, receipt
assert 'Cannot identify pre-stop adopted children' in receipt['error'], receipt
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", driver, str(SUPERVISOR)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
