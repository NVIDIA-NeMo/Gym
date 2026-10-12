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
    assert supervisor["exec_timeout"](timeout=timeout, cleanup_timeout=cleanup) > timeout + 2 * cleanup


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
def test_sandbox_owner_tears_down_services_after_harness_exit(tmp_path: Path, ending: str, detached: bool) -> None:
    # The isolated parent emulates the sandbox owner and reaps every test process.
    # Service descendants must survive both normal and interrupted harness exits.
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
worker = root / 'worker.py'
worker.write_text("import os, pathlib, signal, subprocess, sys, time\n"
                 "def checkpoint(*_):\n"
                 "    pathlib.Path('checkpoint').write_text('saved before exit')\n"
                 "    raise SystemExit(0)\n"
                 "ending = sys.argv[1]\n"
                 "if ending in ('timeout', 'cancel', 'grace'): signal.signal(signal.SIGTERM, checkpoint)\n"
                 "if ending == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                 "pathlib.Path('worker.pid').write_text(str(os.getpid()))\n"
                 "child = subprocess.Popen([sys.executable, 'service.py'], stdout=subprocess.DEVNULL, "
                 "stderr=subprocess.DEVNULL, start_new_session=" + str(detached) + ")\n"
                 "pathlib.Path('service.pid').write_text(str(child.pid))\n"
                 "deadline = time.monotonic() + 5\n"
                 "while not pathlib.Path('port').exists():\n"
                 "    assert child.poll() is None and time.monotonic() < deadline\n"
                 "    time.sleep(0.01)\n"
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
        while not (root / 'port').exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == (1 if ending in ('stubborn', 'killed') else 0), (stdout, stderr)
    receipt = json.loads((root / 'cleanup.json').read_text())
    assert receipt['cleanup_confirmed'] is (ending not in ('stubborn', 'killed')), receipt
    assert receipt['timed_out'] is timed_out, receipt
    expected = {'normal': 0, 'crash': 7, 'grace': 0, 'stubborn': -signal.SIGKILL, 'killed': -signal.SIGKILL}.get(ending, 0)
    assert receipt['return_code'] == expected, receipt
    if ending in ('stubborn', 'killed'):
        assert 'sandbox teardown required' in receipt['error'], receipt
    else:
        assert receipt['error'] is None, receipt
    if ending == 'grace':
        assert (root / 'checkpoint').read_text() == 'saved before exit'
    try:
        os.kill(int((root / 'worker.pid').read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('supervised worker still running')
    port = int((root / 'port').read_text())
    with urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=2) as response:
        assert response.read() == b'task solved'
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
        assert receipt["cleanup_confirmed"] is False
        assert "terminated by a signal" in receipt["error"]
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
    assert summary["cleanup_confirmed"] is False
    assert "terminated by a signal" in summary["error"]
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
    assert summary["error"] == "launch handle lost after process creation"
    assert summary["cleanup_confirmed"] is False
    assert summary["child_alive"] is True
