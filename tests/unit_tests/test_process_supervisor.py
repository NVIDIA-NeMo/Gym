# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the uploaded script in an isolated interpreter, without Gym imports."""

import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


SUPERVISOR = Path(__file__).parents[2] / "nemo_gym/agent_utils/process_supervisor.py"


@pytest.mark.parametrize("timeout, cleanup", [(1, 0.1), (2700, 10), (21600, 100)])
def test_exec_timeout_reserves_all_cleanup_phases(timeout: float, cleanup: float) -> None:
    supervisor = runpy.run_path(str(SUPERVISOR))
    assert supervisor["exec_timeout"](timeout=timeout, cleanup_timeout=cleanup) > timeout + 3 * cleanup


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")
@pytest.mark.parametrize(
    "ending,preserve",
    [
        (ending, preserve)
        for ending in ("normal", "crash", "timeout", "cancel", "grace")
        for preserve in (False, True)
        if not (ending == "normal" and preserve)
    ],
)
def test_supervisor_reaps_detached_tools_and_preserves_term_grace(tmp_path: Path, ending: str, preserve: bool) -> None:
    uploaded = tmp_path / "process_supervisor.py"
    shutil.copyfile(SUPERVISOR, uploaded)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, pathlib, signal, subprocess, sys, time\n"
        "def checkpoint(*_):\n"
        "    time.sleep(0.1)\n"
        "    pathlib.Path('checkpoint').write_text('saved before kill')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, checkpoint if sys.argv[1] == 'grace' else signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "pathlib.Path('pids.json').write_text(json.dumps([os.getpid(), child.pid]))\n"
        "if sys.argv[1] == 'crash':\n"
        "    raise SystemExit(7)\n"
        "if sys.argv[1] != 'normal':\n"
        "    time.sleep(60)\n"
    )
    timeout = 0.8 if ending in ("timeout", "grace") else 10
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(uploaded),
            "--timeout",
            str(timeout),
            "--cleanup-timeout",
            "0.5",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            *(["--preserve-descendants-on-success"] if preserve else []),
            "--",
            sys.executable,
            "-I",
            str(worker),
            ending,
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        if ending == "cancel":
            deadline = time.monotonic() + 5
            while not (tmp_path / "pids.json").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "pids.json").exists()
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        receipt = json.loads((tmp_path / "cleanup.json").read_text())
        assert receipt["cleanup_confirmed"] is True
        assert receipt["error"] is None
        assert receipt["timed_out"] is (ending in ("timeout", "grace"))
        assert receipt["return_code"] == {"normal": 0, "crash": 7, "grace": 0}.get(ending, -signal.SIGKILL)
        if ending == "grace":
            assert (tmp_path / "checkpoint").read_text() == "saved before kill"
        for pid in json.loads((tmp_path / "pids.json").read_text()):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        # Clean up only this test's children if the regression fails.
        if (tmp_path / "pids.json").exists():
            for pid in json.loads((tmp_path / "pids.json").read_text()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc are required")
@pytest.mark.parametrize("detached", [False, True])
def test_borrowed_sandbox_preserves_service_for_verification(tmp_path: Path, detached: bool) -> None:
    # Use a parent subreaper to emulate Resources ownership without leaving orphans
    # on the test host. Both ordinary background jobs and setsid services must survive.
    driver = r"""
import ctypes, json, os, pathlib, signal, subprocess, sys, time, urllib.request
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
supervisor, root, detached = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == 'True'
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
worker.write_text("import pathlib, subprocess, sys, time\n"
                 "child = subprocess.Popen([sys.executable, 'service.py'], "
                 "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=" + str(detached) + ")\n"
                 "pathlib.Path('service.pid').write_text(str(child.pid))\n"
                 "deadline = time.monotonic() + 5\n"
                 "while not pathlib.Path('port').exists():\n"
                 "    assert child.poll() is None and time.monotonic() < deadline\n"
                 "    time.sleep(0.01)\n")
try:
    result = subprocess.run([sys.executable, '-I', supervisor, '--timeout', '10',
        '--cleanup-timeout', '1', '--preserve-descendants-on-success',
        '--receipt', str(root / 'cleanup.json'), '--', sys.executable, str(worker)],
        cwd=root, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((root / 'cleanup.json').read_text())
    assert receipt == {'cleanup_confirmed': True, 'return_code': 0, 'error': None, 'timed_out': False}
    port = int((root / 'port').read_text())
    with urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=2) as response:
        assert response.read() == b'task solved'
finally:
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
        raise AssertionError('Resources teardown left the task service alive')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", driver, str(SUPERVISOR), str(tmp_path), str(detached)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=25,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")
def test_failed_descendant_cleanup_cannot_write_success_receipt(tmp_path: Path) -> None:
    probe = """
import runpy, sys
supervisor = runpy.run_path(sys.argv[1])
def fail(timeout):
    raise TimeoutError('descendant still running')
supervisor['_supervise'].__globals__['_drain_children'] = fail
sys.argv = [sys.argv[1], '--timeout', '1', '--receipt', sys.argv[2], '--', sys.executable, '-c', 'pass']
raise SystemExit(supervisor['main']())
"""
    receipt_path = tmp_path / "cleanup.json"
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(SUPERVISOR), str(receipt_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["cleanup_confirmed"] is False
    assert receipt["error"] == "cleanup: descendant still running"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")
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
            "--timeout",
            "1",
            "--cleanup-timeout",
            "0.2",
            "--receipt",
            str(tmp_path / "cleanup.json"),
            "--",
            sys.executable,
            "-c",
            "import os, pathlib, time; pathlib.Path('worker.pid').write_text(str(os.getpid())); time.sleep(60)",
        ],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "worker.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "worker.pid").exists()
        worker_pid = int((tmp_path / "worker.pid").read_text())
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=10)
        receipt = json.loads((tmp_path / "cleanup.json").read_text())
        assert receipt["timed_out"] is True
        assert receipt["cleanup_confirmed"] is True
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if (tmp_path / "worker.pid").exists():
            try:
                os.kill(int((tmp_path / "worker.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")
def test_sigterm_during_spawn_does_not_lose_child_handle(tmp_path):
    # Deliver SIGTERM after the real child exists but before Popen returns to _supervise().
    # Isolate signal handlers/subreaper state from pytest, and always reap the test child.
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
    assert summary["child_alive"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")
def test_spawned_child_is_reaped_when_popen_loses_handle(tmp_path):
    # A constructor failure after process creation must still drain the subreaper's children.
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
    assert summary["return_code"] != 0
    assert summary["error"] == "launch handle lost after process creation"
    assert summary["cleanup_confirmed"] is True
    assert summary["child_alive"] is False
