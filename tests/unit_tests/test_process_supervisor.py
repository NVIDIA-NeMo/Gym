# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the uploaded bash supervisor as the sandbox runs it, without Gym imports."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.agent_utils.supervisor_client import SUPERVISOR_SCRIPT


SUPERVISOR = str(SUPERVISOR_SCRIPT)
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")


def _supervise(tmp_path: Path, *command: str, timeout: str = "10", **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            SUPERVISOR,
            "--timeout",
            timeout,
            "--cleanup-timeout",
            "0.5",
            "--receipt",
            str(tmp_path / "cleanup.json"),
        ]
        + ["--stop-file", str(tmp_path / "stop.request"), "--", *command],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=20,
        **kwargs,
    )


def _receipt(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "cleanup.json").read_text())


@pytest.mark.parametrize("timeout", ["0", ".", "0.000", "-1", "nan", "inf", "1e3", ""])
def test_supervisor_rejects_invalid_deadlines(tmp_path: Path, timeout: str) -> None:
    result = _supervise(tmp_path, "true", timeout=timeout)
    assert result.returncode == 2
    assert "finite and positive" in result.stderr
    assert not (tmp_path / "cleanup.json").exists()


def test_supervisor_requires_a_command_before_writing_a_receipt(tmp_path: Path) -> None:
    result = _supervise(tmp_path)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert not (tmp_path / "cleanup.json").exists()


@linux_only
@pytest.mark.parametrize("command, return_code", [("exit 0", 0), ("exit 7", 7), ("kill -KILL $$", 137)])
def test_natural_exit_reports_the_harness_status(tmp_path: Path, command: str, return_code: int) -> None:
    result = _supervise(tmp_path, "bash", "-c", command)
    assert result.returncode == 0, result.stderr
    assert _receipt(tmp_path) == {
        "cleanup_confirmed": True,
        "return_code": return_code,
        "error": None,
        "timed_out": False,
    }


@linux_only
def test_runs_with_only_bash_and_core_utilities_on_path(tmp_path: Path) -> None:
    # Task images need not ship Python: the supervisor itself uses only these programs.
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("bash", "sleep", "mv", "sh"):
        (tools / name).symlink_to(shutil.which(name))
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        [str(tools / "bash"), SUPERVISOR, "--timeout", "0.3", "--cleanup-timeout", "0.3"]
        + ["--receipt", str(work / "cleanup.json"), "--stop-file", str(work / "stop"), "--", "sh", "-c", "sleep 30"],
        capture_output=True,
        text=True,
        env={"PATH": str(tools)},
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    receipt = _receipt(work)
    assert receipt["cleanup_confirmed"] is True and receipt["timed_out"] is True, receipt


@linux_only
def test_existing_stop_file_fences_the_launch(tmp_path: Path) -> None:
    stop = tmp_path / "stop.request"
    stop.touch()
    result = subprocess.run(
        ["bash", SUPERVISOR, "--timeout", "10", "--cleanup-timeout", "1", "--receipt", str(tmp_path / "cleanup.json")]
        + ["--stop-file", str(stop), "--", "touch", str(tmp_path / "started")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert _receipt(tmp_path) == {"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": False}
    assert not (tmp_path / "started").exists()


@linux_only
def test_stop_file_stops_the_harness_with_term_first(tmp_path: Path) -> None:
    # The controller requests a stop with this file rather than a signal.
    stop = tmp_path / "stop.request"
    worker = "trap 'echo saved > checkpoint; exit 0' TERM; touch ready; while :; do sleep 0.05; done"
    process = subprocess.Popen(
        ["bash", SUPERVISOR, "--timeout", "30", "--cleanup-timeout", "2", "--receipt", str(tmp_path / "cleanup.json")]
        + ["--stop-file", str(stop), "--", "bash", "-c", worker],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        stop.touch()
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        receipt = _receipt(tmp_path)
        assert receipt == {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
        assert (tmp_path / "checkpoint").read_text() == "saved\n"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@linux_only
@pytest.mark.parametrize("ending", ["normal", "crash", "timeout", "cancel", "grace", "stubborn", "killed"])
@pytest.mark.parametrize("detached", [False, True])
def test_only_detached_services_survive_harness_group_termination(tmp_path: Path, ending: str, detached: bool) -> None:
    # The isolated parent emulates the sandbox owner and reaps every test process.
    # A completed tool leaves a service orphaned before interruption. Only
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
                 "if ending == 'crash': raise SystemExit(7)\n"
                 "if ending == 'killed': os.kill(os.getpid(), signal.SIGKILL)\n"
                 "if ending != 'normal': time.sleep(60)\n")
timed_out = ending in ('timeout', 'grace', 'stubborn')
process = None
try:
    process = subprocess.Popen(['bash', supervisor, '--timeout', '1.5' if timed_out else '10',
        '--cleanup-timeout', '0.5', '--receipt', str(root / 'cleanup.json'), '--stop-file', str(root / 'stop'), '--',
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
    expected = {'crash': 7, 'stubborn': 137, 'killed': 137}.get(ending, 0)
    assert receipt['return_code'] == expected, receipt
    assert receipt['error'] is None, receipt
    if ending == 'grace':
        assert (root / 'checkpoint').read_text() == 'saved before exit'
    try:
        os.killpg(int((root / 'worker.pid').read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        stats = [p.read_text() for p in pathlib.Path('/proc').glob('[0-9]*/stat')]
        group = int((root / 'worker.pid').read_text())
        live = [s for s in stats if int(s.rsplit(')', 1)[1].split()[2]) == group and s.rsplit(')', 1)[1].split()[0] not in 'ZX']
        assert not live, ('supervised process group still running', live)
    port = int((root / 'port').read_text())
    if detached:
        with urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=2) as response:
            assert response.read() == b'task solved'
    else:
        stat = pathlib.Path('/proc/' + (root / 'service.pid').read_text() + '/stat')
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] in 'ZX'
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
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", driver, SUPERVISOR, str(tmp_path), ending, str(detached)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=40,
    )
    assert result.returncode == 0, result.stderr


@linux_only
@pytest.mark.parametrize("cooperative", [True, False])
@pytest.mark.parametrize("ending", ["timeout", "cancel"])
def test_foreground_tool_cannot_mutate_task_after_confirmed_receipt(
    tmp_path: Path, cooperative: bool, ending: str
) -> None:
    # A separate owner reaps any escaped process even when this regression fails.
    driver = r"""
import ctypes, json, os, pathlib, signal, subprocess, sys, time
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
supervisor, root, cooperative, ending = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == 'True', sys.argv[4]
writer = "import pathlib,time\nwhile not pathlib.Path('release').exists(): time.sleep(.01)\npathlib.Path('mutated').touch()"
# The writer is orphaned when the tool dies, but stays in the harness group.
tool = ("import pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c'," + repr(writer) + "])\n"
        "pathlib.Path('writer.pid').write_text(str(child.pid))\nchild.wait()\n")
cli = ("import pathlib,signal,subprocess,sys\n"
       + ("" if cooperative else "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n")
       + "child=subprocess.Popen([sys.executable,'-c'," + repr(tool) + "])\n"
       + "pathlib.Path('tool.pid').write_text(str(child.pid))\nchild.wait()\n")
process = None
try:
    process = subprocess.Popen(['bash', supervisor, '--timeout', '3' if ending == 'timeout' else '10',
        '--cleanup-timeout', '.3', '--receipt', str(root / 'cleanup.json'), '--stop-file', str(root / 'stop'),
        '--', sys.executable, '-c', cli],
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
    (root / 'release').touch()
    time.sleep(.2)
    assert not (root / 'mutated').exists(), 'foreground tool wrote after termination receipt'
    for name in ('tool.pid', 'writer.pid'):
        stat = pathlib.Path('/proc/' + (root / name).read_text() + '/stat')
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] in 'ZX', name + ' still runs'
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
        [sys.executable, "-I", "-c", driver, SUPERVISOR, str(tmp_path), str(cooperative), ending],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
