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


SUPERVISOR = Path(__file__).parents[2] / "nemo_gym/sandbox/process_supervisor.py"


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
    "ending", ["normal", "crash", "timeout", "cancel", "grace", "normal-no-deadline", "cancel-no-deadline"]
)
def test_supervisor_reaps_detached_tools_and_preserves_term_grace(tmp_path: Path, ending: str) -> None:
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
    deadline_args = [] if ending.endswith("-no-deadline") else ["--timeout", str(timeout)]
    ending = ending.removesuffix("-no-deadline")
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(uploaded),
            *deadline_args,
            "--cleanup-timeout",
            "0.5",
            "--receipt",
            str(tmp_path / "cleanup.json"),
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
        assert receipt["timed_out"] is (ending in ("timeout", "cancel", "grace"))
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
@pytest.mark.parametrize("ending", ["normal", "crash", "timeout", "cancel"])
def test_deferred_cleanup_retains_services_only_on_success(tmp_path: Path, ending: str) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "pathlib.Path('service.tmp').write_text(str(child.pid))\n"
        "pathlib.Path('service.tmp').replace('service.pid')\n"
        "if sys.argv[1] == 'crash': raise SystemExit(7)\n"
        "if sys.argv[1] in ('timeout', 'cancel'): time.sleep(60)\n"
    )
    completion, stop, cleanup = (tmp_path / name for name in ("completion.json", "stop", "cleanup.json"))
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(SUPERVISOR),
            "--timeout",
            "0.8" if ending == "timeout" else "10",
            "--cleanup-timeout",
            "0.2",
            "--stop-file",
            str(stop),
            "--completion-receipt",
            str(completion),
            "--receipt",
            str(cleanup),
            "--",
            sys.executable,
            str(worker),
            ending,
        ],
        cwd=tmp_path,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "service.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        service = int((tmp_path / "service.pid").read_text())
        if ending == "cancel":
            process.send_signal(signal.SIGTERM)
        while not completion.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert completion.exists()
        if ending == "normal":
            assert json.loads(completion.read_text()) == {"return_code": 0, "timed_out": False}
            assert process.poll() is None
            assert not cleanup.exists()
            os.kill(service, 0)  # Verification can still reach a live detached service.
            stop.touch()
        process.wait(timeout=5)
        assert json.loads(cleanup.read_text())["cleanup_confirmed"] is True
        with pytest.raises(ProcessLookupError):
            os.kill(service, 0)
    finally:
        stop.touch()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
