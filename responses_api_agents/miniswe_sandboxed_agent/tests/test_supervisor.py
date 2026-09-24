# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Linux process tests: detached grandchildren must exit before cleanup succeeds."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux child subreapers")
RUNNER = Path(__file__).resolve().parents[1] / "sandbox_runner.py"


@pytest.mark.parametrize("mode", ["normal", "cancel", "stop_before_launch"])
def test_cleanup_owns_double_forked_children(tmp_path, mode):
    worker = tmp_path / "worker.py"
    worker.write_text("""
import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
if os.fork() == 0:
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / 'child.pid').write_text(str(os.getpid()))
    time.sleep(120)
    os._exit(0)
while not (root / 'child.pid').exists():
    time.sleep(0.01)
if (root / 'mode').read_text() == 'cancel':
    time.sleep(120)
""")
    (tmp_path / "mode").write_text(mode)
    if mode == "stop_before_launch":
        (tmp_path / "stop").touch()
    command = (
        "import importlib.util; from pathlib import Path; "
        f"spec=importlib.util.spec_from_file_location('runner', {str(RUNNER)!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        f"module.__file__={str(worker)!r}; module.supervise(Path({str(tmp_path)!r}))"
    )
    process = subprocess.Popen([sys.executable, "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child = None
    try:
        if mode != "stop_before_launch":
            deadline = time.monotonic() + 20
            while not (tmp_path / "child.pid").exists():
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "worker did not start"
                time.sleep(0.05)
            child = int((tmp_path / "child.pid").read_text())
            if mode == "cancel":
                (tmp_path / "stop").touch()
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, (stdout, stderr)
        evidence = json.loads((tmp_path / "cleanup.json").read_text())
        assert evidence["status"] == "stopped"
        assert evidence["remaining_pids"] == []
        if child is not None:
            assert child in evidence["terminated_pids"]
            assert not Path(f"/proc/{child}").exists()
        else:
            assert evidence["worker_pid"] is None
            assert not (tmp_path / "child.pid").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
