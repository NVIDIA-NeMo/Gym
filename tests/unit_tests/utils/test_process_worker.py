# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.utils import process_worker


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux child subreaper")


def _wait(path: Path, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.02)


def _worker(tmp_path: Path, script: str, *options: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            process_worker.__file__,
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--grace-seconds",
            "0.2",
            *options,
            "--",
            sys.executable,
            "-c",
            script,
        ],
        cwd=tmp_path,
    )


@pytest.mark.parametrize("mode", ["complete", "failed", "cancelled"])
def test_worker_reaps_detached_descendants_before_finishing(tmp_path, mode):
    sleeper = "import time; time.sleep(60)"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable, '-c', {sleeper!r}], start_new_session=True); "
        "Path('child').write_text(str(p.pid)); "
        + ("time.sleep(60)" if mode == "cancelled" else f"raise SystemExit({7 if mode == 'failed' else 0})")
    )
    process = _worker(tmp_path, script)
    pid = None
    try:
        _wait(tmp_path / "child", process)
        pid = int((tmp_path / "child").read_text())
        if mode == "cancelled":
            process.terminate()
        assert process.wait(timeout=10) == {"complete": 0, "failed": 7, "cancelled": 143}[mode]
        assert not Path(f"/proc/{pid}").exists()
        receipt = json.loads((tmp_path / "receipt.json").read_text())
        assert receipt["state"] == mode
        assert receipt["cleanup_confirmed"] is True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigterm_resistant_descendant_is_killed_and_reaped(tmp_path):
    sleeper = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); Path('ready').touch(); time.sleep(60)"
    )
    script = f"""import subprocess,sys,time
from pathlib import Path
p = subprocess.Popen([sys.executable, '-c', {sleeper!r}], start_new_session=True)
Path('child').write_text(str(p.pid))
while not Path('ready').exists(): time.sleep(.01)
"""
    process = _worker(tmp_path, script)
    assert process.wait(timeout=10) == 0
    assert not Path(f"/proc/{int((tmp_path / 'child').read_text())}").exists()
    assert json.loads((tmp_path / "receipt.json").read_text())["cleanup_confirmed"]


def test_external_cleanup_finishes_before_worker_and_failure_is_not_success(tmp_path):
    process = _worker(tmp_path, "raise SystemExit(0)", "--cleanup-command", "touch cleaned; exit 3")
    assert process.wait(timeout=10) == 70
    assert (tmp_path / "cleaned").exists()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["state"] == "cleanup_failed"
    assert not receipt["cleanup_confirmed"]


def test_worker_preserves_recovery_environment_and_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GYM_RESUBMIT", "0")
    monkeypatch.setenv("GYM_KEEP_BUDGET", "1")
    script = """import os,json
from pathlib import Path
with Path('calls').open('a') as f: f.write('called\n')
Path('env.json').write_text(json.dumps({k: os.environ[k] for k in ('GYM_RESUBMIT','GYM_KEEP_BUDGET')}))
raise SystemExit(7)
""".replace("'called\n'", "'called\\n'")
    process = _worker(tmp_path, script)
    assert process.wait(timeout=10) == 7
    assert (tmp_path / "calls").read_text() == "called\n"
    assert json.loads((tmp_path / "env.json").read_text()) == {"GYM_RESUBMIT": "0", "GYM_KEEP_BUDGET": "1"}
    repeated = _worker(tmp_path, "from pathlib import Path; Path('ran-twice').touch()")
    assert repeated.wait(timeout=10) != 0
    assert not (tmp_path / "ran-twice").exists()
