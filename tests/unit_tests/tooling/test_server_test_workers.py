# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise concurrent CI workers, shard ownership, failure reporting, and cancellation."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import pytest

from nemo_gym.cli.env import _select_shard


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scripts/ci/server_test_workers.py"
WORKER_ENV = ROOT / "scripts/ci/test_worker_env.sh"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CHILD = """import json, os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
marker = Path(sys.argv[1])
pending = marker.with_name('.' + marker.name)
pending.write_text(json.dumps({'pid': os.getpid(), 'group': os.getpgrp()}))
pending.replace(marker)
print('ready', flush=True)
deadline = time.monotonic() + 60
while time.monotonic() < deadline and not (marker.parent / 'stop-children').exists():
    time.sleep(0.02)
"""
WORKER = """#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path
state = Path(os.environ['WORKER_STATE'])
index = int(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('+shard_index=')))
(state / f'started-{index}').write_text(json.dumps({'pid': os.getpid(), 'args': sys.argv[1:],
    'threads': {key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}))
deadline = time.monotonic() + 15
while len(list(state.glob('started-*'))) != 4:
    if time.monotonic() > deadline:
        raise RuntimeError('Four workers did not overlap')
    time.sleep(0.02)
if os.environ.get('SPAWN_CHILD'):
    child = subprocess.Popen([sys.executable, '-c', os.environ['CHILD_CODE'], str(state / f'child-{index}')],
        stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline() == 'ready\\n'
    if os.environ.get('EXIT_ONE_LEADER') and index == 10:
        sys.exit(0)
if os.environ.get('HOLD_WORKERS'):
    time.sleep(60)
(state / f'finished-{index}').touch()
sys.stdout.buffer.write(f'finished worker {index}\\n'.encode() + b'\\xff\\n')
sys.exit(int(os.environ['FAIL_STATUS']) if index == 10 else 0)
"""


@pytest.fixture
def worker_environment(tmp_path: Path) -> dict[str, str]:
    """Provide four real subprocesses with a startup barrier and separate evidence files."""
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "ng_test_all"
    executable.write_text(WORKER)
    executable.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    return dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        WORKER_STATE=str(state),
        FAIL_STATUS="0",
        CHILD_CODE=CHILD,
        **dict.fromkeys(THREAD_VARIABLES, "64"),
    )


def worker_command() -> list[str]:
    """Run the actual CI environment and launcher with paths containing spaces."""
    return [
        "bash",
        "-c",
        'source "$1"; shift; exec "$@"',
        "workers",
        str(WORKER_ENV),
        sys.executable,
        str(LAUNCHER),
        "2",
        "8",
        "+uv_cache_dir=/cache with spaces",
    ]


@pytest.mark.parametrize("status", [0, 1, 5])
def test_workers_overlap_preserve_shard_ownership_and_failures(
    tmp_path: Path, worker_environment: dict[str, str], status: int
) -> None:
    """Every selected component belongs to exactly one concurrent worker, even after a failure."""
    worker_environment["FAIL_STATUS"] = str(status)
    result = subprocess.run(
        worker_command(), cwd=tmp_path, env=worker_environment, capture_output=True, text=True, timeout=25
    )
    assert result.returncode == bool(status), result.stdout + result.stderr
    state = Path(worker_environment["WORKER_STATE"])
    assert len(list(state.glob("finished-*"))) == 4
    paths = [Path(f"resources_servers/server-{index:03}") for index in range(103)]
    selected = []
    for path in state.glob("started-*"):
        record = json.loads(path.read_text())
        args = record["args"]
        assert "+uv_cache_dir=/cache with spaces" in args
        assert all(value == "1" for value in record["threads"].values())
        index = int(next(arg.split("=", 1)[1] for arg in args if arg.startswith("+shard_index=")))
        count = int(next(arg.split("=", 1)[1] for arg in args if arg.startswith("+num_shards=")))
        selected.extend(_select_shard(paths, index, count))
        assert f"finished worker {index}" in result.stdout
    assert sorted(selected) == _select_shard(paths, 2, 8)
    logs = list((tmp_path / "results/server-workers/shard-2").glob("worker-*.log"))
    assert len(logs) == 4
    assert all(log.read_bytes().endswith(b"\xff\n") for log in logs)
    assert "\ufffd" in result.stdout


def test_cancellation_terminates_every_worker(tmp_path: Path, worker_environment: dict[str, str]) -> None:
    """A cancelled launcher must reap its worker process groups before exiting."""
    worker_environment["HOLD_WORKERS"] = "1"
    process = subprocess.Popen(worker_command(), cwd=tmp_path, env=worker_environment, stdout=subprocess.DEVNULL)
    try:
        state = Path(worker_environment["WORKER_STATE"])
        deadline = monotonic() + 15
        while len(list(state.glob("started-*"))) != 4 and monotonic() < deadline:
            sleep(0.02)
        assert len(list(state.glob("started-*"))) == 4
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 128 + signal.SIGTERM
        for path in state.glob("started-*"):
            pid = json.loads(path.read_text())["pid"]
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


def process_is_live(pid: int) -> bool:
    """Treat orphaned zombies as dead without depending on the container init's reaping policy."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="The CI launcher uses Linux process-group ownership")
@pytest.mark.parametrize("leader_exits", [False, True])
def test_cancellation_kills_term_ignoring_children_after_leader_exit(
    tmp_path: Path, worker_environment: dict[str, str], leader_exits: bool
) -> None:
    """A single grace deadline covers every group, including one whose leader has already exited."""
    worker_environment.update(HOLD_WORKERS="1", SPAWN_CHILD="1")
    if leader_exits:
        worker_environment["EXIT_ONE_LEADER"] = "1"
    process = subprocess.Popen(worker_command(), cwd=tmp_path, env=worker_environment, stdout=subprocess.DEVNULL)
    state = Path(worker_environment["WORKER_STATE"])
    try:
        deadline = monotonic() + 15
        while len(list(state.glob("child-*"))) != 4 and monotonic() < deadline:
            sleep(0.02)
        assert len(list(state.glob("child-*"))) == 4
        children = [json.loads(path.read_text()) for path in state.glob("child-*")]
        workers = [json.loads(path.read_text())["pid"] for path in state.glob("started-*")]
        assert len(workers) == 4
        assert {child["group"] for child in children} == set(workers)
        if leader_exits:
            leader = json.loads((state / "started-10").read_text())["pid"]
            while process_is_live(leader) and monotonic() < deadline:
                sleep(0.02)
            assert not process_is_live(leader)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 128 + signal.SIGTERM
        deadline = monotonic() + 2
        while any(process_is_live(child["pid"]) for child in children) and monotonic() < deadline:
            sleep(0.02)
        assert not any(process_is_live(child["pid"]) for child in children)
        assert not any(process_is_live(pid) for pid in workers)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        (state / "stop-children").touch()
