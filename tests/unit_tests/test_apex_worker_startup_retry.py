# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import subprocess
from pathlib import Path

import pytest


SBATCH = Path(__file__).resolve().parents[2] / "scripts" / "run_apex_agents_k3_unified.sbatch"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def _retry_function() -> str:
    match = re.search(r"# BEGIN worker startup retry\n(.*?)# END worker startup retry", SBATCH.read_text(), re.DOTALL)
    assert match, "startup retry block markers are missing from the launcher"
    return match.group(1)


def _run(tmp_path: Path, fail_first: int, **env: str) -> tuple[int, int, str]:
    """Run the retry function against a fake `gym` that fails its first `fail_first` calls."""
    count = tmp_path / "calls"
    gym = tmp_path / "gym"
    gym.write_text(
        "#!/bin/bash\n"
        f'n=$(cat "{count}" 2>/dev/null || echo 0)\n'
        f'echo $((n + 1)) > "{count}"\n'
        f'[ "$n" -lt {fail_first} ] && exit 7\n'
        "exit 0\n"
    )
    gym.chmod(0o755)
    script = (
        "set -Eeuo pipefail\n"
        'log() { echo "$*"; }\n'
        'setsid() { "$@"; }\n'  # macOS has no setsid; the launcher runs on Linux
        "SHARD_INDEX=3\n"
        f"{_retry_function()}\n"
        f'rc=0; run_eval_with_startup_retry "{gym}" eval run || rc=$?\n'
        'echo "rc=${rc}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "WORKER_START_RETRY_DELAY_SECONDS": "0", **env},
        check=True,
    )
    calls = int(count.read_text()) if count.exists() else 0
    return int(re.search(r"^rc=(\d+)$", result.stdout, re.MULTILINE).group(1)), calls, result.stdout


def test_a_clean_run_is_not_retried(tmp_path: Path) -> None:
    rc, calls, _ = _run(tmp_path, fail_first=0)

    assert (rc, calls) == (0, 1)


def test_early_startup_failures_are_retried_until_one_succeeds(tmp_path: Path) -> None:
    rc, calls, output = _run(tmp_path, fail_first=2)

    assert (rc, calls) == (0, 3)
    assert output.count("startup failure); retrying") == 2
    assert "Worker 3: gym eval exited rc=7" in output


def test_a_worker_that_never_starts_gives_up_with_its_exit_code(tmp_path: Path) -> None:
    rc, calls, _ = _run(tmp_path, fail_first=99, WORKER_START_ATTEMPTS="3")

    assert (rc, calls) == (7, 3)


def test_a_failure_after_the_startup_window_is_returned_unchanged(tmp_path: Path) -> None:
    rc, calls, output = _run(tmp_path, fail_first=99, WORKER_STARTUP_WINDOW_SECONDS="0")

    assert (rc, calls) == (7, 1)
    assert "retrying" not in output


@pytest.mark.skipif(shutil.which("perl") is None, reason="perl is needed to start a new process group")
@pytest.mark.parametrize("ignore_term", [False, True])
def test_a_retry_waits_until_the_failed_attempts_servers_are_gone(tmp_path: Path, ignore_term: bool) -> None:
    """A server left over from the failed attempt must be stopped (or force-killed) before the retry starts."""
    child_pid = tmp_path / "child_pid"
    ready = tmp_path / "child_ready"
    seen = tmp_path / "child_alive_at_retry"
    trap = "trap '' TERM; " if ignore_term else ""
    gym = tmp_path / "gym"
    gym.write_text(
        "#!/bin/bash\n"
        f'if [ ! -e "{child_pid}" ]; then\n'
        f'  bash -c "{trap}touch {ready}; sleep 30" &\n'
        f'  echo $! > "{child_pid}"\n'
        f'  while [ ! -e "{ready}" ]; do sleep 0.05; done\n'
        "  exit 7\n"
        "fi\n"
        f'kill -0 "$(cat "{child_pid}")" 2>/dev/null && echo yes > "{seen}" || echo no > "{seen}"\n'
        "exit 0\n"
    )
    gym.chmod(0o755)
    script = (
        "set -Eeuo pipefail\n"
        'log() { echo "$*"; }\n'
        # A real new process group whose id is the background pid, as with util-linux setsid.
        "setsid() { exec perl -e 'setpgrp(0, 0); exec @ARGV or die' \"$@\"; }\n"
        "SHARD_INDEX=3\n"
        f"{_retry_function()}\n"
        f'rc=0; run_eval_with_startup_retry "{gym}" eval run || rc=$?\n'
        'echo "rc=${rc}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "WORKER_START_RETRY_DELAY_SECONDS": "0", "WORKER_STOP_GRACE_SECONDS": "1"},
        check=True,
        timeout=30,
    )

    assert "rc=0" in result.stdout
    assert seen.read_text().strip() == "no"
    assert ("force-killing" in result.stdout) == ignore_term
