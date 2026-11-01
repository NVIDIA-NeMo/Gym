# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A watched login-host reader reports progress on stderr while it runs, and the caller forwards it live.

The reader is the run's pinned Gym install, so the progress comes from the caller's wrapper, which reads the bytes
the reader's session has read from /proc. Stdout and its ``__GYM_READER_*`` protocol are unchanged.
"""

import re
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

import nemo_gym.orchestration.executors.connection as connection
import nemo_gym.orchestration.executors.slurm as slurm
from nemo_gym.orchestration.executors.connection import LocalConnection


PROGRESS = re.compile(r"^gym-reader progress: pass=(\d+) bytes=(\d+) elapsed_s=(\d+)$")
needs_reader_tools = pytest.mark.skipif(
    shutil.which("setsid") is None or shutil.which("timeout") is None or not Path("/proc/self/io").exists(),
    reason="needs setsid, coreutils timeout and /proc/<pid>/io",
)


class RecordingStderr:
    """Stands in for sys.stderr: every write, with the time it arrived."""

    def __init__(self) -> None:
        self.writes: list[tuple[float, str]] = []

    def write(self, text: str) -> int:
        self.writes.append((time.monotonic(), text))
        return len(text)

    def flush(self) -> None:
        pass


def run_held(command: list[str], monkeypatch) -> tuple[str, float, RecordingStderr]:
    """Run ``command`` over a held input on a thread; return its stdout, when it ended, and what reached stderr."""
    stderr = RecordingStderr()
    monkeypatch.setattr(sys, "stderr", stderr)
    result: dict[str, object] = {}

    def run() -> None:
        result["output"] = LocalConnection().run_with_input_held(command)
        result["ended"] = time.monotonic()

    runner = threading.Thread(target=run)
    runner.start()
    runner.join(timeout=60)
    assert not runner.is_alive(), "the command did not end"
    return result["output"], result["ended"], stderr


def test_progress_lines_reach_the_callers_stderr_while_the_command_runs(monkeypatch) -> None:
    command = [
        "echo 'gym-reader progress: pass=1 bytes=7 elapsed_s=0' >&2",
        "echo 'unrelated diagnostics' >&2",
        "sleep 2",
        "echo out",
        "exit 0",
    ]

    output, ended, stderr = run_held(command, monkeypatch)

    assert output == "out\n"  # stdout is returned whole, as before
    assert [text for _, text in stderr.writes] == ["gym-reader progress: pass=1 bytes=7 elapsed_s=0\n"]
    assert stderr.writes[0][0] < ended - 1  # it arrived while the command was still running


def test_a_failed_command_still_reports_all_of_its_stderr(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stderr", RecordingStderr())
    command = ["echo 'gym-reader progress: pass=1 bytes=0 elapsed_s=0' >&2", "echo boom >&2", "exit 3"]
    with pytest.raises(RuntimeError, match="exit code 3") as raised:
        LocalConnection().run_with_input_held(command)
    assert "boom" in str(raised.value) and connection.READER_PROGRESS_PREFIX in str(raised.value)


@needs_reader_tools
def test_the_watched_reader_reports_its_bytes_before_it_ends_and_stdout_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rollouts = artifacts / "rollouts.jsonl"
    rollouts.write_bytes(b"x" * 40_000)
    # A reader that reads the rollouts slowly, then prints its projection line, as the Store reader does.
    reader = (
        f'{sys.executable} -c \'import sys, time; f = open(sys.argv[1], "rb"); '
        f'[(f.read(4096), time.sleep(0.4)) for _ in range(10)]; print("__GYM_MANAGED:{{}}")\' {rollouts}'
    )
    command = slurm._reader_command(
        reader, tmp_path, "{}", bound_s=300, factor=1, token="progress-token", watchdog=True, progress_interval_s=1
    )

    output, ended, stderr = run_held(command, monkeypatch)

    progress = [(at, PROGRESS.match(text.strip())) for at, text in stderr.writes]
    assert progress and all(match for _, match in progress), stderr.writes
    assert progress[0][0] < ended - 1  # the first line arrived well before the reader ended
    assert max(int(match.group(2)) for _, match in progress) >= 4096  # it counts the reader's reads
    # The pass is the bytes read over the rollout bytes; here interpreter start-up reads dominate a tiny file.
    assert all(int(match.group(1)) == int(match.group(2)) // 40_000 + 1 for _, match in progress)
    # Stdout carries the reader protocol exactly as before.
    assert "__GYM_MANAGED:{}" in output.splitlines() and "__GYM_READER_RC:0" in output
    assert not any(line.startswith("gym-reader progress") for line in output.splitlines())
    assert slurm._reader_failure(output, 0.0) == (None, False)


@needs_reader_tools
def test_the_progress_loop_ends_with_the_reader(tmp_path: Path, monkeypatch) -> None:
    # A quick reader with a long interval: the shell must not wait for the loop's sleep before it ends.
    command = slurm._reader_command(
        "true", tmp_path, "{}", bound_s=300, factor=1, token="quick-token", watchdog=True, progress_interval_s=60
    )
    started = time.monotonic()
    output, ended, stderr = run_held(command, monkeypatch)
    assert ended - started < 20 and "__GYM_READER_RC:0" in output and stderr.writes == []
