# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from scripts.prepare_apex_retry_rows import (
    archive_world_startup_failures,
    failure_path_for,
    is_world_startup_failure,
    startup_archive_path_for,
)


STARTUP_ERROR = (
    "sandbox Stirrup rollout exited: TimeoutError: prebuilt world gateway did not become healthy: "
    "environment.log tail: <urlopen error [Errno 111] Connection refused>"
)


def _failure(task: str, rollout: int, attempt: int, error: str = STARTUP_ERROR, cls: str = "sandbox_error") -> dict:
    return {
        "task_id": task,
        "_ng_rollout_index": rollout,
        "_ng_attempt_index": attempt,
        "_ng_failure_class": cls,
        "apex_error": error,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def test_is_world_startup_failure_matches_only_startup_sandbox_errors() -> None:
    assert is_world_startup_failure(_failure("t", 0, 0))
    assert is_world_startup_failure(_failure("t", 0, 0, error="prebuilt world did not finish MCP startup"))
    assert not is_world_startup_failure(_failure("t", 0, 0, error="sandbox Stirrup rollout exited: command failed"))
    assert not is_world_startup_failure(_failure("t", 0, 0, cls="timeout_exceeded"))


def test_startup_failures_are_archived_and_stop_counting_as_attempts(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    crash = _failure("a", 0, 2, error="sandbox Stirrup rollout exited: command failed")
    _write(failure_path_for(output), [_failure("a", 0, 0), _failure("a", 0, 1), crash, _failure("b", 1, 0)])

    assert archive_world_startup_failures(output) == 3
    assert _read(failure_path_for(output)) == [crash]
    assert [(r["task_id"], r["_ng_attempt_index"]) for r in _read(startup_archive_path_for(output))] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
    ]


def test_archiving_is_capped_per_rollout_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    _write(failure_path_for(output), [_failure("a", 0, attempt) for attempt in range(3)])

    assert archive_world_startup_failures(output, max_archived=2) == 2
    assert [r["_ng_attempt_index"] for r in _read(failure_path_for(output))] == [2]
    assert archive_world_startup_failures(output, max_archived=2) == 0
    assert len(_read(startup_archive_path_for(output))) == 2


def test_repeated_startup_failures_reusing_an_attempt_index_still_reach_the_cap(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    # Gym reuses attempt index 0 every time the sidecar was emptied, so each resume writes "the same" attempt.
    for resume in range(5):
        _write(
            failure_path_for(output),
            _read(failure_path_for(output)) + [_failure("a", 0, 0, error=STARTUP_ERROR + str(resume))],
        )
        archive_world_startup_failures(output, max_archived=3)

    assert len(_read(startup_archive_path_for(output))) == 3
    # Past the cap the failures stay in the sidecar as normal attempts, so the rollout can finally run out of retries.
    assert [row["apex_error"][-1] for row in _read(failure_path_for(output))] == ["3", "4"]


def test_identical_row_left_in_both_files_by_an_interrupted_run_is_not_counted_twice(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    failure = _failure("a", 0, 0)
    _write(startup_archive_path_for(output), [failure])
    _write(failure_path_for(output), [failure])

    assert archive_world_startup_failures(output) == 0
    assert _read(failure_path_for(output)) == []
    assert _read(startup_archive_path_for(output)) == [failure]


def test_no_sidecar_is_a_no_op(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"

    assert archive_world_startup_failures(output) == 0
    assert not failure_path_for(output).exists()
    assert not startup_archive_path_for(output).exists()
