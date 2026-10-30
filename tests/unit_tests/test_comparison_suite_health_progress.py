# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify visible health progress without waiting for the production logging interval."""

import logging
import os
from pathlib import Path
from threading import get_ident

import orjson
import pytest

import nemo_gym.comparison.suite.health as health
import nemo_gym.rollout_health as rollout_health
from nemo_gym.path_utils import failures_path_for, materialized_path_for
from nemo_gym.rollout_health import run_health_checks


def _write_rollouts(path: Path, count: int) -> None:
    path.write_bytes(
        b"".join(
            orjson.dumps(
                {
                    "_ng_task_index": 0,
                    "_ng_rollout_index": repeat,
                    "response": {"output": [{"type": "message", "role": "assistant", "content": "ok"}]},
                }
            )
            + b"\n"
            for repeat in range(count)
        )
    )


def test_file_health_logs_start_and_completed_counts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "rollouts.jsonl"
    _write_rollouts(path, 2)

    with caplog.at_level(logging.INFO, logger=health.__name__):
        summary = health.check_rollout_file_health(path, workers=1)

    assert summary.record_count == 2
    assert summary.unobserved == 2
    records = [record for record in caplog.records if record.name == health.__name__]
    messages = [record.getMessage() for record in records]
    assert messages[0] == f"Checking rollout file health: {path}"
    assert any(f"Rollout file health progress: {path}: checking records (0/2 records;" in note for note in messages)
    assert messages[-1].startswith(f"Completed rollout file health: {path} (")
    assert "seconds elapsed): 0 healthy, 0 unhealthy, 2 unobserved of 2 records" in messages[-1]


def test_file_health_logs_unavailable_source(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "missing.jsonl"

    with caplog.at_level(logging.INFO, logger=health.__name__):
        summary = health.check_rollout_file_health(path, workers=1)

    assert summary.unavailable_reason == f"Rollout JSONL not found: {path}"
    records = [record for record in caplog.records if record.name == health.__name__]
    messages = [record.getMessage() for record in records]
    assert messages[0] == f"Checking rollout file health: {path}"
    assert messages[-1].startswith(f"Rollout file health unavailable: {path} (")
    assert summary.unavailable_reason in messages[-1]
    assert records[-1].levelno == logging.WARNING
    assert not any(message.startswith("Completed rollout file health:") for message in messages)


def test_health_progress_logs_phase_changes_and_throttles_record_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "long-running.jsonl"
    # Control only the clock boundary: phase/count handling and logging remain real.
    times = iter((0.0, 29.0, 30.0, 31.0, 60.0, 61.0))
    monkeypatch.setattr(health, "monotonic", lambda: next(times))
    progress = health._log_health_progress(path, started=0.0)

    with caplog.at_level(logging.INFO, logger=health.__name__):
        progress("checking records", 0, 100)
        progress("checking records", 10, 100)
        progress("checking records", 20, 100)
        progress("reading input inventory", 0, None)
        progress("reading input inventory", 10, None)
        progress("reading input inventory", 20, None)

    messages = [record.getMessage() for record in caplog.records if record.name == health.__name__]
    prefix = f"Rollout file health progress: {path}: "
    assert messages == [
        prefix + "checking records (0/100 records; 0 seconds elapsed)",
        prefix + "checking records (20/100 records; 30 seconds elapsed)",
        prefix + "reading input inventory (0 rows; 31 seconds elapsed)",
        prefix + "reading input inventory (20 rows; 61 seconds elapsed)",
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_health_pipeline_reports_progress_in_caller_without_changing_results(tmp_path: Path, workers: int) -> None:
    path = tmp_path / "rollouts.jsonl"
    _write_rollouts(path, 3)
    materialized_path_for(path).write_bytes(
        b"".join(orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": repeat}) + b"\n" for repeat in range(4))
    )
    failures_path_for(path).write_bytes(orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 3}) + b"\n")
    directory = tmp_path / "reports"
    without_progress = run_health_checks(path, output_dir=directory, workers=workers)
    expected_reports = (without_progress.summary_path.read_bytes(), without_progress.verdicts_path.read_bytes())
    updates: list[tuple[str, int, int | None]] = []
    callers: set[tuple[int, int]] = set()

    def record_progress(phase: str, completed: int, total: int | None) -> None:
        updates.append((phase, completed, total))
        callers.add((os.getpid(), get_ident()))

    with_progress = run_health_checks(path, output_dir=directory, workers=workers, progress=record_progress)

    assert with_progress.summary == without_progress.summary
    assert with_progress.rollouts == without_progress.rollouts
    assert (with_progress.summary_path.read_bytes(), with_progress.verdicts_path.read_bytes()) == expected_reports
    assert callers == {(os.getpid(), get_ident())}
    assert [(completed, total) for phase, completed, total in updates if phase == "checking records"] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert {phase for phase, _, _ in updates} >= {
        "indexing rollout files",
        "reading input inventory",
        "reading failure records",
        "reading failure timings",
        "writing health reports",
    }
    assert [(completed, total) for phase, completed, total in updates if phase == "indexing rollout files"][-1] == (
        3,
        None,
    )
    assert [(completed, total) for phase, completed, total in updates if phase == "reading input inventory"][-1] == (
        4,
        None,
    )
    assert updates[-1][:3] == ("writing health reports", 3, 3)


def test_health_progress_preserves_serial_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "rollouts.jsonl"
    _write_rollouts(path, 2)
    expected = run_health_checks(path, output_dir=tmp_path / "reports", workers=1)
    updates: list[tuple[str, int, int | None]] = []

    def unavailable_pool(*, max_workers: int) -> None:
        raise OSError("Process creation unavailable")

    def record_progress(phase: str, completed: int, total: int | None) -> None:
        updates.append((phase, completed, total))

    # The process-creation boundary fails; real file parsing and health calculations still run.
    monkeypatch.setattr(rollout_health, "ProcessPoolExecutor", unavailable_pool)
    with pytest.warns(RuntimeWarning, match="Process pool unavailable"):
        result = run_health_checks(path, output_dir=tmp_path / "reports", workers=2, progress=record_progress)

    assert result.summary == expected.summary
    assert result.rollouts == expected.rollouts
    assert [(completed, total) for phase, completed, total in updates if phase == "checking records serially"] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
