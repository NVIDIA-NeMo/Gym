# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-checks against the harnesses' and Gym's own counters, on the synthetic runs."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
from ng_latency.build import build_run_timeline
from ng_latency.check import run_checks

from tests.unit_tests.latency.conftest import opencode_row, terminus_row


def _run(tmp_path: Path, rows: list[dict]) -> Path:
    run_dir = tmp_path / "replica_0"
    run_dir.mkdir()
    (run_dir / "rollouts.jsonl").write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    build_run_timeline(run_dir, tokenize_url=None)
    return run_dir


def test_consistent_records_pass_every_check(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, [terminus_row(), opencode_row()])
    report = run_checks(run_dir)
    assert report["passed"] and report["failed_checks"] == [] and report["warnings"] == []
    names = {c["name"] for c in report["checks"]}
    assert names >= {
        "agent_span_count",
        "terminus_model_call_times",
        "terminus_command_exec_times",
        "terminus_time_taken",
        "terminus_usages",
        "terminus_compaction_counts",
        "perf_total_latency",
        "perf_tool_calls",
        "opencode_token_accounting",
        "engine_within_server_within_call",
        "server_coverage",
        "unverified_joins",
    }
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["terminus_model_call_times"]["samples"] == 4 and by_name["terminus_usages"]["samples"] == 10
    assert by_name["server_coverage"]["details"] == ["coverage=1.00 threshold=0.99"]
    assert json.loads((run_dir / "metrics" / "timeline_check.json").read_text())["passed"] is True


def test_a_harness_timer_that_disagrees_with_its_span_fails_the_check(tmp_path: Path) -> None:
    row = terminus_row()
    row["model_call_times"][2] = 23.0  # Harbor timed 23 s, the harness span says 24 s
    row["usages"][0]["input_tokens"] = 999
    report = run_checks(_run(tmp_path, [row]))
    assert not report["passed"] and set(report["failed_checks"]) == {"terminus_model_call_times", "terminus_usages"}
    [detail] = [c for c in report["checks"] if c["name"] == "terminus_model_call_times"][0]["details"]
    assert detail.startswith("4-1: call[2]: expected=23.0 actual=24.0")


def test_a_call_without_a_server_record_is_a_coverage_warning_not_a_failure(tmp_path: Path) -> None:
    row = opencode_row()
    for key in ("server_started_at", "server_completed_at", "server_duration_ms"):
        row["ng_trajectory"]["model_calls"][1][key] = None  # the sub-agent step; the main session stays checkable
    row["ng_trajectory"]["gaps"].append({"code": "model_call_join_unverified", "invocation_id": "root"})
    report = run_checks(_run(tmp_path, [row]))
    assert report["passed"] and set(report["warnings"]) == {"server_coverage", "unverified_joins"}
    coverage = [c for c in report["checks"] if c["name"] == "server_coverage"][0]
    assert coverage["details"] == ["coverage=0.67 threshold=0.99"]


def test_engine_time_longer_than_the_server_span_is_reported(tmp_path: Path) -> None:
    row = terminus_row()
    row["ng_trajectory"]["model_calls"][0]["response_metadata"]["engine"]["generation_time_ms"] = 20_000.0
    report = run_checks(_run(tmp_path, [row]))
    assert report["failed_checks"] == ["engine_within_server_within_call"]
