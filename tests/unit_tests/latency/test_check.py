# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-checks against the harnesses' and Gym's own counters, on the synthetic runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import orjson
import pytest
from ng_latency.build import build_run_timeline
from ng_latency.check import run_checks
from ng_latency.derive import derive_trial
from ng_latency.project import project_rollout

from tests.unit_tests.latency.conftest import T0, opencode_row, terminus_row


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


def test_a_decision_without_commands_has_harbors_screen_read_but_no_batch(tmp_path: Path) -> None:
    """Harbor runs its command batch for every decision it could parse, with or without commands: the completion
    claim has an exec time (the screen read alone) and no batch span, a response its parser rejected has neither.
    The check walks the lists in decision order; the derivation counts the claim as an empty batch and the time
    Terminus spends after either decision as known harness time."""
    row = terminus_row()
    t = T0

    def decision(call_id: str, start: float, end: float, **fields) -> dict:
        call = copy.deepcopy(row["ng_trajectory"]["model_calls"][0])
        call.update(
            model_call_id=call_id,
            client_call_id=f"t2-{call_id}",
            started_at=t + start,
            completed_at=t + end,
            duration_ms=(end - start) * 1000,
            processed_at=t + end + 0.01,
            server_started_at=t + start + 0.01,
            server_completed_at=t + end - 0.01,
            server_duration_ms=(end - start) * 1000 - 20,
            parsed_tool_calls=0,
            **fields,
        )
        call["response_metadata"] = {**call["response_metadata"], "response_id": f"resp_{call_id}", "engine": None}
        call["attempts"] = [
            {
                **call["attempts"][0],
                "started_at": t + start,
                "completed_at": t + end,
                "duration_ms": (end - start) * 1000,
            }
        ]
        return call

    unparsable = decision("cap-5", 71.4, 71.5, task_complete=False, parse_error=True)
    claim = decision("cap-4", 71.6, 71.8, task_complete=True, parse_error=False)
    row["ng_trajectory"]["model_calls"] += [unparsable, claim]
    row["model_call_times"] += [0.1, 0.2]
    row["command_exec_times"].append(0.2)  # the claim's screen read; nothing for the unparsable response
    usage = {"input_tokens": 1000, "output_tokens": 60, "output_tokens_details": {"reasoning_tokens": 20}}
    row["usages"] += [usage, usage]
    run_dir = _run(tmp_path, [row])
    report = run_checks(run_dir)
    assert report["passed"], report["failed_checks"]
    batches = [c for c in report["checks"] if c["name"] == "terminus_command_exec_times"][0]
    assert batches["samples"] == 5 and batches["details"] == []
    trial = derive_trial(project_rollout(row), row, replica="replica_0")
    m = trial["metrics"]
    assert m["agent_steps"] == 4 and m["tool_batches"] == 2
    assert m["tool_time_by_operation"]["empty_batches"] == 1
    assert m["no_action_turn"] == {"total": 1, "by_reason": {"parse_failure": 1}}
    # the two batches' reads (0.1 s each), the prompt after the unparsable response (0.1 s) and the claim's read
    # until the agent's end (0.2 s)
    assert m["known_no_model_tool_time_by_event"]["terminus_observation_assembly"] == pytest.approx(0.5)
    assert m["unexplained_no_model_tool_time"] == pytest.approx(9.2 - 0.3 - 0.3)


def test_engine_time_longer_than_the_server_span_is_reported(tmp_path: Path) -> None:
    row = terminus_row()
    row["ng_trajectory"]["model_calls"][0]["response_metadata"]["engine"]["generation_time_ms"] = 20_000.0
    report = run_checks(_run(tmp_path, [row]))
    assert report["failed_checks"] == ["engine_within_server_within_call"]
