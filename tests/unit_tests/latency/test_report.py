# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark report on a synthetic job directory with one replica, both agent kinds and one sidecar failure."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pytest
from ng_latency.analyze import run_analysis
from ng_latency.build import build_run_timeline
from ng_latency.check import run_checks
from ng_latency.limitations import LIMITATIONS
from ng_latency.report import build_summary, occupancy, scan_timeline, state_bands, write_report

from tests.unit_tests.latency.conftest import opencode_row, terminus_row


@pytest.fixture(scope="module")
def job_dir(tmp_path_factory) -> Path:
    job = tmp_path_factory.mktemp("job")
    replica = job / "replica_0"
    replica.mkdir()
    rows = [terminus_row(), opencode_row()]
    (replica / "rollouts.jsonl").write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    (replica / "rollouts_failures.jsonl").write_bytes(
        orjson.dumps({"_ng_failure_class": "client_error", "task_id": "task-7"}) + b"\n"
    )
    (replica / "rollouts_materialized_inputs.jsonl").write_bytes(b'{"a":1}\n' * 3)
    (replica / "metrics").mkdir()
    (replica / "metrics" / "run_info.json").write_text(
        json.dumps({"benchmark": "swebench", "slurm_job_id": "123", "node": "n1", "sandbox_timeout_s": 3600})
    )
    build_run_timeline(replica, tokenize_url=None)
    run_checks(replica)
    run_analysis(replica)
    return job


@pytest.fixture(scope="module")
def summary(job_dir):
    return build_summary(job_dir)


def test_outcomes_count_dispatched_scored_and_sidecar_failures(summary):
    outcomes = summary["outcomes"]
    assert outcomes["rollouts"] == 2 and outcomes["planned"] == 3
    assert outcomes["dispatched"] == 3 and outcomes["scoring_denominator"] == 2
    assert outcomes["collector_failures"] == 1 and outcomes["collector_failures_by_class"] == {"client_error": 1}
    assert outcomes["failures_path"].endswith("replica_0/rollouts_failures.jsonl")
    assert outcomes["success"] == 1 and outcomes["failure"] == 1


def test_validation_uses_server_coverage_and_run_has_no_sessions(summary):
    [replica] = summary["validation"]["replicas"]
    assert replica["server_coverage"] == 1.0 and replica["check_passed"] is True
    assert summary["validation"]["trusted"] is True
    assert "sessions" not in summary["run"] and summary["run"]["slurm_job_id"] == "123"
    timing = summary["time"]
    # first rollout start to last rollout end, on the node clock; both synthetic rollouts start at T0
    assert timing["benchmark_run_time_s"] == pytest.approx(111.999, abs=1e-3)
    assert timing["wall_clock_s"] == {
        "overall": timing["benchmark_run_time_s"],
        "replica_0": timing["benchmark_run_time_s"],
    }
    assert "downtime_s" not in timing


def test_limitations_flags_and_timeline_links(summary):
    assert summary["limitations"] == [{"title": t, "what": w, "implication": i} for t, w, i in LIMITATIONS]
    quality = summary["data_quality"]
    assert "caveat_codes" not in quality
    assert quality["flags"]["unexplained_gap_over_5pct"]["count"] == 2
    assert quality["flags"]["unexplained_gap_over_5pct"]["text"]
    pages = {r["rollout_id"]: r["page"] for r in summary["per_task"]}
    assert pages["4-1"].endswith("timelines.html#replica_0/4-1")
    assert "caveats" not in summary["per_task"][0]


def test_per_step_tax_and_engine_vs_agent_views(summary):
    where = summary["where_time_goes"]
    tax = where["per_step_tax"]
    assert tax["steps"] == 4 and tax["unexplained_per_step"]["mean"] == pytest.approx((4.6 + 2.75) / 2)
    assert tax["model_per_step"]["mean"] == pytest.approx((39.0 / 2 + 18.0 / 2) / 2)
    eva = where["engine_vs_agent"]
    assert eva["calls"] == 6  # every server record of both rollouts carries vLLM's engine durations
    # generated tokens come from the server record (completion = output + reasoning): 60, 400, 80 / 60, 30, 40
    assert {b["bucket"]: b["calls"] for b in eva["buckets"]} == {"<100": 5, "100-1k": 1}
    assert all(b["agent_s"]["median"] >= b["engine_e2e_s"]["median"] for b in eva["buckets"])
    dist = summary["distributions"]
    assert set(dist["e2e_agent_time"]) == {"success", "failure"} and dist["e2e_agent_time"]["success"]["values"] == [
        53.0
    ]
    assert dist["model_call_time"]["all"]["ecdf"][-1] == [24.0, 1.0]


def test_occupancy_and_state_bands_without_sessions(job_dir):
    traces = scan_timeline(job_dir / "replica_0" / "metrics" / "rollouts_timeline.jsonl", "replica_0")
    occ = occupancy(traces, bin_s=30.0)
    assert occ["rollouts"][0][2] == 0.0 and occ["bins"][0]["t"] == 0.0
    assert all(b["active"] == b["in_model"] + b["in_tool"] + b["elsewhere"] for b in occ["bins"])
    bands = state_bands(traces, bins=10)
    assert len(bands["bins"]) == 10 and bands["rollouts"] == 2
    assert all(abs(sum(b[k] for k in ("model", "tool", "other")) - 1.0) < 1e-6 for b in bands["bins"])


def test_write_report_puts_limitations_first(job_dir):
    result = write_report(job_dir)
    assert result["trusted"] is True and result["rollouts"] == 2
    page = (job_dir / "metrics" / "report.html").read_text(encoding="utf-8")
    assert page.index('id="limitations"') < page.index('id="run"')
    assert "Two clocks" in page and "rollouts_failures.jsonl" in page and "caveat" not in page.lower()
    assert "timelines.html#replica_0/4-1" in page


def test_failed_vs_successful_reads_the_summary_split(summary):
    """One successful (terminus) and one failed (opencode) row: both columns exist and ratios are per metric."""
    side = summary["failed_vs_successful"]
    rows = {r["metric"]: r for r in side["rows"]}
    assert "e2e_agent_time" in rows and "model_calls" in rows
    agent = rows["e2e_agent_time"]
    assert agent["n_success"] == 1 and agent["n_failure"] == 1
    assert agent["success"] is not None and agent["failure"] is not None
    if agent["success"] and agent["success"] > 0:
        assert agent["ratio"] == round(agent["failure"] / agent["success"], 3)
    # a metric present for neither group is omitted rather than shown as a row of dashes
    assert all(r["success"] is not None or r["failure"] is not None for r in side["rows"])


def test_write_report_shows_failed_vs_successful_under_outcomes(job_dir):
    write_report(job_dir)
    page = (job_dir / "metrics" / "report.html").read_text(encoding="utf-8")
    assert page.index('id="outcome-at-a-glance"') < page.index('id="failed-vs-successful"')
    assert "Failed vs successful rollouts" in page
