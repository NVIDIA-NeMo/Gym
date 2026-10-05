# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.nemotron_3_5_super_analysis_shim import derive, intersect, subtract, summarize, total, union


class TestIntervalAlgebra:
    def test_union_merges_overlaps_and_touching(self):
        assert union([(0, 2), (1, 3), (3, 4), (6, 7)]) == [(0, 4), (6, 7)]

    def test_total_never_double_counts(self):
        assert total([(0, 10), (2, 5), (4, 12)]) == 12

    def test_subtract_and_intersect(self):
        assert subtract([(0, 10)], [(2, 4), (8, 20)]) == [(0, 2), (4, 8)]
        assert intersect([(0, 5)], [(3, 9), (4, 6)]) == [(3, 5)]


def _rollout(*, with_offset: bool = True) -> dict:
    # Harness clock: rollout 100..200, agent 110..190. Sandbox clock runs +5 s ahead, so the
    # artifact-timed tool spans and the OpenCode agent span are stored shifted by +5.
    offset = 5.0
    sb = lambda t: t + offset  # noqa: E731
    return {
        "reward": 1.0,
        "ng_perf": {"rollout_started_at": 100.0, "rollout_completed_at": 200.0, "total_latency_ms": 100_000},
        "ng_agent_observations": {
            "records": [
                {
                    "kind": "sandbox",
                    "role": "agent",
                    "clock_offset_s": offset if with_offset else None,
                    "clock_offset_uncertainty_s": 0.02 if with_offset else None,
                }
            ]
        },
        "ng_trajectory": {
            "task_id": "t",
            "rollout_id": "t-0",
            "invocations": [{"invocation_id": "root", "started_at": sb(110.0), "completed_at": sb(190.0)}],
            "model_calls": [
                {
                    "model_call_id": "m1",
                    "started_at": 110.0,
                    "completed_at": 120.0,
                    "duration_ms": 10_000,
                    "model_call_purpose": "agent_step",
                    "model_response_kind": "tool_call",
                    "token_stats": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 100,
                        "reasoning_tokens": 50,
                        "cached_tokens": 500,
                    },
                    "attempts": [
                        {"attempt_index": 1, "status": "timeout", "duration_ms": 4_000},
                        {"attempt_index": 2, "status": "completed", "duration_ms": 6_000},
                    ],
                    "response_metadata": {"engine": {"queue_time_ms": 30.0}},
                },
                {
                    "model_call_id": "m2",
                    "started_at": 150.0,
                    "completed_at": 160.0,
                    "duration_ms": 10_000,
                    "model_call_purpose": "compaction_summary",
                    "model_response_kind": "text",
                    "token_stats": {"prompt_tokens": 3000, "completion_tokens": 200},
                },
                {
                    "model_call_id": "m3",
                    "started_at": 170.0,
                    "completed_at": 190.0,
                    "duration_ms": 20_000,
                    "model_call_purpose": "agent_step",
                    "model_response_kind": "text",
                    "token_stats": {"prompt_tokens": 1500, "completion_tokens": 100, "reasoning_tokens": 0},
                },
            ],
            "tool_calls": [
                {
                    "invocation_id": "root",
                    "tool_call_id": "c1",
                    "tool_name": "bash",
                    "operation": "cd /x && pytest -x",
                    "requested_at": sb(120.5),
                    "started_at": sb(121.0),
                    "completed_at": sb(141.0),
                    "duration_ms": 20_000,
                    "response_received_at": sb(142.0),
                    "timing_source": "artifact",
                    "status": "completed",
                },
            ],
            "compactions": [{"invocation_id": "root", "observed_at": 145.0, "completed_at": 165.0}],
            "gaps": [{"code": "tool_response_boundary_approximate"}],
        },
    }


class TestDerive:
    def test_bounds_and_unions_on_one_clock(self):
        m = derive(_rollout())
        assert m["e2e_rollout_time_s"] == 100.0
        assert m["e2e_agent_time_s"] == 80.0  # shifted back from the sandbox clock
        assert m["pre_agent_time_s"] == 10.0 and m["post_agent_time_s"] == 10.0
        iu = m["interval_union"]
        assert iu["model_s"] == 40.0  # 10 + 10 + 20, disjoint
        assert iu["tool_s"] == 20.0  # 121..141 after -5 shift
        assert iu["overlap_s"] == 0.0
        assert iu["non_model_tool_s"] == pytest.approx(20.0)  # 80 - 60
        # compaction 145..165 minus its own model call 150..160 leaves 10 s of known time
        assert iu["known_no_model_tool_s"] == pytest.approx(10.0)
        assert iu["unexplained_s"] == pytest.approx(10.0)
        assert "unexplained_agent_time_over_threshold" in m["flags"]  # 10/80 > 5%
        assert m["clock"] == {"artifact_timed_spans": True, "offset_s": 5.0, "uncertainty_s": 0.02}

    def test_critical_path_splits_the_agent_interval_exactly(self):
        cp = derive(_rollout())["critical_path"]
        assert cp["model_s"] + cp["tool_s"] + cp["other_s"] == pytest.approx(cp["duration_s"])
        assert cp["model_fraction"] == pytest.approx(0.5)

    def test_counts_tokens_and_retries(self):
        m = derive(_rollout())
        assert m["counts"]["agent_steps"] == 2 and m["counts"]["compactions"] == 1
        assert m["counts"]["valid_tool_action_responses"] == 1 and m["counts"]["valid_final_responses"] == 1
        t = m["tokens"]
        assert t["input"] == 5500 and t["reasoning_share_of_generated"] == pytest.approx(50 / 450)
        assert t["prompt_cache_hit_share"] == pytest.approx(500 / 5500)
        assert t["context_growth"]["first"] == 1000 and t["context_growth"]["last"] == 1500
        r = m["retries"]
        assert (r["attempts_total"], r["failed_model_attempts"], r["calls_with_retry"]) == (2, 1, 1)
        assert r["model_retry_overhead_s"] == pytest.approx(4.0)
        assert m["tool_dispatch_delay_s"]["mean"] == pytest.approx(0.5)
        assert m["tool_observation_delay_s"]["mean"] == pytest.approx(1.0)
        assert m["engine_queue_time_ms"]["mean"] == 30.0
        assert "pytest" in m["tool_time_by_operation"]

    def test_missing_offset_is_flagged_not_guessed(self):
        m = derive(_rollout(with_offset=False))
        assert "clock_offset_unavailable" in m["flags"]
        assert m["coverage"]["attempts"] is True and m["coverage"]["tool_response_boundary_approximate"] is True

    def test_old_captures_report_coverage_instead_of_failing(self):
        m = derive(
            {
                "reward": 0.0,
                "ng_perf": {"total_latency_ms": 5000},
                "ng_trajectory": {
                    "task_id": "t",
                    "rollout_id": "t-1",
                    "invocations": [{"invocation_id": "r"}],
                    "model_calls": [{"model_call_id": "m", "started_at": 1.0, "completed_at": 2.0}],
                },
            }
        )
        assert m["e2e_rollout_time_s"] == 5.0 and m["e2e_agent_time_s"] is None
        assert m["coverage"]["agent_span"] is False and m["retries"]["attempts_total"] is None
        assert m["counts"]["response_kind_unknown"] == 1


def test_summary_aggregates_and_reports_coverage():
    s = summarize([derive(_rollout()), derive(_rollout(with_offset=False))])
    assert s["rollouts"] == 2
    assert s["metrics"]["e2e_agent_time_s"]["p50"] == 80.0
    assert s["coverage"]["attempts"] == {"rollouts": 2, "fraction": 1.0}
    assert s["flags"]["clock_offset_unavailable"] == 1
