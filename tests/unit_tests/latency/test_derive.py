# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived metrics on the two synthetic rollouts: every number below is worked out by hand from the fixtures."""

from __future__ import annotations

import pytest
from ng_latency.derive import DEFINITIONS, FLAGS, derive_trial
from ng_latency.project import project_rollout
from ng_latency.summarize import summarize


@pytest.fixture
def terminus_trial(terminus_record):
    return derive_trial(project_rollout(terminus_record), terminus_record, replica="replica_0")


@pytest.fixture
def opencode_trial(opencode_record):
    return derive_trial(project_rollout(opencode_record), opencode_record, replica="replica_0")


def test_terminus_lifecycle_and_unions(terminus_trial):
    m = terminus_trial["metrics"]
    assert terminus_trial["agent_kind"] == "terminus2" and terminus_trial["outcome"]["success"] is True
    assert m["e2e_rollout_time"] == pytest.approx(94.9995) and m["e2e_agent_time"] == 53.0
    assert m["pre_agent_time"] == 19.0 and m["post_agent_time"] == pytest.approx(22.9995)
    assert m["lifecycle_residual_time"] == pytest.approx(0.0, abs=1e-6)
    assert m["model_union_time"] == 39.0 and m["tool_union_time"] == pytest.approx(3.1)
    assert m["non_model_tool_time"] == pytest.approx(10.9)
    # known: the compaction outside the summary call (1.5 s) and the two screen reads after each batch (0.1 s each)
    assert m["known_no_model_tool_time_by_event"] == {
        "compaction": pytest.approx(1.5),
        "subagent": 0.0,
        "terminus_observation_assembly": pytest.approx(0.2),
    }
    assert m["unexplained_no_model_tool_time"] == pytest.approx(9.2) and m["unexplained_fraction"] == pytest.approx(
        9.2 / 53
    )
    assert m["per_step_unexplained_time"] == pytest.approx(4.6)
    assert "unexplained_gap_over_5pct" in terminus_trial["flags"]


def test_terminus_model_calls_engine_and_retries(terminus_trial):
    m = terminus_trial["metrics"]
    assert m["agent_steps"] == 2 and m["model_calls"] == 3
    assert m["model_calls_by_purpose"] == {"agent_step": 2, "compaction": 1}
    assert m["model_call_time"]["sum"] == 39.0 and m["model_call_time"]["p99"] == 24.0
    assert m["model_engine_queue_time"]["sum"] == pytest.approx(0.08)
    assert m["model_engine_e2e_time"]["sum"] == pytest.approx(9.85 + 4.91 + 9.92)
    assert m["gym_hop_overhead_time"]["sum"] == pytest.approx(0.13 + 0.07 + 0.06)
    assert m["failed_model_attempts"] == {"total": 1, "by_status": {"timeout": 1}}
    assert terminus_trial["series"]["attempts_total"] == [1, 1, 2]
    assert m["model_retry_overhead_time"]["sum"] == 14.0 and m["retry_rate"] == pytest.approx(1 / 3)
    assert m["retry_delay_fraction"] == pytest.approx(14 / 53)
    assert (m["valid_tool_action_responses"], m["valid_final_responses"], m["produced_action"]) == (2, 1, 2)
    assert m["no_action_turn"] == {"total": 0, "by_reason": {}}
    assert m["reasoning_share_of_generated_tokens"] == pytest.approx(50 / 540, abs=1e-6)
    assert m["prompt_cache_hit_share"] == pytest.approx(2100 / 12000, abs=1e-6)
    assert m["output_tokens_per_agent_call"] == 45.0
    assert terminus_trial["series"]["context_length_at_call"] == [1000, 2000] and m["context_growth"] == 1000


def test_terminus_tools_batches_compaction_and_waits(terminus_trial):
    m = terminus_trial["metrics"]
    assert m["tool_calls"] == 3 and m["tool_batches"] == 2
    assert m["tool_execution_time"]["sum"] == pytest.approx(3.1)
    assert sorted(terminus_trial["series"]["tool_dispatch_delay"]) == pytest.approx([0.1, 0.1, 1.1])
    assert m["tool_observation_delay"]["count"] == 2 and m["tool_observation_delay"]["max"] == pytest.approx(0.1)
    assert set(terminus_trial["by_operation"]) == {"ls", "pytest", "wait"}
    assert m["tool_time_by_operation"] == {"timed_out_batches": 0, "error_batches": 0, "empty_batches": 0}
    assert m["observation_volume"]["unit"] == "batch" and m["observation_volume"]["output_chars_total"] == 250
    assert m["compactions"] == 1 and m["compaction_time"]["sum"] == 6.5 and m["compaction_reasons"] == {"proactive": 1}
    assert terminus_trial["series"]["compaction_tokens_before"] == [1080]
    assert m["wait_analysis"] == {
        "waits": 3,
        "wait_time": pytest.approx(3.1),
        "bare_waits": 1,
        "bare_wait_time": 1.0,
        "trivial_waits": 0,
        "trivial_wait_time": 0.0,
    }
    assert m["critical_path_duration"] == 53.0 and m["cp_model_time"] == 39.0
    assert m["cp_tool_time"] == pytest.approx(3.1) and m["cp_other_time"] == pytest.approx(10.9)
    assert terminus_trial["series"]["step_action"] == ["ls", "wait"]
    assert terminus_trial["series"]["step_preceded_by"] == ["first step", "right after a compaction"]


def test_opencode_lifecycle_subagent_and_tools(opencode_trial):
    m = opencode_trial["metrics"]
    assert opencode_trial["agent_kind"] == "opencode" and opencode_trial["outcome"]["success"] is False
    assert m["e2e_agent_time"] == pytest.approx(46.0)
    # pre and post agent are a node-clock difference plus the sandbox's own startup / shutdown durations
    assert m["pre_agent_time"] == pytest.approx(5.0 + 12.0) and m["post_agent_time"] == pytest.approx(31.999 + 4.0)
    assert m["lifecycle_residual_time"] == pytest.approx(111.999 - (17.0 + 46.0 + 35.999))
    assert m["model_union_time"] == pytest.approx(18.0) and m["tool_union_time"] == pytest.approx(27.4)
    assert m["known_no_model_tool_time_by_event"]["opencode_step_tail"] == pytest.approx(0.1)
    assert m["known_no_model_tool_time_by_event"]["opencode_shutdown"] == 0.0  # after agent_end, outside the window
    assert m["unexplained_no_model_tool_time"] == pytest.approx(5.5) and m["agent_steps"] == 2
    assert m["subagent_invocations"] == {"total": 1, "resumed": 0} and m["subagent_time"]["sum"] == 6.0
    assert m["agent_to_subagent_time"]["sum"] == pytest.approx(0.8) and m["subagent_to_agent_time"]["sum"] == 20.5
    assert m["tool_calls"] == 2 and m["tool_execution_time"]["sum"] == pytest.approx(39.2)
    assert m["tool_dispatch_delay"]["max"] == 0.0 and m["tool_observation_delay"]["max"] == 0.0
    assert set(opencode_trial["by_operation"]) == {"bash", "task"}
    assert m["model_calls_by_purpose"] == {"agent_step": 2, "subagent_step": 1}
    assert (m["valid_tool_action_responses"], m["valid_final_responses"]) == (1, 1)
    assert m["parallel_execution_time"] > 0  # the sub-agent ran while bash was still running
    assert opencode_trial["series"]["step_action"] == ["bash+task", "final text"]
    assert opencode_trial["series"]["step_preceded_by"] == ["first step", "after a sub-agent returned"]
    assert set(opencode_trial["flags"]) == {"tool_response_boundary_approximate", "unexplained_gap_over_5pct"}


def test_definitions_cover_every_metric_and_flags_are_documented(terminus_trial, opencode_trial):
    for trial in (terminus_trial, opencode_trial):
        undefined = {name for name in trial["metrics"] if name not in DEFINITIONS and not name.endswith("_by_event")}
        assert undefined <= {"critical_path_node_ids", "compaction_reasons", "model_calls_by_purpose"}, undefined
        assert set(trial["flags"]) <= set(FLAGS)
        assert "caveat_codes" not in trial


def test_summary_splits_by_outcome_and_reports_field_coverage(terminus_trial, opencode_trial):
    summary = summarize([terminus_trial, opencode_trial])
    assert summary["trials"] == 2 and summary["by_outcome"] == {"failure": 1, "success": 1}
    assert summary["agent_kinds"] == {"opencode": 1, "terminus2": 1}
    agent_time = summary["metrics"]["e2e_agent_time"]
    assert (
        agent_time["all"]["count"] == 2
        and agent_time["success"]["mean"] == 53.0
        and agent_time["failure"]["mean"] == 46.0
    )
    assert agent_time["all"]["p99"] == 53.0
    assert summary["metrics"]["model_call_time"]["pooled"]["count"] == 6
    assert summary["metrics"]["model_call_time"]["by_outcome"]["success"]["pooled"]["p99"] == 24.0
    assert summary["field_coverage"]["subagent_time"] == {"trials_with_value": 1, "fraction": 0.5}
    assert summary["field_coverage"]["e2e_agent_time"] == {"trials_with_value": 2, "fraction": 1.0}
    assert summary["flags"] == {"tool_response_boundary_approximate": 1, "unexplained_gap_over_5pct": 2}
    assert "caveat_codes" not in summary
