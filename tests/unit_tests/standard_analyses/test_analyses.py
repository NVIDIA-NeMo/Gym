# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from dataclasses import replace

import pytest

from nemo_gym.standard_analyses import Call, Compaction, Options, Rollout, Tool, analyze
from nemo_gym.standard_analyses.models import Table
from tests.unit_tests.standard_analyses.fixtures import demo_records


def rollout(repeat=0, *, task="a", success=True, seconds=10, tokens=100, **kwargs):
    return Rollout(
        run_id="run",
        task_id=task,
        repeat=repeat,
        agent="agent",
        model="model",
        benchmark="bench",
        config_id="policy-v1",
        success=success,
        capture_complete=True,
        calls=(Call("c", 0, tokens, 20, duration_s=seconds),),
        **kwargs,
    )


def result(name, rows, **kwargs):
    return next(r for r in analyze(rows, **kwargs).results if r.name == name)


def test_brian_thresholds_use_call_time_and_output_including_reasoning_once():
    r = replace(
        rollout(),
        calls=(
            Call("a", 0, 1000, 500, duration_s=1),
            Call("b", 1, 2000, 1000, duration_s=3),
            Call("c", 2, 10000, 3000, duration_s=6),
            Call("d", 3, 12000, 11000, duration_s=10),
        ),
    )
    a = result("long_calls", [r])
    assert a.metrics["output_gt_1000_call_fraction"] == 0.75
    assert a.metrics["output_gt_1000_time_fraction"] == 0.95
    assert a.metrics["output_gt_10000_call_fraction"] == 0.25
    assert a.metrics["output_gt_10000_time_fraction"] == 0.5
    assert a.metrics["output_gt_10000_token_fraction"] == 12000 / 25000
    assert a.metrics["reasoning_gt_1000_call_fraction"] == 0.5


def test_missing_reasoning_is_unknown_and_does_not_become_zero():
    r = replace(rollout(), calls=(Call("a", 0, 2000, None, duration_s=3),))
    a = result("long_calls", [r])
    assert a.metrics["output_gt_1000_call_fraction"] == 1
    assert a.metrics["reasoning_gt_1000_call_fraction"] is None
    assert a.coverage["calls_missing_reasoning"] == 1
    assert a.status == "partial"


def test_reasoning_only_measurements_remain_available_with_output_unknown():
    r = replace(rollout(), calls=(Call("a", 0, reasoning_tokens=2000, duration_s=3),))
    a = result("long_calls", [r])
    assert a.status == "partial"
    assert a.metrics["reasoning_gt_1000_call_fraction"] == 1
    assert a.metrics["output_gt_1000_call_fraction"] is None


@pytest.mark.parametrize("kwargs", [{"max_rows": 1.5}, {"max_rows": True}, {"thresholds": (True,)}, {"pass_k": (0,)}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        Options(**kwargs)


def test_time_share_denominator_needs_both_token_and_time_measurements():
    r = replace(
        rollout(),
        calls=(
            Call("a", 0, 2000, duration_s=3),
            Call("b", 1, None, duration_s=100),
            Call("c", 2, 1, duration_s=1),
        ),
    )
    assert result("long_calls", [r]).metrics["output_gt_1000_time_fraction"] == 0.75


def test_repeat_excess_baseline_in_denominator_and_success_scopes():
    rows = [rollout(0, seconds=10), rollout(1, seconds=20), rollout(2, success=False, seconds=30)]
    a = result("repeat_excess", rows)
    assert a.metrics["all_scored_excess_fraction"] == 0.5
    assert a.metrics["successful_excess_fraction"] == 10 / 30
    assert a.metrics["all_scored_compared_rollouts"] == 3
    assert a.tables["detail"].rows[0][7] == 0


def test_baseline_tie_breaks_by_repeat_index_and_does_not_choose_failed_fastest():
    rows = [rollout(5, seconds=10), rollout(2, seconds=10), rollout(0, success=False, seconds=1)]
    a = result("repeat_excess", rows)
    assert {r[5] for r in a.tables["detail"].rows} == {2}
    assert a.metrics["all_scored_excess_fraction"] == 0


def test_unknown_completeness_missing_time_unhealthy_and_unscored_excluded():
    rows = [
        rollout(0),
        replace(rollout(1), capture_complete=None),
        rollout(2, seconds=None),
        rollout(3, health="unhealthy"),
        rollout(4, success=None),
    ]
    a = result("repeat_excess", rows)
    assert a.status == "unavailable"
    assert a.metrics["all_scored_excess_fraction"] is None
    assert a.coverage["rollouts_usable"] == 1


def test_unknown_health_is_reported_without_becoming_known_healthy():
    a = result("repeat_excess", [rollout(0), rollout(1, seconds=20)])
    assert a.coverage["rollouts_health_unknown"] == 2
    assert a.metrics["all_scored_compared_rollouts"] == 2


def test_single_repeat_and_no_success_are_explicitly_unavailable():
    assert result("repeat_excess", [rollout()]).status == "unavailable"
    assert result("repeat_excess", [rollout(0, success=False), rollout(1, success=False)]).status == "unavailable"


def test_identical_shard_duplicates_deduplicate_conflicting_attempts_fail():
    report = analyze([rollout(), rollout()])
    assert report.results[0].coverage["rollouts_total"] == 1
    with pytest.raises(ValueError, match="Conflicting attempts"):
        analyze([rollout(), rollout(seconds=20)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other"),
        ("benchmark", "other"),
        ("agent", "other"),
        ("model", "other"),
        ("config_id", "other"),
    ],
)
def test_incompatible_cohorts_never_compare(field, value):
    with pytest.raises(ValueError, match="cohort"):
        analyze([rollout(), replace(rollout(1), **{field: value})])


def test_cumulative_tokens_propagate_missing_measurements():
    r = replace(rollout(), calls=(Call("a", 0, 3), Call("b", 1), Call("c", 2, 4)))
    rows = result("token_accumulation", [r]).tables["detail"].rows
    assert [r[8] for r in rows] == [3, None, None]


def test_pass_at_k_estimator_and_task_specific_coverage():
    rows = [rollout(i, success=i == 0) for i in range(4)] + [rollout(task="b")]
    a = result("pass_at_k", rows)
    assert a.metrics["pass_at_1"] == 0.625
    assert a.metrics["pass_at_2"] == 0.5
    assert a.metrics["pass_at_4"] == 1
    assert a.metrics["pass_at_8"] is None
    assert a.metrics["tasks_at_1"] == 2
    assert a.metrics["tasks_at_2"] == 1


def test_runtime_requires_real_wall_time_and_uses_one_median_per_task():
    assert result("runtime_tails", [rollout()]).status == "unavailable"
    rows = [rollout(0, wall_s=1), rollout(1, wall_s=3), rollout(task="b", wall_s=100)]
    a = result("runtime_tails", rows)
    assert a.metrics["task_wall_s_p90"] == pytest.approx(90.2)
    assert a.tables["detail"].rows[0][0] == "b"
    assert a.tables["detail"].rows[0][4] is True


def test_parallel_call_time_is_not_agent_wall_time():
    r = replace(
        rollout(),
        wall_s=10,
        calls=(Call("a", 0, duration_s=10, started_at=0),),
        tools=(Tool("t", duration_s=10, started_at=0),),
    )
    row = result("time_distribution", [r]).tables["detail"].rows[0]
    assert row[2:7] == [10, 10, 0.5, 10, 10]


def test_compaction_comparison_respects_invocation_and_adjacent_boundaries():
    calls = tuple(
        Call(str(i), i, 100, reasoning_tokens=v, invocation_id="main") for i, v in enumerate((10, 10, 20, 20, 40, 40))
    )
    r = replace(rollout(), calls=calls, compactions=(Compaction("main", "2"), Compaction("main", "4")))
    a = result("compaction_impact", [r])
    assert [row[8] for row in a.tables["detail"].rows] == [2, 2]
    assert result("compaction_impact", [replace(r, compactions=(Compaction("other", "2"),))]).status == "unavailable"


def test_model_tool_environment_errors_are_separate_observations():
    r = replace(
        rollout(),
        calls=(Call("a", 0, duration_s=301, status="upstream_error"),),
        tools=(Tool("t", status="timeout"),),
        environment_status="oom",
    )
    a = result("system_flags", [r])
    assert a.metrics["error_count"] == 3
    assert a.metrics["slow_call_count"] == 1
    assert {row[2] for row in a.tables["detail"].rows} == {"model", "tool", "environment"}


def test_display_truncation_never_changes_aggregate_metrics():
    rows = demo_records()
    full = analyze(rows)
    bounded = analyze(rows, Options(max_rows=1))
    for a, b in zip(full.results, bounded.results):
        assert a.metrics == b.metrics
        assert len(b.tables["detail"].rows) <= 1
        assert b.tables["detail"].total_rows == a.tables["detail"].total_rows


def test_fixed_registry_supports_disable_and_rejects_unknown_names():
    assert "long_calls" not in [r.name for r in analyze([rollout()], disabled=["long_calls"]).results]
    with pytest.raises(ValueError, match="Unknown analyses"):
        analyze([rollout()], disabled=["my_custom_plugin"])


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, True])
def test_invalid_measurements_rejected(value):
    with pytest.raises(ValueError):
        Call("a", 0, duration_s=value)


def test_duplicate_calls_invalid_reasoning_and_table_width_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        replace(rollout(), calls=(Call("same", 0), Call("same", 1)))
    with pytest.raises(ValueError, match="exceed"):
        Call("a", 0, 1, 2)
    with pytest.raises(ValueError, match="width"):
        Table.build(["a"], [[1, 2]], 1)


def test_result_json_has_no_nan_and_is_order_independent():
    rows = demo_records()
    a = analyze(rows).to_dict()
    b = analyze(list(reversed(rows))).to_dict()
    assert a == b
    json.dumps(a, allow_nan=False)
