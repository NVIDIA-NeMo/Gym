# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SciCode benchmark metrics: sub-step-weighted accuracy, across-run variability and token accounting.

Rows are shaped as /verify returns them: ``step_usage`` and ``token_usage_version`` come from scicode_agent's
step loop, from its /run or its agent session alike.
"""

import statistics
from typing import Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.metrics import compute_aggregate_metrics
from resources_servers.scicode.app import (
    TOKEN_USAGE_VERSION,
    ScicodeResourcesServer,
    ScicodeResourcesServerConfig,
)


def _server() -> ScicodeResourcesServer:
    return ScicodeResourcesServer(
        config=ScicodeResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="scicode"),
        server_client=MagicMock(spec=ServerClient),
    )


def _usage(output_tokens, input_tokens=10, cached_tokens=2, reasoning_tokens=3):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


_ZERO = _usage(0, input_tokens=0, cached_tokens=0, reasoning_tokens=0)


def _total(usages: list[dict]) -> dict:
    total = _ZERO
    for usage in usages:
        total = {
            "input_tokens": total["input_tokens"] + usage["input_tokens"],
            "output_tokens": total["output_tokens"] + usage["output_tokens"],
            "total_tokens": total["total_tokens"] + usage["total_tokens"],
            "input_tokens_details": {
                "cached_tokens": total["input_tokens_details"]["cached_tokens"]
                + usage["input_tokens_details"]["cached_tokens"]
            },
            "output_tokens_details": {
                "reasoning_tokens": total["output_tokens_details"]["reasoning_tokens"]
                + usage["output_tokens_details"]["reasoning_tokens"]
            },
        }
    return total


def _row(*steps: tuple[str, Optional[dict]], **fields) -> dict:
    """A verified row whose sub-steps ended with these (status, usage) pairs, as the agent's step loop reports them.

    The response's usage sums the generated steps, or is None when one of them has unknown usage.
    """
    step_usage = [
        {
            "step_number": f"1.{i + 1}",
            "status": status,
            "usage": _ZERO if usage is None and status != "generated" else usage,
        }
        for i, (status, usage) in enumerate(steps)
    ]
    generated = [step["usage"] for step in step_usage if step["status"] == "generated"]
    usage = None if any(item is None for item in generated) else _total(generated)
    return {
        "reward": 0.0,
        "token_usage_version": TOKEN_USAGE_VERSION,
        "step_usage": step_usage,
        "response": {"usage": usage},
        **fields,
    }


def _generated(*tokens: Optional[int]) -> list[tuple[str, Optional[dict]]]:
    return [("generated", _usage(t) if t is not None else None) for t in tokens]


def _aggregate(rows):
    server = _server()
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, **row} for i, row in enumerate(rows)]
    return compute_aggregate_metrics(rows, server.compute_metrics, server.get_key_metrics)


def _repeat(idx, passed, total):
    return {
        "_ng_rollout_index": idx,
        "num_steps_passed": passed,
        "num_steps_total": total,
        "problem_accuracy": passed == total,
    }


class TestHeadlineMetrics:
    def test_compute_metrics_subtask_accuracy_is_substep_weighted(self):
        # Two problems: 1/2 and 3/4 passed -> 4/6, NOT the mean of ratios (0.5, 0.75).
        tasks = [
            [{"num_steps_passed": 1, "num_steps_total": 2}],
            [{"num_steps_passed": 3, "num_steps_total": 4}],
        ]
        assert _server().compute_metrics(tasks) == {"subtask_accuracy": 4 / 6}

    def test_compute_metrics_no_steps(self):
        assert _server().compute_metrics([]) == {"subtask_accuracy": 0.0}

    def test_get_key_metrics_includes_subtask_accuracy(self):
        agent_metrics = {"mean/reward": 0.1875, "max/reward": 1.0, "subtask_accuracy": 0.414}
        key = _server().get_key_metrics(agent_metrics)
        assert key["subtask_accuracy"] == 0.414
        assert key["mean/reward"] == 0.1875
        assert "max/reward" not in key  # only mean/* + subtask_accuracy are headline


class TestAcrossRunStats:
    def test_three_runs_hand_computed(self):
        # Two problems x three repeats. Per-run problem accuracy is the mean of
        # problem_accuracy over problems; per-run subtask accuracy is the
        # sub-step-weighted pool - the same definitions as the headline metrics.
        tasks = [
            [_repeat(0, 2, 2), _repeat(1, 1, 2), _repeat(2, 0, 2)],
            [_repeat(0, 3, 4), _repeat(1, 4, 4), _repeat(2, 3, 4)],
        ]
        m = _server().compute_metrics(tasks)
        # Pooled headline is unchanged: (2+1+0+3+4+3) / (3*2 + 3*4) = 13/18.
        assert m["subtask_accuracy"] == pytest.approx(13 / 18)
        problem_runs = [1 / 2, 1 / 2, 0.0]  # run means of problem_accuracy
        subtask_runs = [5 / 6, 5 / 6, 3 / 6]  # per-run pooled sub-step fractions
        assert m["mean/problem_accuracy/std_dev_across_runs"] == pytest.approx(statistics.stdev(problem_runs))
        assert m["subtask_accuracy/std_dev_across_runs"] == pytest.approx(statistics.stdev(subtask_runs))
        assert not any(k.endswith("std_err_across_runs") for k in m)

    def test_single_repeat_emits_only_pooled_metric(self):
        tasks = [[_repeat(0, 1, 2)], [_repeat(0, 3, 4)]]
        assert _server().compute_metrics(tasks) == {"subtask_accuracy": 4 / 6}

    def test_rollout_index_alignment_not_arrival_order(self):
        ordered = [
            [_repeat(0, 2, 2), _repeat(1, 0, 2)],
            [_repeat(0, 4, 4), _repeat(1, 1, 4)],
        ]
        shuffled = [list(reversed(task)) for task in ordered]
        assert _server().compute_metrics(shuffled) == _server().compute_metrics(ordered)

    def test_uneven_repeat_counts_use_min_k(self):
        tasks = [
            [_repeat(0, 2, 2), _repeat(1, 0, 2), _repeat(2, 1, 2)],
            [_repeat(0, 4, 4), _repeat(1, 0, 4)],
        ]
        m = _server().compute_metrics(tasks)
        # k = 2: subtask runs (2+4)/6 = 1.0 and (0+0)/6 = 0.0.
        assert m["subtask_accuracy/std_dev_across_runs"] == pytest.approx(statistics.stdev([1.0, 0.0]))

    def test_identical_runs_zero_std(self):
        tasks = [
            [_repeat(0, 1, 2), _repeat(1, 1, 2)],
            [_repeat(0, 4, 4), _repeat(1, 4, 4)],
        ]
        m = _server().compute_metrics(tasks)
        assert m["mean/problem_accuracy/std_dev_across_runs"] == 0.0
        assert m["subtask_accuracy/std_dev_across_runs"] == 0.0

    def test_get_key_metrics_includes_across_run_stats(self):
        agent_metrics = {
            "mean/problem_accuracy": 0.19,
            "mean/problem_accuracy/std_dev_across_runs": 0.02,
            "subtask_accuracy": 0.41,
            "subtask_accuracy/std_dev_across_runs": 0.015,
            "std/reward": 0.4,
        }
        key = _server().get_key_metrics(agent_metrics)
        assert "mean/problem_accuracy/std_dev_across_runs" in key
        assert "subtask_accuracy/std_dev_across_runs" in key
        assert "std/reward" not in key


class TestTokenAccounting:
    def test_unequal_step_counts_and_repeats_are_pooled(self):
        # Two attempts each: one-step problem (100, 200), three-step problem
        # (300+400+500, 600+700+800). Total 3600 / 4 problems / 8 steps.
        rows = [
            {**_row(*_generated(*tokens)), "_ng_task_index": task_idx, "_ng_rollout_index": repeat_idx}
            for task_idx, attempts in enumerate(([[100], [200]], [[300, 400, 500], [600, 700, 800]]))
            for repeat_idx, tokens in enumerate(attempts)
        ]
        result = _aggregate(rows)
        for name, per_problem, per_subproblem in (("input", 20, 10), ("output", 900, 450), ("total", 920, 460)):
            assert result.key_metrics[f"mean/{name}_tokens_per_problem"] == per_problem
            assert result.key_metrics[f"mean/{name}_tokens_per_subproblem"] == per_subproblem
            assert result.key_metrics[f"mean/{name}_tokens"] == per_problem
        assert result.agent_metrics["num_generated_steps"] == 8
        assert result.agent_metrics["num_subproblems"] == 8
        assert result.agent_metrics["num_steps_with_usage"] == 8
        assert result.key_metrics["generation_coverage"] == 1.0
        assert result.key_metrics["token_usage_complete"] is True
        assert result.agent_metrics["token_usage_version"] == 1
        assert all(isinstance(value, (int, float)) for value in result.agent_metrics.values())
        assert [g["mean/output_tokens"] for g in result.group_level_metrics] == [150, 1650]

    def test_prefilled_and_context_skipped_steps_are_not_generated(self):
        row = _row(
            ("prefilled", None),
            ("generated", _usage(100)),
            ("context_window_exceeded", None),
            ("skipped", None),
        )
        result = _aggregate([row])
        metrics = result.key_metrics
        assert metrics["mean/output_tokens_per_problem"] == 100
        assert metrics["mean/output_tokens_per_subproblem"] == pytest.approx(100 / 3)
        assert metrics["generation_coverage"] == 1 / 3
        assert metrics["token_usage_complete"] is True
        assert result.agent_metrics["num_subproblems"] == 3

    @pytest.mark.parametrize("missing_index", [0, 1])
    def test_missing_usage_stays_unknown_in_collection(self, missing_index):
        tokens = [100, 300]
        tokens[missing_index] = None
        unknown = _row(*_generated(*tokens))
        known = _row(*_generated(200))
        result = _aggregate([unknown, {**known, "_ng_task_index": 1}])
        for name in ("input", "output", "total"):
            for suffix in ("", "_per_problem", "_per_subproblem"):
                assert result.key_metrics[f"mean/{name}_tokens{suffix}"] is None
        assert result.agent_metrics["num_generated_steps"] == 3
        assert result.agent_metrics["num_steps_with_usage"] == 2
        assert result.key_metrics["token_usage_complete"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_usage", [False, True])
    async def test_aggregate_endpoint_suppresses_all_partial_token_statistics(self, missing_usage):
        rows = [
            {
                **_row(*_generated(None if missing_usage and task_idx == repeat_idx == 0 else 100)),
                "_ng_task_index": task_idx,
                "_ng_rollout_index": repeat_idx,
            }
            for task_idx in range(2)
            for repeat_idx in range(2)
        ]
        body = AggregateMetricsRequest(verify_responses=rows)
        original = body.model_dump()
        result = await _server().aggregate_metrics(body)
        assert body.model_dump() == original
        assert result.key_metrics["token_usage_complete"] is not missing_usage
        assert result.key_metrics["generation_coverage"] == 1.0
        assert all("subtask_accuracy" in repeat for repeat in result.repeat_level_metrics)
        assert "mean_across_repeats/subtask_accuracy" in result.agent_metrics
        for section in (result.agent_metrics, result.key_metrics):
            assert all(value is not None and not isinstance(value, str) for value in section.values())
        if missing_usage:
            assert "mean/output_tokens_per_problem" not in result.key_metrics

            def check_no_numeric_tokens(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if "tokens" in key:
                            assert not isinstance(item, (int, float)), (key, item)
                        check_no_numeric_tokens(item)
                elif isinstance(value, list):
                    for item in value:
                        check_no_numeric_tokens(item)

            check_no_numeric_tokens(result.model_dump())
        else:
            assert result.key_metrics["mean/output_tokens_per_problem"] == 100
            assert result.agent_metrics["max/output_tokens"] == 100
            assert all(group["mean/output_tokens"] == 100 for group in result.group_level_metrics)

    @pytest.mark.parametrize("kind", ["empty", "prefilled", "context"])
    def test_no_generations_zero_usage_with_fixed_subproblem_denominator(self, kind):
        steps = {
            "empty": [],
            "prefilled": [("prefilled", None)],
            "context": [("context_window_exceeded", None), ("skipped", None)],
        }[kind]
        metrics = _aggregate([_row(*steps)]).key_metrics
        for name in ("input", "output", "total"):
            assert metrics[f"mean/{name}_tokens_per_problem"] == 0
            assert metrics[f"mean/{name}_tokens_per_subproblem"] == (0 if kind == "context" else None)
        assert metrics["generation_coverage"] == (0 if kind == "context" else None)

    def test_truncated_generation_counts_all_usage(self):
        metrics = _aggregate([_row(("generated", _usage(262000, input_tokens=144)))]).key_metrics
        for scope in ("problem", "subproblem"):
            assert metrics[f"mean/input_tokens_per_{scope}"] == 144
            assert metrics[f"mean/output_tokens_per_{scope}"] == 262000
            assert metrics[f"mean/total_tokens_per_{scope}"] == 262144

    def test_correctness_does_not_filter_token_usage(self):
        row = _row(*_generated(1000, 1000, 1000))
        one_correct = {**row, "step_results": [True, False, False], "num_steps_passed": 1, "num_steps_total": 3}
        two_correct = {**row, "step_results": [True, True, False], "num_steps_passed": 2, "num_steps_total": 3}
        first = _aggregate([one_correct]).key_metrics
        second = _aggregate([two_correct]).key_metrics
        assert first["subtask_accuracy"] == 1 / 3
        assert second["subtask_accuracy"] == 2 / 3
        assert first["mean/output_tokens_per_problem"] == second["mean/output_tokens_per_problem"] == 3000
        assert first["mean/output_tokens_per_subproblem"] == second["mean/output_tokens_per_subproblem"] == 1000

    def test_legacy_rollouts_do_not_gain_whole_problem_metrics(self):
        row = {"reward": 0.0, "response": {"usage": _usage(100)}}
        metrics = _aggregate([row]).key_metrics
        assert metrics["mean/output_tokens"] == 100
        assert "mean/output_tokens_per_problem" not in metrics
        assert "mean/output_tokens_per_subproblem" not in metrics

    def test_mixed_accounting_versions_rejected(self):
        old_row = {"reward": 0.0, "response": {"usage": _usage(200)}}
        with pytest.raises(ValueError, match="different token accounting versions"):
            _aggregate([_row(*_generated(100)), old_row])

    def test_a_counted_failure_row_leaves_token_usage_incomplete(self):
        # count_failure_classes_as_zero adds a failed rollout as its dataset fields and a zero reward, no response.
        failed = {"reward": 0.0, "mask_sample": False, "problem_id": "2"}
        result = _aggregate([_row(*_generated(100, 200)), failed])
        assert result.key_metrics["token_usage_complete"] is False
        assert result.key_metrics["mean/output_tokens_per_problem"] is None
        assert result.key_metrics["mean/output_tokens_per_subproblem"] is None
        assert (result.agent_metrics["num_generated_steps"], result.agent_metrics["num_steps_with_usage"]) == (2, 2)

    def test_only_counted_failure_rows_report_no_token_metrics(self):
        result = _aggregate([{"reward": 0.0, "mask_sample": False}])
        assert "token_usage_complete" not in result.key_metrics

    @pytest.mark.asyncio
    async def test_aggregate_endpoint_with_a_counted_failure_row_exports_no_token_totals(self):
        rows = [
            {**_row(*_generated(100)), "num_steps_passed": 1, "num_steps_total": 1},
            {"reward": 0.0, "mask_sample": False, "problem_id": "2"},
        ]
        rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, **row} for i, row in enumerate(rows)]
        body = AggregateMetricsRequest(verify_responses=rows)
        original = body.model_dump()
        result = await _server().aggregate_metrics(body)
        assert body.model_dump() == original
        assert result.key_metrics["token_usage_complete"] is False
        assert "mean/output_tokens_per_problem" not in result.key_metrics
        token_numbers = {k: v for k, v in result.agent_metrics.items() if "tokens" in k}
        assert all(not isinstance(v, (int, float)) for v in token_numbers.values()), token_numbers


class TestHeadlineRepeatIntervals:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_name", ["scicode", "scicode_verified_with_background_agent"])
    async def test_endpoint_exports_weighted_subtask_and_problem_intervals(self, agent_name):
        rows = [
            {
                **_repeat(repeat, passed, total),
                "_ng_task_index": task,
                "agent_ref": {"name": agent_name},
                "reward": float(passed == total),
            }
            for task, total, values in [(0, 2, [2, 1, 0]), (1, 4, [3, 4, 3])]
            for repeat, passed in enumerate(values)
        ]
        result = await _server().aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
        for metric, mean, sem in [("subtask_accuracy", 13 / 18, 1 / 9), ("mean/problem_accuracy", 1 / 3, 1 / 6)]:
            assert result.key_metrics[f"mean_across_repeats/{metric}"] == pytest.approx(mean)
            assert result.key_metrics[f"ci_low_95_across_repeats/{metric}"] == pytest.approx(
                mean - 4.30265272975 * sem
            )
            assert result.key_metrics[f"ci_high_95_across_repeats/{metric}"] == pytest.approx(
                mean + 4.30265272975 * sem
            )
        assert result.key_metrics["subtask_accuracy"] == pytest.approx(13 / 18)
        assert all("subtask_accuracy" in repeat for repeat in result.repeat_level_metrics)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_name", ["scicode", "scicode_verified_with_background_agent"])
    async def test_single_repeat_has_no_headline_interval(self, agent_name):
        rows = [{**_repeat(0, 1, 2), "_ng_task_index": 0, "agent_ref": {"name": agent_name}, "reward": 0.0}]
        result = await _server().aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
        assert not any(key.startswith("ci_low_95_across_repeats/") for key in result.key_metrics)

    @pytest.mark.asyncio
    async def test_verified_eight_repeat_intervals_match_independent_estimates(self):
        passed_by_problem = [(2, [2, 1, 0, 2, 0, 1, 2, 1]), (4, [4, 4, 3, 2, 4, 0, 4, 3])]
        rows = [
            {
                **_repeat(repeat, passed, total),
                "_ng_task_index": task,
                "agent_ref": {"name": "scicode_verified_with_background_agent"},
                "reward": float(passed == total),
            }
            for task, (total, passed_values) in enumerate(passed_by_problem)
            for repeat, passed in enumerate(passed_values)
        ]
        result = await _server().aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
        subtask_means = [(left + right) / 6 for left, right in zip(*[values for _, values in passed_by_problem])]
        problem_means = [
            (float(left == 2) + float(right == 4)) / 2
            for left, right in zip(*[values for _, values in passed_by_problem])
        ]
        # Student-t critical value for seven degrees of freedom, independent of Gym's helper.
        for metric, values in (("subtask_accuracy", subtask_means), ("mean/problem_accuracy", problem_means)):
            mean = statistics.mean(values)
            half_width = 2.3646242515927844 * statistics.stdev(values) / 8**0.5
            assert result.key_metrics[f"mean_across_repeats/{metric}"] == pytest.approx(mean)
            assert result.key_metrics[f"ci_low_95_across_repeats/{metric}"] == pytest.approx(mean - half_width)
            assert result.key_metrics[f"ci_high_95_across_repeats/{metric}"] == pytest.approx(mean + half_width)
