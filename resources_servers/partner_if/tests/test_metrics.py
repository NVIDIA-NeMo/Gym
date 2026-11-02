# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark metrics: all-IF pass and partial IF credit, overall and per origin. Synthetic rollouts only."""

import pytest
from pytest import approx

from nemo_gym.config import AggregateMetricsRequest
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin, origin_subset, score
from resources_servers.partner_if.tests.test_app import server


def rollout(task, repeat, reward, passed, total, origin):
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "if_result": {"passed": passed, "total": total},
        "verifier_metadata": {"partner_if": {"task_id": f"T{task}", "origin": origin}},
    }


def suite():
    """Three tasks, three repeats each; two spellings of one origin."""
    attempts = [
        ("VendorA", [(1, 2, 2), (0, 1, 2), (1, 2, 2)]),
        ("vendora", [(0, 0, 2), (0, 1, 2), (0, 1, 2)]),
        ("VendorB", [(1, 3, 3), (1, 3, 3), (0, 2, 3)]),
    ]
    return [
        [rollout(task, repeat, *result, origin) for repeat, result in enumerate(results)]
        for task, (origin, results) in enumerate(attempts)
    ]


# Per-task partial credit across the three repeats: 5/6, 1/3 and 8/9.
HEADLINE = {
    "pass@1[avg-of-3]/all_if_pass": 100 * 4 / 9,
    "pass@1[avg-of-3]/if_partial": 100 * (5 / 6 + 1 / 3 + 8 / 9) / 3,
    "pass@3/all_if_pass": 100 * 2 / 3,
}


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"reward": 1, "if_result": {"passed": 2, "total": 2}}, {"all_if_pass": 1.0, "if_partial": 1.0}),
        ({"reward": 0, "if_result": {"passed": 1, "total": 4}}, {"all_if_pass": 0.0, "if_partial": 0.25}),
        # A rollout counted as zero without running has no IF result and earns no partial credit.
        ({"reward": 0}, {"all_if_pass": 0.0, "if_partial": 0.0}),
        ({"reward": 0, "if_result": None}, {"all_if_pass": 0.0, "if_partial": 0.0}),
    ],
)
def test_score_is_binary_all_if_pass_with_partial_credit(result, expected):
    assert score(result) == expected


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"partner_if": {"origin": "VendorA"}}, "origin/vendora"),
        ({"partner_if": {"origin": "VENDORA"}}, "origin/vendora"),
        ({"partner_if": {"origin": "VendorB"}}, "origin/vendorb"),
        ({"partner_if": {"origin": ""}}, None),
        ({"partner_if": {}}, None),
        ({}, None),
        (None, None),
    ],
)
def test_origin_subset_is_case_insensitive_and_optional(metadata, expected):
    assert origin_subset({"reward": 1, "verifier_metadata": metadata}) == expected


def test_metrics_report_pass_at_k_and_average_for_both_scores_and_each_origin():
    metrics = PartnerIFMetricsMixin().compute_metrics(suite())
    assert {key: metrics[key] for key in HEADLINE} == approx(HEADLINE)
    assert metrics["pass@1/all_if_pass"] == approx(100 * 4 / 9)
    # Partial credit is continuous, so pass@k is the best of the first k attempts.
    assert metrics["pass@1/if_partial"] == approx(100 * 2 / 3)
    assert metrics["pass@3/if_partial"] == approx(100 * 2.5 / 3)
    # One origin is one subset, however its name is spelled.
    assert {key.split("/")[1] for key in metrics if key.startswith("origin/")} == {"vendora", "vendorb"}
    assert metrics["origin/vendora/pass@1[avg-of-3]/all_if_pass"] == approx(100 / 3)
    assert metrics["origin/vendora/pass@1[avg-of-3]/if_partial"] == approx(100 * (5 / 6 + 1 / 3) / 2)
    assert metrics["origin/vendora/pass@3/all_if_pass"] == approx(50)
    assert metrics["origin/vendorb/pass@3/all_if_pass"] == approx(100)
    assert metrics["origin/vendorb/pass@1[avg-of-3]/if_partial"] == approx(100 * 8 / 9)


def test_zero_row_scores_nothing_and_keeps_its_task_in_the_origin_subset():
    task = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0.0},
        rollout(0, 1, 1, 2, 2, "VendorC"),
        rollout(0, 2, 0, 1, 2, "VendorC"),
    ]
    metrics = PartnerIFMetricsMixin().compute_metrics([task])
    assert metrics["pass@1[avg-of-3]/all_if_pass"] == approx(100 / 3)
    assert metrics["pass@1[avg-of-3]/if_partial"] == approx(100 * 1.5 / 3)
    assert metrics["origin/vendorc/pass@1[avg-of-3]/if_partial"] == approx(100 * 1.5 / 3)


def test_key_metrics_are_the_highest_k_averages_and_all_if_pass_at_k():
    mixin = PartnerIFMetricsMixin()
    metrics = mixin.compute_metrics(suite())
    assert any(key.endswith("std_dev_across_runs") for key in metrics)
    assert mixin.get_key_metrics(metrics) == approx(HEADLINE)


async def test_scorer_aggregates_measured_rollouts_and_reports_masked_ones_as_coverage():
    rows = [row for task in suite() for row in task]
    masked = {"_ng_task_index": 3, "_ng_rollout_index": 0, "reward": 0.0, "mask_sample": True}
    result = await server().aggregate_metrics(AggregateMetricsRequest(verify_responses=[*rows, masked]))
    assert result.key_metrics == approx(
        HEADLINE
        | {
            "coverage/measured_rollouts": 9,
            "coverage/masked_rollouts": 1,
            "coverage/measured_tasks": 3,
            "coverage/fully_masked_tasks": 1,
        }
    )
    assert result.agent_metrics["origin/vendora/pass@3/all_if_pass"] == approx(50)


@pytest.mark.parametrize("module,name", [("app", "PartnerIFResourcesServer"), ("swe_app", "SWEIFResourcesServer")])
def test_shared_and_swe_scorers_use_the_benchmark_metrics(module, name):
    from importlib import import_module

    scorer = getattr(import_module(f"resources_servers.partner_if.{module}"), name)
    assert scorer.compute_metrics is PartnerIFMetricsMixin.compute_metrics
    assert scorer.get_key_metrics is PartnerIFMetricsMixin.get_key_metrics
