# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError
from scipy import stats

from nemo_gym.comparison.loading import LoadedRun
from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests.paired import (
    PairedTestConfig,
    build_report,
    paired_task_deltas,
    render_markdown,
    resolve_metrics,
    run_metric,
    summary,
)
from tests.unit_tests.test_compare import _entry, _group, _write_run


BASE = {"baseline_rollouts_jsonl_fpath": "a.jsonl", "candidate_rollouts_jsonl_fpaths": ["b.jsonl"]}


def _run(groups, key_metrics=None) -> LoadedRun:
    return LoadedRun(
        agent_name="agent",
        agent_metrics={},
        key_metrics=key_metrics or {},
        group_level_metrics=groups,
        num_tasks=len(groups),
    )


def _g(task_index, **fields):
    """A bare `group_level_metrics` entry (distinct from test_compare's `_group`, which computes mean/min/max)."""
    return {"_ng_task_index": task_index, **fields}


def two_runs(tmp_path):
    baseline = _write_run(
        tmp_path,
        "run_a",
        [
            _entry(
                key_metrics={"mean/reward": 0.75, "pass@1/accuracy": 75.0},
                groups=[_group(i, [1.0, 1.0]) for i in range(6)],
            )
        ],
    )
    candidate = _write_run(
        tmp_path,
        "run_b",
        [
            _entry(
                key_metrics={"mean/reward": 0.5, "pass@1/accuracy": 50.0},
                groups=[_group(i, [0.0, 1.0]) for i in range(6)],
            )
        ],
    )
    return baseline, candidate


def config_for(baseline, candidate, **overrides) -> PairedTestConfig:
    return PairedTestConfig.model_validate(
        {
            "baseline_rollouts_jsonl_fpath": str(baseline),
            "candidate_rollouts_jsonl_fpaths": [str(candidate)],
            **overrides,
        }
    )


class TestPairedTestConfig:
    def test_valid_minimal_config(self):
        config = PairedTestConfig.model_validate(BASE)
        assert config.test == "paired" and config.metric is None and config.margin is None

    @pytest.mark.parametrize("margin", [0, -0.01, -5])
    def test_non_positive_margin_is_rejected(self, margin):
        with pytest.raises(ValidationError, match="--margin must be a positive number"):
            PairedTestConfig.model_validate({**BASE, "margin": margin})

    def test_filename_parts_reflect_the_framing_and_the_metric_subset(self):
        assert PairedTestConfig.model_validate(BASE).filename_parts() == ["two-sided"]
        assert PairedTestConfig.model_validate({**BASE, "margin": 0.01}).filename_parts() == ["margin-0.01"]
        assert PairedTestConfig.model_validate({**BASE, "metric": ["reward", "a/b"]}).filename_parts() == [
            "metric-reward+a-b",
            "two-sided",
        ]


class TestPairedTaskDeltas:
    def test_ties_are_included_not_filtered(self):
        baseline = _run([_g(0, **{"mean/reward": 0.5}), _g(1, **{"mean/reward": 0.3})])
        candidate = _run([_g(0, **{"mean/reward": 0.5}), _g(1, **{"mean/reward": 0.1})])
        assert paired_task_deltas(baseline, candidate, "reward") == pytest.approx([0.0, -0.2])

    def test_missing_metric_on_one_side_drops_that_task_only(self):
        baseline = _run([_g(0, **{"mean/reward": 0.5}), _g(1, **{"mean/reward": 0.5})])
        candidate = _run([_g(0, **{"mean/reward": 0.4}), _g(1)])
        assert paired_task_deltas(baseline, candidate, "reward") == pytest.approx([-0.1])

    def test_no_data_returns_none_not_empty_list(self):
        assert paired_task_deltas(_run([_g(0, **{"mean/reward": 0.5})]), _run([_g(0)]), "reward") is None


class TestResolveMetrics:
    def test_explicit_request_is_returned_verbatim_deduped(self):
        resolved, skipped = resolve_metrics(_run([]), _run([]), ["reward", "reward", "output_tokens"])
        assert resolved == ["reward", "output_tokens"] and skipped == []

    def test_default_skips_non_mean_and_no_data_metrics(self):
        baseline = _run([_g(0, **{"mean/reward": 1.0})], key_metrics={"mean/reward": 1.0, "pass@1/acc": 1.0})
        candidate = _run([_g(0, **{"mean/reward": 0.5})], key_metrics={"mean/reward": 0.5, "pass@1/acc": 0.5})
        resolved, skipped = resolve_metrics(baseline, candidate, None)
        assert resolved == ["reward"] and skipped == ["pass@1/acc"]


class TestRunMetric:
    def test_no_common_task_returns_a_note_rather_than_raising(self):
        baseline, candidate = _run([_g(0, **{"mean/reward": 1.0})]), _run([_g(1, **{"mean/reward": 0.0})])
        result = run_metric(baseline, candidate, metric="reward", margin=None, alpha=0.05)
        assert result.n_pairs == 0 and result.p_value is None and "no per-task" in result.note

    def test_single_paired_task_cannot_estimate_se(self):
        baseline, candidate = _run([_g(0, **{"mean/reward": 1.0})]), _run([_g(0, **{"mean/reward": 0.7})])
        result = run_metric(baseline, candidate, metric="reward", margin=None, alpha=0.05)
        assert result.n_pairs == 1 and result.se is None and "cannot estimate" in result.note

    def test_zero_variance_nonzero_mean_is_significant(self):
        baseline = _run([_g(i, **{"mean/reward": 0.5}) for i in range(3)])
        candidate = _run([_g(i, **{"mean/reward": 0.0}) for i in range(3)])
        result = run_metric(baseline, candidate, metric="reward", margin=None, alpha=0.05)
        assert result.se == 0.0 and result.p_value == 0.0 and result.significant is True

    def test_p_value_matches_scipy_ttest_1samp_directly(self):
        deltas = [0.2, -0.1, 0.3, 0.05, -0.05, 0.15]
        baseline = _run([_g(i, **{"mean/reward": 0.0}) for i in range(len(deltas))])
        candidate = _run([_g(i, **{"mean/reward": d}) for i, d in enumerate(deltas)])
        result = run_metric(baseline, candidate, metric="reward", margin=None, alpha=0.05)
        _, expected_p = stats.ttest_1samp(deltas, popmean=0.0)
        assert result.p_value == pytest.approx(expected_p)

    def test_regression_within_margin_is_not_meaningfully_worse(self):
        deltas = [-0.05, -0.06, -0.04, -0.05, -0.05, -0.06]
        baseline = _run([_g(i, **{"mean/reward": 0.0}) for i in range(len(deltas))])
        candidate = _run([_g(i, **{"mean/reward": d}) for i, d in enumerate(deltas)])
        result = run_metric(baseline, candidate, metric="reward", margin=0.2, alpha=0.05)
        assert result.significant is True and result.p_value < 0.05


class TestBuildReport:
    def test_default_tests_every_key_metric_with_pairing_data_and_notes_the_rest(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path)), "gym eval stat-test ...")
        assert [result.metric for result in report.results] == ["reward"]
        assert report.notes == ["Skipped 1 key metric(s) with no per-task pairing data: pass@1/accuracy."]

    def test_an_explicitly_named_metric_with_no_pairing_data_raises(self, tmp_path):
        config = config_for(*two_runs(tmp_path), metric=["does_not_exist"])
        with pytest.raises(ConfigError, match="does_not_exist"):
            build_report(config, "gym eval stat-test ...")

    def test_no_key_metric_has_pairing_data_raises(self, tmp_path):
        baseline = _write_run(tmp_path, "run_a", [_entry(key_metrics={}, groups=[_group(0, [1.0])])])
        candidate = _write_run(tmp_path, "run_b", [_entry(key_metrics={}, groups=[_group(0, [0.0])])])
        with pytest.raises(ConfigError, match="No key metric has per-task pairing data"):
            build_report(config_for(baseline, candidate), "gym eval stat-test ...")


class TestReportRendering:
    def test_markdown_is_a_plain_line_per_metric(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        markdown = render_markdown(report)
        assert "gym eval stat-test: paired" in markdown
        assert "reward: n=6" in markdown

    def test_a_report_with_no_results_says_so(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        report.results = []
        assert "No metrics were tested." in render_markdown(report)

    def test_summary_reports_each_metric_and_every_path_written(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        text = "\n".join(summary(report, [tmp_path / "out.json"]))
        assert "reward: n=6" in text and str(tmp_path / "out.json") in text
