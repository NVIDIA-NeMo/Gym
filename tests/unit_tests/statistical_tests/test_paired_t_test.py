# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError
from scipy import stats

from nemo_gym.comparison.loading import LoadedRun
from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests.common import render_markdown, summary
from nemo_gym.statistical_tests.paired_t_test import (
    PairedTTestConfig,
    build_report,
    paired_task_deltas,
    resolve_metrics,
    run_metric,
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


def config_for(baseline, candidate, **overrides) -> PairedTTestConfig:
    return PairedTTestConfig.model_validate(
        {
            "baseline_rollouts_jsonl_fpath": str(baseline),
            "candidate_rollouts_jsonl_fpaths": [str(candidate)],
            **overrides,
        }
    )


class TestPairedTTestConfig:
    def test_valid_minimal_config(self):
        config = PairedTTestConfig.model_validate(BASE)
        assert config.test == "paired-t-test" and config.metric is None and config.margin is None

    def test_the_default_is_a_two_sided_test_with_no_margin(self):
        config = PairedTTestConfig.model_validate(BASE)
        assert config.alternative == "two-sided" and config.margin is None

    @pytest.mark.parametrize("margin", [[-0.01], [0.1, -5]])
    def test_a_negative_margin_is_rejected(self, margin):
        with pytest.raises(ValidationError, match="--margin must be non-negative"):
            PairedTTestConfig.model_validate({**BASE, "margin": margin, "metric": ["a", "b"]})

    @pytest.mark.parametrize("margin", [[float("nan")], [float("inf")], [float("-inf")], [0.1, float("nan")]])
    def test_a_non_finite_margin_is_rejected(self, margin):
        """NaN survives a bare `< 0` check (every NaN comparison is False) and would yield a NaN p-value."""
        with pytest.raises(ValidationError, match="--margin must be non-negative"):
            PairedTTestConfig.model_validate({**BASE, "margin": margin, "metric": ["a", "b"]})

    @pytest.mark.parametrize("margin", [[0.0], [0.0, 0.0], [0.01, 2.5]])
    def test_a_non_negative_margin_is_accepted(self, margin):
        """Zero is the default and stays legal: it makes a one-sided test the plain shifted-null test."""
        assert PairedTTestConfig.model_validate({**BASE, "margin": margin, "metric": ["a", "b"]}).margin == margin

    def test_one_margin_per_metric_is_accepted_but_a_mismatched_count_is_not(self):
        config = PairedTTestConfig.model_validate({**BASE, "metric": ["a", "b"], "margin": [0.01, 0.02]})
        assert config.margin == [0.01, 0.02]
        # A single value is always fine -- it applies to every metric.
        assert PairedTTestConfig.model_validate({**BASE, "metric": ["a", "b"], "margin": [0.01]}).margin == [0.01]
        with pytest.raises(ValidationError, match="one per --metric"):
            PairedTTestConfig.model_validate({**BASE, "metric": ["a", "b"], "margin": [0.01, 0.02, 0.03]})

    def test_a_duplicate_metric_cannot_hide_a_margin(self):
        """`--metric acc,acc --margin 0.1,0.2` used to pass and then silently drop 0.2: margins pair 1:1."""
        with pytest.raises(ValidationError, match="one per --metric"):
            PairedTTestConfig.model_validate({**BASE, "metric": ["acc", "acc"], "margin": [0.1, 0.2]})
        assert PairedTTestConfig.model_validate({**BASE, "metric": ["acc", "acc"]}).metric == ["acc"]

    def test_a_margin_list_without_a_metric_list_is_rejected(self):
        """Without --metric there is no order to line the margins up against."""
        with pytest.raises(ValidationError, match="one per --metric"):
            PairedTTestConfig.model_validate({**BASE, "margin": [0.01, 0.02]})

    def test_filename_parts_reflect_the_alternative_the_margins_and_the_metric_subset(self):
        assert PairedTTestConfig.model_validate(BASE).filename_parts() == ["two-sided"]
        assert PairedTTestConfig.model_validate({**BASE, "alternative": "candidate-lower"}).filename_parts() == [
            "candidate-lower"
        ]
        assert PairedTTestConfig.model_validate({**BASE, "margin": [0.01]}).filename_parts() == [
            "two-sided",
            "margin-0.01",
        ]
        assert PairedTTestConfig.model_validate(
            {**BASE, "metric": ["reward", "a/b"], "margin": [0.01, 0.02]}
        ).filename_parts() == ["metric-reward+a-b", "two-sided", "margin-0.01+0.02"]


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
        result = run_metric(baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="two-sided")
        assert result.n_pairs == 0 and result.p_value is None and "no per-task" in result.note

    def test_single_paired_task_cannot_estimate_se(self):
        baseline, candidate = _run([_g(0, **{"mean/reward": 1.0})]), _run([_g(0, **{"mean/reward": 0.7})])
        result = run_metric(baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="two-sided")
        assert result.n_pairs == 1 and result.se is None and "cannot estimate" in result.note

    def test_zero_variance_nonzero_mean_is_significant(self):
        baseline = _run([_g(i, **{"mean/reward": 0.5}) for i in range(3)])
        candidate = _run([_g(i, **{"mean/reward": 0.0}) for i in range(3)])
        result = run_metric(baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="two-sided")
        assert result.se == 0.0 and result.p_value == 0.0 and result.significant is True

    def test_zero_variance_one_sided_answers_the_directional_question_against_the_margin(self):
        """The shortcut still has to respect both the direction and the margin, not just `mean_diff != 0`."""
        baseline, candidate = self._runs([-0.5] * 4)
        kwargs = dict(metric="reward", margin=0.2, alpha=0.05)
        assert run_metric(baseline, candidate, **kwargs, alternative="candidate-lower").significant is True
        assert run_metric(baseline, candidate, **kwargs, alternative="candidate-higher").significant is False
        # A consistent drop smaller than the margin is not the change the margin asked about.
        smaller_than_the_margin = self._runs([-0.1] * 4)
        assert run_metric(*smaller_than_the_margin, **kwargs, alternative="candidate-lower").significant is False

    def test_zero_variance_treats_float64_rounding_noise_as_no_change(self):
        """`0.1 + 0.2 != 0.3` in float64: a constant ~1e-17 offset is noise, not a difference with p=0."""
        baseline = _run([_g(i, **{"mean/reward": 0.1 + 0.2}) for i in range(4)])
        candidate = _run([_g(i, **{"mean/reward": 0.3}) for i in range(4)])
        kwargs = dict(metric="reward", margin=0.0, alpha=0.05)
        assert run_metric(baseline, candidate, **kwargs, alternative="two-sided").significant is False
        assert run_metric(baseline, candidate, **kwargs, alternative="candidate-lower").significant is False

    def test_a_thin_pairing_is_reported_without_a_verdict(self):
        """One task in common out of ten a side: report the overlap, do not conclude from it."""
        baseline = _run([_g(i, **{"mean/reward": 0.4}) for i in range(10)])
        candidate = _run([_g(i + 9, **{"mean/reward": 0.7}) for i in range(10)])
        result = run_metric(baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="two-sided")
        assert result.n_pairs == 1 and result.p_value is None and result.significant is None
        assert "1 of 10 tasks paired" in result.note

    def test_a_pairing_at_the_coverage_floor_is_still_tested(self):
        baseline = _run([_g(i, **{"mean/reward": 0.4}) for i in range(6)])
        candidate = _run([_g(i, **{"mean/reward": 0.7 + i * 0.01}) for i in range(3)])
        result = run_metric(baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="two-sided")
        assert result.n_pairs == 3 and result.p_value is not None

    def test_p_value_matches_scipy_ttest_1samp_directly(self):
        deltas = [0.2, -0.1, 0.3, 0.05, -0.05, 0.15]
        baseline = _run([_g(i, **{"mean/reward": 0.0}) for i in range(len(deltas))])
        candidate = _run([_g(i, **{"mean/reward": d}) for i, d in enumerate(deltas)])
        result = run_metric(baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="two-sided")
        _, expected_p = stats.ttest_1samp(deltas, popmean=0.0)
        assert result.p_value == pytest.approx(expected_p)

    def _runs(self, deltas):
        baseline = _run([_g(i, **{"mean/reward": 0.0}) for i in range(len(deltas))])
        return baseline, _run([_g(i, **{"mean/reward": d}) for i, d in enumerate(deltas)])

    def test_mde_matches_the_power_formula_and_widens_for_a_two_sided_test(self):
        """(t_alpha + t_power) * se, derived here from scipy rather than from the implementation."""
        from nemo_gym.statistical_tests.paired_t_test import MDE_POWER

        deltas = [0.2, -0.1, 0.3, 0.05, -0.05, 0.15]
        baseline, candidate = self._runs(deltas)
        kwargs = dict(metric="reward", margin=0.0, alpha=0.05)

        two_sided = run_metric(baseline, candidate, **kwargs, alternative="two-sided")
        one_sided = run_metric(baseline, candidate, **kwargs, alternative="candidate-higher")

        df = len(deltas) - 1
        expected = (stats.t.ppf(1 - 0.05 / 2, df) + stats.t.ppf(MDE_POWER, df)) * two_sided.se
        assert two_sided.minimum_detectable_effect == pytest.approx(expected)
        # A one-sided test spends all of alpha in one tail, so it can detect a smaller effect.
        assert one_sided.minimum_detectable_effect < two_sided.minimum_detectable_effect

    def test_mde_scales_with_the_sample(self):
        """The reason to report it: more tasks, smaller detectable effect."""
        small = run_metric(
            *self._runs([0.2, -0.1, 0.3, 0.05]), metric="reward", margin=0.0, alpha=0.05, alternative="two-sided"
        )
        large = run_metric(
            *self._runs([0.2, -0.1, 0.3, 0.05] * 6), metric="reward", margin=0.0, alpha=0.05, alternative="two-sided"
        )
        assert large.minimum_detectable_effect < small.minimum_detectable_effect

    def test_no_mde_without_a_standard_error_to_scale_it(self):
        """n<2 and zero-variance both leave it unset rather than reporting a meaningless 0."""
        one_task = run_metric(
            _run([_g(0, **{"mean/reward": 1.0})]),
            _run([_g(0, **{"mean/reward": 0.7})]),
            metric="reward",
            margin=0.0,
            alpha=0.05,
            alternative="two-sided",
        )
        degenerate = run_metric(
            _run([_g(i, **{"mean/reward": 0.5}) for i in range(3)]),
            _run([_g(i, **{"mean/reward": 0.0}) for i in range(3)]),
            metric="reward",
            margin=0.0,
            alpha=0.05,
            alternative="two-sided",
        )
        assert one_task.minimum_detectable_effect is None
        assert degenerate.se == 0.0 and degenerate.minimum_detectable_effect is None

    def test_a_drop_smaller_than_the_margin_is_not_detected(self):
        """A 5pp drop is real but under the 20pp bar, so it does not count as a meaningful change."""
        baseline, candidate = self._runs([-0.05, -0.06, -0.04, -0.05, -0.05, -0.06])
        result = run_metric(
            baseline, candidate, metric="reward", margin=0.2, alpha=0.05, alternative="candidate-lower"
        )
        assert result.significant is False

    def test_a_consistent_drop_is_detected_once_the_margin_stops_hiding_it(self):
        baseline, candidate = self._runs([-0.05, -0.06, -0.04, -0.05, -0.05, -0.06])
        result = run_metric(
            baseline, candidate, metric="reward", margin=0.0, alpha=0.05, alternative="candidate-lower"
        )
        assert result.significant is True and result.p_value < 0.05

    def test_candidate_lower_mirrors_candidate_higher(self):
        """Same data, opposite question: an improvement is detected by `higher` and never by `lower`."""
        baseline, candidate = self._runs([0.05, 0.06, 0.04, 0.05, 0.05, 0.06])
        kwargs = dict(metric="reward", margin=0.0, alpha=0.05)
        assert run_metric(baseline, candidate, **kwargs, alternative="candidate-higher").significant is True
        assert run_metric(baseline, candidate, **kwargs, alternative="candidate-lower").significant is False

    def test_one_sided_p_values_match_scipy_at_a_shifted_null(self):
        """Pins the null LOCATION and the TAIL, so a future sign flip cannot pass silently."""
        deltas = [0.2, -0.1, 0.3, 0.05, -0.05, 0.15]
        baseline, candidate = self._runs(deltas)
        kwargs = dict(metric="reward", margin=0.1, alpha=0.05)
        higher = run_metric(baseline, candidate, **kwargs, alternative="candidate-higher")
        lower = run_metric(baseline, candidate, **kwargs, alternative="candidate-lower")
        assert higher.p_value == pytest.approx(stats.ttest_1samp(deltas, popmean=0.1, alternative="greater").pvalue)
        assert lower.p_value == pytest.approx(stats.ttest_1samp(deltas, popmean=-0.1, alternative="less").pvalue)

    def test_alpha_decides_significance_rather_than_a_hardcoded_level(self):
        baseline, candidate = self._runs([0.2, -0.1, 0.3, 0.05, -0.05, 0.15])
        kwargs = dict(metric="reward", margin=0.0, alternative="two-sided")
        result = run_metric(baseline, candidate, **kwargs, alpha=0.05)
        assert result.significant is False and result.p_value > 0.05
        assert run_metric(baseline, candidate, **kwargs, alpha=0.5).significant is True


class TestBuildReport:
    def test_default_tests_every_key_metric_with_pairing_data_and_notes_the_rest(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path)), "gym eval stat-test ...")
        assert [result.metric for result in report.results] == ["reward"]
        assert report.notes == ["Skipped 1 key metric(s) with no per-task pairing data: pass@1/accuracy."]

    def test_each_margin_is_applied_to_its_own_metric(self, tmp_path):
        """`--metric reward,latency --margin 0.01,0.5`: swapping the pairing flips both verdicts."""
        baseline = _write_run(
            tmp_path, "run_a", [_entry(groups=[_group(i, [0.5], extra={"mean/latency": 10.0}) for i in range(6)])]
        )
        candidate = _write_run(
            tmp_path,
            "run_b",
            [_entry(groups=[_group(i, [0.6 + i * 0.01], extra={"mean/latency": 10.02 + i * 0.01}) for i in range(6)])],
        )
        config = config_for(
            baseline, candidate, metric=["reward", "latency"], margin=[0.01, 0.5], alternative="candidate-higher"
        )

        results = build_report(config, "gym eval stat-test ...").results

        # reward rose ~0.125 (well past its 0.01 margin); latency rose ~0.045 (nowhere near its 0.5 margin).
        assert [(r.metric, r.margin) for r in results] == [("reward", 0.01), ("latency", 0.5)]
        assert [r.significant for r in results] == [True, False]

    def test_unpaired_tasks_raise_a_warning_on_the_report(self, tmp_path):
        baseline = _write_run(tmp_path, "run_a", [_entry(groups=[_group(i, [1.0, 0.0]) for i in range(4)])])
        candidate = _write_run(tmp_path, "run_b", [_entry(groups=[_group(i, [1.0, 1.0]) for i in range(3)])])
        report = build_report(config_for(baseline, candidate, metric=["reward"]), "gym eval stat-test ...")
        assert report.results[0].n_pairs == 3
        assert report.warnings == ["Not every task paired across the two runs: reward (3 of 4)."]

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
    def test_markdown_leads_with_the_findings_and_ends_with_how_to_reproduce_them(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        markdown = render_markdown(report)

        assert markdown.startswith("# Statistical tests results")
        assert "- reward (paired t-test): n=6" in markdown
        # Results come first; the inputs and the command that produced them sit underneath.
        assert markdown.index("- reward (paired t-test)") < markdown.index("## Input data")
        assert markdown.index("## Input data") < markdown.index("## Reproduction")
        assert "- Command: `gym eval stat-test ...`" in markdown

    def test_markdown_keeps_one_item_per_line(self, tmp_path):
        """Bulleted, so markdown does not reflow the run pair into a single paragraph."""
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        report.notes = ["skipped something"]
        markdown = render_markdown(report)
        assert "- Note: skipped something" in markdown
        assert "- Baseline:  " in markdown and "- Candidate: " in markdown

    def test_a_report_with_no_results_says_so(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        report.results = []
        assert "No metrics were tested." in render_markdown(report)

    def test_summary_reports_each_metric_and_every_path_written(self, tmp_path):
        report = build_report(config_for(*two_runs(tmp_path), metric=["reward"]), "gym eval stat-test ...")
        text = "\n".join(summary(report, [tmp_path / "out.json"]))
        assert "reward (paired t-test): n=6" in text and str(tmp_path / "out.json") in text
