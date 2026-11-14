# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise direct planning inputs, actual CLI results, and inspectable reports."""

import json
import math
from pathlib import Path

import pytest
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser
from pydantic import ValidationError
from scipy.stats import binomtest

from nemo_gym.cli.main import build_parser, main
from nemo_gym.config.global_config import reset_global_config_dict
from nemo_gym.eval.comparison.planning.schema import RepeatPlanningManifest, RepeatPlanningResult
from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import (
    BenchmarkSamples,
    MetricScale,
    aggregate_welch,
    normalize_and_average_metrics_per_repeat,
    summarize_sample,
)


def _manifest(**overrides: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "aggregation_level": "benchmark",
        "validation_trials": 0,
        "cells": [{"model": "model-a", "benchmark": "gpqa", "baseline_std_points": 2.0}],
        **overrides,
    }


def _write_plan(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _run_cli(plan: Path, output: Path, monkeypatch, *extra: str) -> RepeatPlanningResult:
    reset_global_config_dict()
    monkeypatch.setattr(
        "sys.argv",
        ["gym", "eval", "plan-repeats", "--plan", str(plan), "--output-dir", str(output), *extra],
    )
    main()
    return RepeatPlanningResult.model_validate_json((output / "repeat_planning_report.json").read_bytes())


def test_planning_flags_preserve_paths_in_hydra_translation() -> None:
    args, unknown = build_parser().parse_known_args(
        [
            "eval",
            "plan-repeats",
            "--plan",
            "results/[v2] plan.yaml",
            "--output-dir",
            "reports with spaces",
            "--report-format",
            "json",
        ]
    )
    assert unknown == []
    tokens = [token for flag in args._command.flags for token in flag.translate_to_hydra(args)]
    parsed = OverridesParser.create().parse_overrides(tokens)
    assert {override.key_or_group: override.value() for override in parsed} == {
        "repeat_plan_fpath": "results/[v2] plan.yaml",
        "output_dirpath": "reports with spaces",
        "report_format": "json",
    }


def test_manifest_defaults_preserve_one_fixed_aggregate() -> None:
    payload = _manifest()
    del payload["validation_trials"]
    manifest = RepeatPlanningManifest.model_validate(payload)
    assert manifest.baseline.mode == "optimize"
    assert manifest.candidate_experiments == 1
    assert (manifest.margin_points, manifest.target_power, manifest.alpha, manifest.expected_delta_points) == (
        1.0,
        0.8,
        0.05,
        0.0,
    )
    assert manifest.max_repeats == 20000
    assert manifest.seed == 20261109
    assert manifest.validation_trials == 10000
    assert manifest.cells[0].repeat_step == 1
    assert manifest.cells[0].repeat_cost == 1
    assert manifest.cells[0].candidate_std_points is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema_version": True}, "integer 1"),
        ({"schema_version": 2}, "Input should be 1"),
        ({"aggregation_level": "per_benchmark"}, "benchmark"),
        ({"cells": []}, "at least 1 item"),
        ({"candidate_experiments": True}, "valid integer"),
        ({"candidate_experiments": 0}, "greater than or equal to 1"),
        ({"validation_trials": -1}, "greater than or equal to 0"),
        ({"margin_points": 0}, "greater than 0"),
        ({"target_power": 1}, "less than 1"),
        ({"alpha": 0.5}, "less than 0.5"),
        ({"source": "run.json"}, "Extra inputs are not permitted"),
        ({"baseline": {"mode": "balanced"}}, "optimize"),
        ({"baseline": {"mode": "existing"}}, "requires existing_baseline_repeats"),
    ],
)
def test_manifest_rejects_invalid_or_unsupported_controls(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RepeatPlanningManifest.model_validate(_manifest(**overrides))


@pytest.mark.parametrize(
    "cell_changes,message",
    [
        ({"model": " "}, "at least 1 character"),
        ({"baseline_std_points": -1}, "greater than or equal to 0"),
        ({"baseline_std_points": math.inf}, "finite number"),
        ({"candidate_std_points": True}, "valid number"),
        ({"repeat_step": 1.5}, "valid integer"),
        ({"repeat_cost": 0}, "greater than 0"),
        ({"existing_baseline_repeats": 10}, "only allowed"),
        ({"metrics": []}, "at least 1 item"),
        ({"metrics": [{"name": "accuracy", "baseline_std_points": 1}]}, "either scalar"),
    ],
)
def test_manifest_rejects_invalid_cell_inputs(cell_changes: dict[str, object], message: str) -> None:
    payload = _manifest(cells=[{"model": "m", "benchmark": "b", "baseline_std_points": 2, **cell_changes}])
    with pytest.raises(ValidationError, match=message):
        RepeatPlanningManifest.model_validate(payload)


def test_manifest_enforces_cell_identity_scope_and_shared_metric_order() -> None:
    first = {"model": "a", "benchmark": "omni", "baseline_std_points": 1}
    with pytest.raises(ValidationError, match="unique"):
        RepeatPlanningManifest.model_validate(_manifest(aggregation_level="suite", cells=[first, first]))
    second = {**first, "model": "b"}
    with pytest.raises(ValidationError, match="exactly one"):
        RepeatPlanningManifest.model_validate(_manifest(cells=[first, second]))
    with pytest.raises(ValidationError, match="same model"):
        RepeatPlanningManifest.model_validate(_manifest(aggregation_level="model", cells=[first, second]))
    metrics = [{"name": name, "baseline_std_points": 1} for name in ("accuracy", "non-hallucination")]
    with pytest.raises(ValidationError, match="same metric names and order"):
        RepeatPlanningManifest.model_validate(
            _manifest(
                aggregation_level="suite",
                cells=[
                    {"model": "a", "benchmark": "omni", "baseline_std_points": 1, "metrics": metrics},
                    {"model": "b", "benchmark": "omni", "baseline_std_points": 1, "metrics": list(reversed(metrics))},
                ],
            )
        )
    with pytest.raises(ValidationError, match="Metric names must be unique"):
        RepeatPlanningManifest.model_validate(
            _manifest(cells=[{"model": "a", "benchmark": "omni", "metrics": [metrics[0], metrics[0]]}])
        )


def test_real_cli_plans_metric_composites_and_preserves_full_precision(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = _write_plan(
        tmp_path / "plan [v1].yaml",
        _manifest(
            candidate_experiments=3,
            cells=[
                {
                    "model": "model-a",
                    "benchmark": "omniscience",
                    "repeat_step": 4,
                    "repeat_cost": 2.5,
                    "baseline_std_points": math.sqrt(5),
                    "candidate_std_points": 3.0,
                    "metrics": [
                        {"name": "accuracy", "baseline_std_points": 2.0, "candidate_std_points": 3.0},
                        {"name": "non-hallucination", "baseline_std_points": 4.0},
                    ],
                }
            ],
        ),
    )
    original = plan.read_bytes()
    output = tmp_path / "reports"
    output.mkdir()
    (output / "compare_report.md").write_text("existing comparison")
    result = _run_cli(plan, output, monkeypatch, "++policy_api_key=sk-private-example")
    cell = result.cells[0]
    assert cell.weight == 1
    assert [metric.weight for metric in cell.metrics] == [0.5, 0.5]
    assert [metric.candidate_std_defaulted for metric in cell.metrics] == [False, True]
    assert cell.baseline_std_points == pytest.approx(math.sqrt(5))
    assert cell.candidate_std_points == 3.0
    assert cell.baseline_repeats % 4 == cell.candidate_repeats_per_experiment % 4 == 0
    assert result.allocation.statistics.variance == pytest.approx(
        5 / cell.baseline_repeats + 9 / cell.candidate_repeats_per_experiment
    )
    assert result.allocation.statistics.power >= 0.8
    assert result.totals.total_additional_passes == cell.baseline_repeats + 3 * cell.candidate_repeats_per_experiment
    assert result.totals.total_cost == 2.5 * result.totals.total_additional_passes
    assert cell.new_repeat_cost == result.totals.total_cost
    assert result.simulation is None
    assert plan.read_bytes() == original
    assert (output / "compare_report.md").read_text() == "existing comparison"
    payload = json.loads((output / "repeat_planning_report.json").read_text())
    assert payload["cells"][0]["baseline_std_points"] != 2.236
    markdown = (output / "repeat_planning_report.md").read_text()
    assert "| model-a | omniscience | 1.000 | 2.236 | 3.000 |" in markdown
    assert "Approximate planning power" in markdown
    assert "Simulation validation is disabled" in markdown
    assert "Omitted candidate SDs default" in markdown
    assert "sk-private-example" not in markdown
    assert "sk-private-example" not in result.command
    assert "++policy_api_key=****" in result.command
    assert "Total new passes:" in capsys.readouterr().out


def test_real_cli_reuses_existing_baseline_and_counts_candidate_experiments(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(
        tmp_path / "existing.yaml",
        _manifest(
            baseline={"mode": "existing"},
            candidate_experiments=4,
            cells=[
                {
                    "model": "a",
                    "benchmark": "b",
                    "baseline_std_points": 1.0,
                    "candidate_std_points": 2.0,
                    "existing_baseline_repeats": 100,
                    "repeat_cost": 7.0,
                }
            ],
        ),
    )
    result = _run_cli(plan, tmp_path / "reports", monkeypatch, "--report-format", "json")
    cell = result.cells[0]
    assert cell.baseline_repeats == cell.existing_baseline_repeats == 100
    assert cell.new_baseline_repeats == result.totals.new_baseline_passes == 0
    assert result.totals.existing_baseline_passes == 100
    assert result.totals.candidate_passes_per_experiment == cell.candidate_repeats_per_experiment
    assert result.totals.total_additional_passes == 4 * cell.candidate_repeats_per_experiment
    assert result.totals.total_cost == 7 * result.totals.total_additional_passes
    assert not (tmp_path / "reports" / "repeat_planning_report.md").exists()


def test_cli_metric_composites_match_gym_on_orthogonal_repeat_observations(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(
        tmp_path / "orthogonal.yaml",
        _manifest(
            cells=[
                {
                    "model": "a",
                    "benchmark": "omniscience",
                    "baseline_std_points": math.sqrt(5),
                    "candidate_std_points": 2.5,
                    "metrics": [
                        {"name": "accuracy", "baseline_std_points": 2.0, "candidate_std_points": 3.0},
                        {"name": "non-hallucination", "baseline_std_points": 4.0},
                    ],
                }
            ]
        ),
    )
    result = _run_cli(plan, tmp_path / "reports", monkeypatch)
    cell = result.cells[0]

    def observations(count: int, first_std: float, second_std: float) -> list[list[float]]:
        # Centered vectors (a,-a,0,...) and (b,b,-2b,0,...) have zero dot
        # product. Their exact sample SDs are the supplied normalized SDs.
        first = first_std / 100 * math.sqrt((count - 1) / 2)
        second = second_std / 100 * math.sqrt((count - 1) / 6)
        rows = [[0.5, 0.5] for _ in range(count)]
        rows[0][0] += first
        rows[1][0] -= first
        rows[0][1] += second
        rows[1][1] += second
        rows[2][1] -= 2 * second
        return rows

    scales = [MetricScale(0, 1), MetricScale(0, 1)]
    baseline = summarize_sample(
        normalize_and_average_metrics_per_repeat(observations(cell.baseline_repeats, 2, 4), scales)
    )
    candidate = summarize_sample(
        normalize_and_average_metrics_per_repeat(observations(cell.candidate_repeats_per_experiment, 3, 4), scales)
    )
    assert baseline.sample_variance == pytest.approx(5.0)
    assert candidate.sample_variance == pytest.approx(6.25)
    gym = aggregate_welch([BenchmarkSamples(baseline, candidate)], confidence_level=0.9)
    assert gym.variance == pytest.approx(5 / baseline.count + 6.25 / candidate.count)
    assert gym.delta == pytest.approx(0, abs=1e-12)
    assert result.allocation.statistics.variance == pytest.approx(gym.variance)
    assert result.allocation.statistics.standard_error == pytest.approx(gym.standard_error)
    assert result.allocation.statistics.degrees_of_freedom == pytest.approx(gym.degrees_of_freedom)
    assert result.allocation.statistics.half_width == pytest.approx(gym.ci_high)


@pytest.mark.parametrize("level", ["model", "suite"])
def test_real_cli_keeps_benchmark_weights_equal_with_unequal_metric_counts(
    tmp_path: Path, monkeypatch, level: str
) -> None:
    plan = _write_plan(
        tmp_path / "suite.yaml",
        _manifest(
            aggregation_level=level,
            cells=[
                {"model": "a", "benchmark": "gpqa", "baseline_std_points": 2.0},
                {
                    "model": "a",
                    "benchmark": "omniscience",
                    "repeat_cost": 3.0,
                    "baseline_std_points": math.sqrt(2),
                    "metrics": [
                        {"name": "accuracy", "baseline_std_points": 2.0},
                        {"name": "non-hallucination", "baseline_std_points": 2.0},
                    ],
                },
            ],
        ),
    )
    result = _run_cli(plan, tmp_path / "reports", monkeypatch)
    first, second = result.cells
    assert [cell.weight for cell in result.cells] == [0.5, 0.5]
    assert [metric.weight for metric in first.metrics] == [1.0]
    assert [metric.weight for metric in second.metrics] == [0.5, 0.5]
    expected_variance = 0.25 * (
        4 / first.baseline_repeats
        + 4 / first.candidate_repeats_per_experiment
        + 2 / second.baseline_repeats
        + 2 / second.candidate_repeats_per_experiment
    )
    assert result.allocation.statistics.variance == pytest.approx(expected_variance)
    assert result.totals.total_cost == (
        first.new_baseline_repeats
        + first.candidate_repeats_per_experiment
        + 3 * (second.new_baseline_repeats + second.candidate_repeats_per_experiment)
    )
    assert result.settings.aggregation_level == level
    assert result.allocation.statistics.power >= result.settings.target_power


def test_cli_rejects_multiple_metric_sds_without_the_composite_sd(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = _write_plan(
        tmp_path / "missing-composite.yaml",
        _manifest(
            cells=[
                {
                    "model": "a",
                    "benchmark": "omniscience",
                    "metrics": [
                        {"name": "accuracy", "baseline_std_points": 2.0},
                        {"name": "non-hallucination", "baseline_std_points": 2.0},
                    ],
                }
            ]
        ),
    )
    output = tmp_path / "reports"
    with pytest.raises(SystemExit) as error:
        _run_cli(plan, output, monkeypatch)
    assert error.value.code == 1
    message = " ".join(capsys.readouterr().out.split())
    assert "Multiple metrics require cell-level baseline_std_points" in message
    assert "averaging the selected metrics within each repeat" in message
    assert not output.exists()


@pytest.mark.parametrize("second_sign, composite_variance", [(1, 4.0), (-1, 0.0)])
def test_cli_measured_composite_sds_match_gym_for_related_metric_scores(
    tmp_path: Path, monkeypatch, second_sign: int, composite_variance: float
) -> None:
    scales = [MetricScale(0, 100), MetricScale(0, 100)]
    historical = [[48.0, 50.0 - 2 * second_sign], [50.0, 50.0], [52.0, 50.0 + 2 * second_sign]]
    composite = summarize_sample(normalize_and_average_metrics_per_repeat(historical, scales))
    assert composite.sample_variance == composite_variance
    plan = _write_plan(
        tmp_path / "composite.yaml",
        _manifest(
            aggregation_level="suite",
            cells=[
                {"model": "a", "benchmark": "gpqa", "baseline_std_points": 1.0},
                {
                    "model": "a",
                    "benchmark": "omniscience",
                    "baseline_std_points": math.sqrt(composite.sample_variance),
                    "metrics": [
                        {"name": "accuracy", "baseline_std_points": 2.0},
                        {"name": "non-hallucination", "baseline_std_points": 2.0},
                    ],
                },
            ],
        ),
    )
    result = _run_cli(plan, tmp_path / "reports", monkeypatch)
    control, measured = result.cells
    assert measured.baseline_std_points**2 == measured.candidate_std_points**2 == composite_variance
    assert [cell.weight for cell in result.cells] == [0.5, 0.5]
    assert [metric.weight for metric in measured.metrics] == [0.5, 0.5]

    def observations(count: int, std: float, sign: int) -> list[list[float]]:
        distance = std * math.sqrt((count - 1) / 2)
        rows = [[50.0, 50.0] for _ in range(count)]
        rows[0] = [50 - distance, 50 - sign * distance]
        rows[1] = [50 + distance, 50 + sign * distance]
        return rows

    samples = []
    for cell, std, sign in ((control, 1.0, 1), (measured, 2.0, second_sign)):
        baseline = summarize_sample(
            normalize_and_average_metrics_per_repeat(observations(cell.baseline_repeats, std, sign), scales)
        )
        candidate = summarize_sample(
            normalize_and_average_metrics_per_repeat(
                observations(cell.candidate_repeats_per_experiment, std, sign), scales
            )
        )
        samples.append(BenchmarkSamples(baseline, candidate))
    gym = aggregate_welch(samples, confidence_level=0.9)
    statistics = result.allocation.statistics
    assert statistics.variance == pytest.approx(gym.variance)
    assert statistics.standard_error == pytest.approx(gym.standard_error)
    assert statistics.degrees_of_freedom == pytest.approx(gym.degrees_of_freedom)
    assert statistics.half_width == pytest.approx(gym.ci_high)


def test_cli_rejects_an_entirely_constant_metric_composite(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = _write_plan(
        tmp_path / "constant.yaml",
        _manifest(
            cells=[
                {
                    "model": "a",
                    "benchmark": "omniscience",
                    "baseline_std_points": 0.0,
                    "metrics": [
                        {"name": "accuracy", "baseline_std_points": 2.0},
                        {"name": "non-hallucination", "baseline_std_points": 2.0},
                    ],
                }
            ]
        ),
    )
    output = tmp_path / "reports"
    with pytest.raises(SystemExit) as error:
        _run_cli(plan, output, monkeypatch)
    assert error.value.code == 1
    assert "all one-pass variances are zero" in " ".join(capsys.readouterr().out.split())
    assert not output.exists()


def test_cli_preserves_single_metric_sd_inputs(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(
        tmp_path / "single-metric.yaml",
        _manifest(
            cells=[
                {
                    "model": "a",
                    "benchmark": "gpqa",
                    "metrics": [{"name": "accuracy", "baseline_std_points": 2.0, "candidate_std_points": 3.0}],
                }
            ]
        ),
    )
    result = _run_cli(plan, tmp_path / "reports", monkeypatch)
    cell = result.cells[0]
    assert cell.baseline_std_points == 2.0
    assert cell.candidate_std_points == 3.0
    assert result.allocation.statistics.variance == pytest.approx(
        4 / cell.baseline_repeats + 9 / cell.candidate_repeats_per_experiment
    )


def test_cli_plans_from_composite_sds_and_keeps_metric_sds_descriptive(tmp_path: Path, monkeypatch) -> None:
    cell = {
        "model": "a",
        "benchmark": "omniscience",
        "baseline_std_points": 1.0,
        "metrics": [
            {"name": "accuracy", "baseline_std_points": 2.0, "candidate_std_points": 3.0},
            {"name": "non-hallucination", "baseline_std_points": 2.0, "candidate_std_points": 3.0},
        ],
    }
    plan = _write_plan(tmp_path / "composite.yaml", _manifest(cells=[cell]))
    first = _run_cli(plan, tmp_path / "first", monkeypatch)
    assert first.cells[0].baseline_std_points == first.cells[0].candidate_std_points == 1.0
    assert [metric.candidate_std_points for metric in first.cells[0].metrics] == [3.0, 3.0]
    assert "Omitted candidate SDs default" in (tmp_path / "first" / "repeat_planning_report.md").read_text()
    for metric in cell["metrics"]:
        metric["baseline_std_points"] = 4.0
        metric["candidate_std_points"] = 5.0
    _write_plan(plan, _manifest(cells=[cell]))
    second = _run_cli(plan, tmp_path / "second", monkeypatch)
    assert second.allocation == first.allocation
    assert [metric.baseline_std_points for metric in second.cells[0].metrics] == [4.0, 4.0]
    assert [metric.candidate_std_points for metric in second.cells[0].metrics] == [5.0, 5.0]


@pytest.mark.parametrize(
    "name,baseline,candidate,power",
    [
        ("benchmark", (15,), (12,), 0.8057546329230019),
        ("suite", (16, 32), (96, 160), 0.8015822293047277),
        ("multiple-metrics", (8, 10), (8, 10), 0.9186697623451402),
    ],
)
def test_checked_in_examples_reproduce_documented_allocations(
    tmp_path: Path, monkeypatch, name: str, baseline: tuple[int, ...], candidate: tuple[int, ...], power: float
) -> None:
    plan = Path(__file__).resolve().parents[3] / "examples" / "repeat-planning" / f"{name}.yaml"
    original = plan.read_bytes()
    result = _run_cli(plan, tmp_path / "reports", monkeypatch)
    assert result.allocation.baseline_counts == baseline
    assert result.allocation.candidate_counts == candidate
    assert result.allocation.statistics.power == pytest.approx(power, rel=1e-12, abs=1e-12)
    assert result.simulation is not None
    assert result.simulation.trials == 10000
    if name == "benchmark":
        rate = result.simulation.power
        assert rate.successes == 8066
        assert (round(rate.ci_low, 4), round(rate.ci_high, 4)) == (0.7987, 0.8142)
    assert plan.read_bytes() == original


@pytest.mark.parametrize("trials", [1, 100])
def test_real_cli_validates_fixed_allocation_with_reproducible_simulation(
    tmp_path: Path, monkeypatch, trials: int
) -> None:
    plan = _write_plan(tmp_path / "simulation.yaml", _manifest(validation_trials=trials, seed=87))
    first = _run_cli(plan, tmp_path / "first", monkeypatch)
    second = _run_cli(plan, tmp_path / "second", monkeypatch)
    assert first.allocation == second.allocation
    assert first.simulation == second.simulation
    simulation = first.simulation
    assert simulation is not None
    assert simulation.trials == trials
    assert simulation.seed == 87
    for rate in (simulation.power, simulation.false_positive, simulation.coverage):
        assert rate.trials == trials
        assert rate.estimate == rate.successes / rate.trials
        assert 0 <= rate.ci_low <= rate.estimate <= rate.ci_high <= 1
        assert rate.ci_high > rate.ci_low
        expected = binomtest(rate.successes, rate.trials).proportion_ci(method="wilson")
        assert (rate.ci_low, rate.ci_high) == pytest.approx((expected.low, expected.high))
    assert simulation.false_positive.successes + simulation.coverage.successes == trials
    markdown = (tmp_path / "first" / "repeat_planning_report.md").read_text()
    assert "| One-sided lower-bound coverage |" in markdown
    assert "Wilson 95% CI" in markdown
    assert "it can differ from the planning approximation" in markdown


@pytest.mark.parametrize("argument", [None, "+help=false"])
def test_real_cli_requires_a_plan(argument: str | None, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["gym", "eval", "plan-repeats", *([argument] if argument else [])])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "requires --plan PATH" in " ".join(capsys.readouterr().out.split())


def test_configuration_help_does_not_require_a_plan(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["gym", "eval", "plan-repeats", "+help=true"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code in (0, None)
    output = " ".join(capsys.readouterr().out.split())
    assert "RepeatPlanningCLIConfig" in output
    assert "requires --plan" not in output


@pytest.mark.parametrize("mode", ["invalid-yaml", "invalid-design"])
def test_real_cli_reports_actionable_errors_before_writing_outputs(
    tmp_path: Path, monkeypatch, capsys, mode: str
) -> None:
    plan = tmp_path / "bad.yaml"
    if mode == "invalid-yaml":
        plan.write_text("cells: [")
    else:
        _write_plan(plan, _manifest(expected_delta_points=-1.0))
    output = tmp_path / "reports"
    with pytest.raises(SystemExit) as error:
        _run_cli(plan, output, monkeypatch)
    assert error.value.code == 1
    message = " ".join(capsys.readouterr().out.split())
    assert str(plan) in "".join(message.split())
    assert "Cannot load repeat planning manifest" in message or "negative non-inferiority margin" in message
    assert not output.exists()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_report_writer_protects_input_plan_before_writing_either_report(
    tmp_path: Path, monkeypatch, capsys, alias: str
) -> None:
    output = tmp_path / "reports"
    output.mkdir()
    collision = output / "repeat_planning_report.json"
    plan = collision if alias == "direct" else tmp_path / "input.yaml"
    _write_plan(plan, _manifest())
    if alias == "symlink":
        collision.symlink_to(plan)
    elif alias == "hardlink":
        collision.hardlink_to(plan)
    original = plan.read_bytes()
    with pytest.raises(SystemExit) as error:
        _run_cli(plan, output, monkeypatch)
    assert error.value.code == 1
    assert "would overwrite the input plan" in " ".join(capsys.readouterr().out.split())
    assert plan.read_bytes() == collision.read_bytes() == original
    assert not (output / "repeat_planning_report.md").exists()
