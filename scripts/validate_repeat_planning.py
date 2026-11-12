# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently audit repeat planning against saved numeric reference artifacts.

Example::

    python scripts/validate_repeat_planning.py --reference-root /path/to/poc \
        --output-dir /path/to/audit --suite-report /path/to/suite_compare_report.json

The reference checkout supplies data, never imported executable code. All
statistical checks use an independent scalar formula and Gym's public API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import binomtest, nct, t

from nemo_gym.eval import (
    BenchmarkSamples,
    PlanningCell,
    PlanningParameters,
    RepeatAllocation,
    SampleSummary,
    aggregate_welch,
    plan_repeats,
    power_for_counts,
    simulate_plan,
)


def independent_statistics(
    cells: Sequence[PlanningCell],
    baseline_counts: Sequence[int],
    candidate_counts: Sequence[int],
    *,
    parameters: PlanningParameters,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute the fixed-variance Welch approximation without Gym utilities."""
    selected_weights = list(weights) if weights is not None else [1 / len(cells)] * len(cells)
    contributions, denominators = [], []
    for cell, baseline, candidate, weight in zip(
        cells, baseline_counts, candidate_counts, selected_weights, strict=True
    ):
        for variance, count in ((cell.baseline_variance, baseline), (cell.candidate_variance, candidate)):
            contribution = weight * weight * variance / count
            contributions.append(contribution)
            denominators.append(contribution * contribution / (count - 1))
    variance = math.fsum(contributions)
    standard_error = math.sqrt(variance)
    degrees = variance * variance / math.fsum(denominators)
    critical = float(t.ppf(1 - parameters.alpha, degrees))
    power = float(
        nct.sf(critical, degrees, (parameters.margin_points + parameters.expected_delta_points) / standard_error)
    )
    return dict(
        variance=variance,
        standard_error=standard_error,
        degrees_of_freedom=degrees,
        critical_value=critical,
        half_width=critical * standard_error,
        power=power,
    )


def _assert_close(actual: float, expected: float, *, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-11):
        raise ValueError(f"{context}: Gym={actual:.17g}, reference={expected:.17g}")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(row: Mapping[str, str]) -> PlanningCell:
    variance = float(row["variance_points2"])
    return PlanningCell(
        model=row["model"],
        benchmark=row["benchmark"],
        baseline_variance=variance,
        candidate_variance=variance,
        repeat_step=int(row["prepared_repeat_step"]),
    )


def validate_scalar_optimality(output: Path) -> list[dict[str, object]]:
    """Exhaust all cheaper scalar designs for unequal variance, increments and reuse."""
    cases = [(1.0, 1.0, 1, 1), (0.25, 1.0, 1, 1), (1.0, 0.25, 2, 4), (0.2, 0.8, 3, 8)]
    records = []
    for baseline_variance, candidate_variance, step, experiments in cases:
        cell = PlanningCell("toy", "benchmark", baseline_variance, candidate_variance, repeat_step=step)
        parameters = PlanningParameters(candidate_experiments=experiments, margin_points=2, max_repeats=100)
        actual = plan_repeats([cell], parameters=parameters)
        optimum: float = math.inf
        feasible: list[tuple[int, int]] = []
        first = math.ceil(2 / step) * step
        for baseline in range(first, 101, step):
            for candidate in range(first, 101, step):
                cost = baseline + experiments * candidate
                if cost > min(optimum, actual.total_cost):
                    continue
                power = independent_statistics([cell], [baseline], [candidate], parameters=parameters)["power"]
                if power >= parameters.target_power:
                    if cost < optimum:
                        feasible = []
                    optimum = cost
                    feasible.append((baseline, candidate))
        selected = actual.baseline_counts[0], actual.candidate_counts[0]
        if selected not in feasible or actual.total_cost != optimum:
            raise ValueError(f"Scalar optimum differs from exhaustive search: {selected}, independent={feasible}")
        records.append(
            dict(
                baseline_variance=baseline_variance,
                candidate_variance=candidate_variance,
                repeat_step=step,
                candidate_experiments=experiments,
                baseline_count=selected[0],
                candidate_count=selected[1],
                independent_minimum_cost=optimum,
                gym_cost=actual.total_cost,
                power=actual.statistics.power,
            )
        )
    _write_csv(output / "scalar_optimality.csv", records)
    return records


def validate_reference(reference_root: Path, output: Path) -> dict[str, object]:
    """Compare stored presentation calculations with Gym at identical frozen counts."""
    data = reference_root / "repeat_planning/input_data/notebook"
    if not data.is_dir():
        data = reference_root
    inputs_path = data / "planning_input.csv"
    inputs = _csv_rows(inputs_path)
    by_identity = {(row["model"], row["benchmark"]): row for row in inputs}
    if len(by_identity) != len(inputs):
        raise ValueError("Reference planning input contains duplicate model/benchmark identities")
    plans_path = data / "plans.json"
    plans = json.loads(plans_path.read_text())
    if plans["input_sha256"] != _sha256(inputs_path):
        raise ValueError("Reference planning input differs from the pinned plans.json hash")
    reference_files = [inputs_path, plans_path]
    designs = []
    locations = [
        "single_benchmark",
        "single_benchmark/power_70",
        "single_benchmark/power_60",
        "single_benchmark/margin_2/power_80",
        "single_benchmark/margin_2/power_70",
        "single_benchmark/margin_2/power_60",
    ]
    for location in locations:
        path = data / location / "single_benchmark_comparisons.csv"
        reference_files.append(path)
        rows = _csv_rows(path)
        if {(row["model"], row["benchmark"]) for row in rows} != set(by_identity):
            raise ValueError(f"{path}: reference design identities differ from planning inputs")
        for row in rows:
            source = by_identity[row["model"], row["benchmark"]]
            cell = _cell(source)
            _assert_close(cell.baseline_variance, float(row["variance_points2"]), context=f"{location}: variance")
            parameters = PlanningParameters(
                margin_points=float(row["margin_points"]),
                target_power=float(row["target"]),
                alpha=float(row["alpha_per_comparison"]),
                expected_delta_points=float(row["effect_points"]),
            )
            counts = [int(row["passes_per_arm"])]
            actual = power_for_counts([cell], counts, counts, parameters=parameters)
            oracle = independent_statistics([cell], counts, counts, parameters=parameters)
            for field, value in oracle.items():
                _assert_close(getattr(actual, field), value, context=f"{location}: {field}")
            for field, reference in (
                ("power", "analytic_power"),
                ("standard_error", "se_points"),
                ("degrees_of_freedom", "df"),
                ("half_width", "half_width_points"),
            ):
                _assert_close(getattr(actual, field), float(row[reference]), context=f"{location}: stored {field}")
            designs.append(
                dict(
                    model=cell.model,
                    benchmark=cell.benchmark,
                    margin=parameters.margin_points,
                    target=parameters.target_power,
                    baseline_count=counts[0],
                    candidate_count=counts[0],
                    gym_se=actual.standard_error,
                    reference_se=float(row["se_points"]),
                    gym_df=actual.degrees_of_freedom,
                    reference_df=float(row["df"]),
                    gym_power=actual.power,
                    reference_power=float(row["analytic_power"]),
                )
            )
    _write_csv(output / "reference_benchmark_parity.csv", designs)
    suite_rows = []
    cells = [_cell(row) for row in inputs]
    for name in ("pooled", "separate"):
        plan = plans[name]
        for result in plan["results"]:
            indices = [i for i, cell in enumerate(cells) if name == "pooled" or cell.model == result["comparison"]]
            selected = [cells[i] for i in indices]
            counts = [int(plan["counts_per_arm"][i]) for i in indices]
            weights = [float(plan["weights"][i]) for i in indices]
            parameters = PlanningParameters(margin_points=plan["margin_points"], alpha=plan["alpha_per_comparison"])
            actual = power_for_counts(selected, counts, counts, parameters=parameters, weights=weights)
            oracle = independent_statistics(selected, counts, counts, parameters=parameters, weights=weights)
            for field, value in oracle.items():
                _assert_close(
                    getattr(actual, field), value, context=f"{name}/{result['comparison']}: independent {field}"
                )
            for field, reference in (
                ("power", "power"),
                ("standard_error", "se_points"),
                ("degrees_of_freedom", "df"),
            ):
                _assert_close(
                    getattr(actual, field), result[reference], context=f"{name}/{result['comparison']}: {field}"
                )
            equal_cells = power_for_counts(selected, counts, counts, parameters=parameters)
            suite_rows.append(
                dict(
                    comparison=result["comparison"],
                    cells=len(selected),
                    gym_se=actual.standard_error,
                    reference_se=result["se_points"],
                    gym_df=actual.degrees_of_freedom,
                    reference_df=result["df"],
                    gym_power=actual.power,
                    reference_power=result["power"],
                    product_equal_cell_power=equal_cells.power,
                    weights_equal=len(set(weights)) == 1,
                )
            )
    _write_csv(output / "reference_suite_parity.csv", suite_rows)
    shared_path = data / "shared_baseline/margin_2/cells.csv"
    reference_files.append(shared_path)
    shared_records = []
    for row in _csv_rows(shared_path):
        cell = _cell(row)
        parameters = PlanningParameters(candidate_experiments=48, margin_points=2)
        actual = plan_repeats([cell], parameters=parameters)
        expected = int(row["baseline_passes_once"]), int(row["passes_per_candidate"])
        selected = actual.baseline_counts[0], actual.candidate_counts[0]
        if selected != expected:
            raise ValueError(f"Shared baseline {cell.model}/{cell.benchmark}: Gym {selected}, reference {expected}")
        _assert_close(actual.statistics.power, float(row["analytic_power"]), context="Shared baseline power")
        shared_records.append(
            dict(
                model=cell.model,
                benchmark=cell.benchmark,
                baseline=selected[0],
                candidate=selected[1],
                power=actual.statistics.power,
            )
        )
    _write_csv(output / "shared_baseline_parity.csv", shared_records)
    return dict(
        input_cells=len(inputs),
        benchmark_designs=len(designs),
        suite_designs=len(suite_rows),
        shared_baseline_designs=len(shared_records),
        shared_baseline_arm_counts=2 * len(shared_records),
        reference_source_commit=plans["source_commit"],
        variance_sources=dict(Counter(row["sd_source"] for row in inputs)),
        files={str(path.relative_to(data)): _sha256(path) for path in reference_files},
        max_power_error=max(abs(float(row["gym_power"]) - float(row["reference_power"])) for row in designs),
    )


def export_suite_planning_inputs(report_path: Path, output: Path) -> list[dict[str, object]]:
    """Reduce a saved comparison report to direct-SD planning YAML and compact evidence."""
    report = json.loads(report_path.read_text())
    records = []
    for cell in report["cells"]:
        if not cell["eligible"]:
            continue
        summaries = {}
        for side in ("baseline", "candidate"):
            evidence = cell[side]
            if evidence["coverage"]["status"] != "verified":
                raise ValueError(f"{cell['benchmark']} {side}: complete task coverage is not verified")
            scores = evidence["repeat_scores"]
            mean = math.fsum(scores) / len(scores)
            variance = math.fsum((value - mean) ** 2 for value in scores) / (len(scores) - 1)
            _assert_close(variance, evidence["summary"]["sample_variance"], context=f"{cell['benchmark']} {side}")
            summaries[side] = dict(
                count=len(scores), mean=mean, variance=variance, std=math.sqrt(variance), scores=scores
            )
        records.append(dict(model=cell["model"], benchmark=cell["benchmark"], **summaries))
    if not records:
        raise ValueError(f"Saved suite report '{report_path}' has no eligible model/benchmark cells to export.")
    groups = [("suite", "suite", records)]
    groups.extend(
        ("benchmark", "benchmark" if len(records) == 1 else f"benchmark_{index}", [row])
        for index, row in enumerate(records, 1)
    )
    for level, suffix, selected in groups:
        manifest = dict(
            schema_version=1,
            aggregation_level=level,
            baseline=dict(mode="existing"),
            candidate_experiments=1,
            margin_points=1,
            target_power=0.8,
            alpha=0.05,
            validation_trials=10000,
            cells=[
                dict(
                    model=row["model"],
                    benchmark=row["benchmark"],
                    baseline_std_points=row["baseline"]["std"],
                    candidate_std_points=row["candidate"]["std"],
                    existing_baseline_repeats=row["baseline"]["count"],
                    repeat_step=1,
                )
                for row in selected
            ],
        )
        (output / f"cluster_{suffix}_plan.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "cluster_numeric_evidence.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def validate_raw_normal_simulation(*, trials: int) -> dict[str, object]:
    """Compare direct normal observations with the production sufficient-statistic simulation."""
    cells = [PlanningCell("example", "first", 1.0, 1.5), PlanningCell("example", "second", 3.0, 0.7)]
    baseline, candidate = (11, 16), (19, 13)
    parameters = PlanningParameters()
    allocation = RepeatAllocation(
        baseline,
        candidate,
        power_for_counts(cells, baseline, candidate, parameters=parameters),
        baseline,
        sum(baseline) + sum(candidate),
        float(sum(baseline) + sum(candidate)),
        "frozen unequal-count diagnostic",
    )
    production = simulate_plan(cells, allocation, parameters=parameters, trials=trials, seed=20261109)
    rng = np.random.default_rng(20261110)
    observations = [
        (
            rng.normal(scale=math.sqrt(cell.baseline_variance), size=(trials, nb)),
            rng.normal(scale=math.sqrt(cell.candidate_variance), size=(trials, nc)),
        )
        for cell, nb, nc in zip(cells, baseline, candidate, strict=True)
    ]
    successes = false_positives = 0
    for trial in range(trials):
        samples = [
            BenchmarkSamples(
                SampleSummary(len(left[trial]), float(np.mean(left[trial])), float(np.var(left[trial], ddof=1))),
                SampleSummary(len(right[trial]), float(np.mean(right[trial])), float(np.var(right[trial], ddof=1))),
            )
            for left, right in observations
        ]
        result = aggregate_welch(samples, confidence_level=0.9)
        successes += result.ci_low > -parameters.margin_points
        false_positives += result.ci_low > 0
    raw = {}
    for name, count, comparison in (
        ("power", successes, production.power),
        ("false_positive", false_positives, production.false_positive),
    ):
        interval = binomtest(count, trials).proportion_ci(method="wilson")
        if interval.high < comparison.ci_low or comparison.ci_high < interval.low:
            raise ValueError(f"Raw-normal and sufficient-statistic {name} intervals disagree")
        raw[name] = dict(estimate=count / trials, ci_low=interval.low, ci_high=interval.high)
    return dict(
        baseline_counts=baseline,
        candidate_counts=candidate,
        trials=trials,
        raw_seed=20261110,
        sufficient_statistics_seed=20261109,
        raw_normal=raw,
        production=asdict(production),
        interval_overlap=True,
    )


def run_validation(
    *, reference_root: Path | None, output: Path, suite_report: Path | None, trials: int
) -> dict[str, object]:
    """Stage a complete audit before publishing its outputs.

    Failed checks preserve previous outputs. Publishing replaces generated
    files only, with completion markers last; publication errors remove those
    markers so partial outputs cannot appear to be a completed audit.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
        staging = Path(directory)
        receipt = _write_validation(
            reference_root=reference_root, output=staging, suite_report=suite_report, trials=trials
        )
        output.mkdir(parents=True, exist_ok=True)
        completion_files = ("validation_receipt.json", "VALIDATION.md")
        for name in completion_files:
            (output / name).unlink(missing_ok=True)
        try:
            for path in sorted(staging.iterdir()):
                if path.name not in completion_files:
                    path.replace(output / path.name)
            for name in completion_files:
                (staging / name).replace(output / name)
        except OSError:
            for name in completion_files:
                (output / name).unlink(missing_ok=True)
            raise
        return receipt


def _write_validation(
    *, reference_root: Path | None, output: Path, suite_report: Path | None, trials: int
) -> dict[str, object]:
    """Write one audit into an unpublished staging directory."""
    receipt: dict[str, object] = dict(
        gym_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=_sha256(Path(__file__)),
        scalar_optimality=validate_scalar_optimality(output),
    )
    receipt["gym_source_sha256"] = {
        str(path): _sha256(path)
        for path in {
            Path(inspect.getfile(function))
            for function in (plan_repeats, power_for_counts, simulate_plan, aggregate_welch)
        }
    }
    if reference_root is not None:
        receipt["reference"] = validate_reference(reference_root, output)
        revision = subprocess.run(
            ["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
        )
        receipt["reference_checkout_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    if suite_report is not None:
        receipt["cluster_input"] = dict(
            source_sha256=_sha256(suite_report), cells=export_suite_planning_inputs(suite_report, output)
        )
    simulations = []
    for name, variance, margin in (("ordinary", 1.0, 1.0), ("low_count_counterexample", 0.21983263058853658, 2.0)):
        cell = PlanningCell("illustration", name, variance, variance)
        parameters = PlanningParameters(margin_points=margin)
        allocation = plan_repeats([cell], parameters=parameters)
        simulation = simulate_plan([cell], allocation, parameters=parameters, trials=trials, seed=20261109)
        simulations.append(dict(case=name, allocation=asdict(allocation), simulation=asdict(simulation)))
    receipt["simulations"] = simulations
    receipt["raw_normal_simulation"] = validate_raw_normal_simulation(trials=trials)
    (output / "validation_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    paragraphs = [
        "# Repeat planning validation\n",
        "The reference artifacts are numeric inputs, not executable dependencies. The audit compares Gym with an independently written Welch formula and pinned presentation outputs. Counts are frozen for formula comparisons; old balanced counts do not create a product planning mode.\n",
        "## Method\n",
        "Gym computes uncertainty using its existing Welch comparison and estimates prospective power with a noncentral Student-t distribution. The standalone audit independently calculates these quantities and compares them with frozen reference designs.\n",
        f"Calculation source: [repeat planning]({Path(inspect.getfile(power_for_counts)).resolve()}). Full-precision input and output CSVs accompany this report.\n",
        "The observation unit is one complete benchmark pass. The delta is candidate minus baseline. SDs use normalized 0–100 points.\n",
        "With alpha=0.05, the lower bound is 95% one-sided (equivalently the lower endpoint of a 90% two-sided interval). The attached comparison README uses 95% two-sided intervals for its separate historical comparison.\n",
        "## Checks\n",
        f"- Independent exhaustive scalar optimum checks: {len(receipt['scalar_optimality'])}.\n",
    ]
    if "reference" in receipt:
        reference = receipt["reference"]
        paragraphs.extend(
            [
                f"- Pinned reference input cells: {reference['input_cells']}; benchmark designs: {reference['benchmark_designs']}; aggregate designs: {reference['suite_designs']}.\n",
                f"- Shared baseline K=48: {reference['shared_baseline_designs']} plans / {reference['shared_baseline_arm_counts']} matching arm counts.\n",
                f"- Maximum benchmark power difference: {reference['max_power_error']:.3g}. See the full-precision CSV tables.\n",
                f"- Reference variance sources: {reference['variance_sources']}. Proxy values are declared planning assumptions, not observed repeat variance.\n",
                "- Reference pooled weights give models equal weight and then benchmarks equal weight within each model. The product uses equal cells. The suite parity table reports both calculations explicitly.\n",
            ]
        )
    paragraphs.extend(
        [
            "## Simulation and interpretation\n",
            "Monte Carlo executes Gym's actual Welch calculation with newly simulated sample means and variances at frozen planned counts. The receipt includes the seed, trial count and interval estimates. The low-count example deliberately demonstrates that analytic power is an approximation, and an analytic target is not a demonstrated finite-sample power guarantee.\n",
            "| Case | Baseline / candidate repeats | Analytic power | Simulated power (95% interval) | Boundary false positive |\n| --- | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['case']} | {row['allocation']['baseline_counts'][0]} / {row['allocation']['candidate_counts'][0]} | {row['allocation']['statistics']['power']:.4%} | {row['simulation']['power']['estimate']:.2%} ({row['simulation']['power']['ci_low']:.2%}–{row['simulation']['power']['ci_high']:.2%}) | {row['simulation']['false_positive']['estimate']:.2%} |"
                for row in simulations
            ],
            "Pilot variance uncertainty is separate from the conditional calculations checked here. Repeating after inspecting the final decision is outside this fixed-count design.\n",
            "A separate two-benchmark diagnostic generates complete raw normal observations at unequal arm counts, recomputes their sample means and variances, and calls Gym's Welch comparison. Its power and false-positive intervals overlap those from the production sufficient-statistic simulator; full seeds and rates are in the receipt.\n",
            "## Cluster inputs\n",
            "The optional suite report is reduced to complete-pass scores and SDs, with no response text. Generated direct-SD YAML files exercise benchmark and suite planning with the historical baseline frozen.\n",
        ]
    )
    (output / "VALIDATION.md").write_text("\n".join(paragraphs))
    return receipt


def main() -> None:
    """Run the standalone validation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite-report", type=Path)
    parser.add_argument("--trials", type=int, default=10000)
    args = parser.parse_args()
    receipt = run_validation(
        reference_root=args.reference_root, output=args.output_dir, suite_report=args.suite_report, trials=args.trials
    )
    print(
        f"Validation passed: {args.output_dir / 'VALIDATION.md'} ({len(receipt['scalar_optimality'])} independent optima)"
    )


if __name__ == "__main__":
    main()
