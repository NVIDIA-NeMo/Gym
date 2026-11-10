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
"""Behavior tests for suite manifests, artifact processing, and numerical integration."""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.config.config_types import ConfigError
from nemo_gym.environment.comparison import resolve_benchmark_reward_metric
from nemo_gym.eval.comparison.loading import resolve_aggregate_metrics_fpath
from nemo_gym.eval.comparison.suite.loading import load_suite_run
from nemo_gym.eval.comparison.suite.processing import compare_suite_cell
from nemo_gym.eval.comparison.suite.report import write_suite_reports
from nemo_gym.eval.comparison.suite.runner import build_suite_comparison
from nemo_gym.eval.comparison.suite.schema import (
    SuiteComparisonConfig,
    SuiteComparisonResult,
    SuiteMetricResult,
    SuiteRunSource,
)
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, materialized_path_for


AGENT = "benchmark_agent"


def _write_run(
    directory: Path,
    values: list[dict[str, float]],
    *,
    tasks: tuple[int, ...] = (0, 1),
    planned_tasks: tuple[int, ...] | None = None,
    write_inputs: bool = True,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    rollouts = directory / "rollouts.jsonl"
    # Aggregate comparison must not attempt to parse trajectories.
    rollouts.write_text("trajectory contents are deliberately unavailable\n")
    entry = {
        "agent_ref": {"name": AGENT},
        "agent_metrics": {"num_repeats": len(values)},
        "repeat_level_metrics": [
            {"_ng_rollout_index": repeat, "sample_count": len(tasks), "missing_count": 0, **metrics}
            for repeat, metrics in enumerate(values)
        ]
        if len(values) > 1
        else [],
        "group_level_metrics": [
            {
                "_ng_task_index": task,
                "expected_num_rollouts": len(values),
                "missing_num_rollouts": 0,
                "rollout_infos": [
                    {"_ng_task_index": task, "_ng_rollout_index": repeat} for repeat in range(len(values))
                ],
            }
            for task in tasks
        ],
    }
    aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps([entry]))
    if write_inputs:
        rows = [
            {"agent_ref": {"name": AGENT}, "_ng_task_index": task, "_ng_rollout_index": repeat}
            for task in (planned_tasks if planned_tasks is not None else tasks)
            for repeat in range(len(values))
        ]
        materialized_path_for(rollouts).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    return str(rollouts)


def _values(*scores: float) -> list[dict[str, float]]:
    return [{"mean/reward": score} for score in scores]


def _scale(name: str = "mean/reward", *, low: float = 0, high: float = 1, higher: bool = True) -> dict:
    return {"name": name, "low": low, "high": high, "higher_is_better": higher}


def _manifest(tmp_path: Path, *, benchmarks: dict, models: dict) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "benchmarks": benchmarks, "models": models}))
    return path


def _compare(path: Path, confidence: float = 0.90) -> SuiteComparisonResult:
    return build_suite_comparison(
        SuiteComparisonConfig(suite_fpath=str(path), confidence_level=confidence),
        "gym eval compare-suite",
        reward_metric_resolver=resolve_benchmark_reward_metric,
    )


def _pair(tmp_path: Path, name: str, baseline: list[dict[str, float]], candidate: list[dict[str, float]]) -> dict:
    return {
        "baseline": {"rollouts": _write_run(tmp_path / name / "baseline", baseline)},
        "candidate": {"rollouts": _write_run(tmp_path / name / "candidate", candidate)},
    }


def _edit_entry(rollouts: str, update) -> None:
    path = aggregate_metrics_path_for(Path(rollouts))
    entries = orjson.loads(path.read_bytes())
    update(entries[0])
    path.write_bytes(orjson.dumps(entries))


def _write_routed_run(
    directory: Path,
    sources: list[tuple[str, str, tuple[int, ...], list[dict[str, float]]]],
) -> str:
    """Write stored summaries and taskset-routed plans for several scoring servers."""
    directory.mkdir(parents=True, exist_ok=True)
    rollouts = directory / "rollouts.jsonl"
    rollouts.write_text("trajectory contents are deliberately unavailable\n")
    entries = []
    rows = []
    for server, agent, tasks, values in sources:
        entries.append(
            {
                "agent_ref": {"name": agent},
                "_ng_environment_server": server,
                "agent_metrics": {"num_repeats": len(values)},
                "repeat_level_metrics": [
                    {"_ng_rollout_index": repeat, "sample_count": len(tasks), "missing_count": 0, **metrics}
                    for repeat, metrics in enumerate(values)
                ],
                "group_level_metrics": [
                    {
                        "_ng_task_index": task,
                        "expected_num_rollouts": len(values),
                        "missing_num_rollouts": 0,
                        "rollout_infos": [
                            {"_ng_task_index": task, "_ng_rollout_index": repeat} for repeat in range(len(values))
                        ],
                    }
                    for task in tasks
                ],
            }
        )
        rows.extend(
            {"_ng_environment_server": server, "_ng_task_index": task, "_ng_rollout_index": repeat}
            for task in tasks
            for repeat in range(len(values))
        )
    aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps(entries))
    materialized_path_for(rollouts).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    return str(rollouts)


def test_multiple_metrics_preserve_equal_benchmark_and_equal_cell_weights(tmp_path: Path) -> None:
    paired_metrics = _pair(
        tmp_path,
        "two_metrics",
        [{"mean/reward": 0.2, "accuracy": 20}, {"mean/reward": 0.4, "accuracy": 40}],
        [{"mean/reward": 0.4, "accuracy": 40}, {"mean/reward": 0.6, "accuracy": 60}],
    )
    path = _manifest(
        tmp_path,
        benchmarks={
            "first": {"metrics": [_scale(), _scale("accuracy", high=100)]},
            "second": {"metrics": [_scale()]},
        },
        models={
            "model-a": {
                "first": paired_metrics,
                "second": _pair(tmp_path, "second", _values(0.2, 0.2), _values(0.8, 0.8)),
            },
            "model-b": {"second": _pair(tmp_path, "third", _values(0.4, 0.6), _values(0.2, 0.4))},
        },
    )
    result = _compare(path)
    first = next(cell for cell in result.cells if cell.benchmark == "first")
    assert first.baseline.repeat_scores == pytest.approx([20, 40])
    assert first.candidate.repeat_scores == pytest.approx([40, 60])
    assert first.baseline.summary.sample_variance == pytest.approx(200)
    assert first.metrics[1].baseline_raw == [20, 40]
    assert first.metrics[1].baseline_scaled == pytest.approx([20, 40])
    assert first.model_weight == 0.5
    assert first.overall_weight == pytest.approx(1 / 3)
    assert {aggregate.name: aggregate.statistics.delta for aggregate in result.per_model} == pytest.approx(
        {"model-a": 40, "model-b": -20}
    )
    # Equal cells, including sparse models; averaging the two model means would instead yield 10.
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.weights == pytest.approx([1 / 3] * 3)
    assert result.overall.statistics.variance == pytest.approx(400 / 9)
    assert all(cell.baseline.coverage.status == "verified" for cell in result.cells)
    assert not result.warnings


def test_resolves_reward_bounds_relative_paths_and_keeps_sources_unchanged(tmp_path: Path, monkeypatch) -> None:
    baseline = _write_run(tmp_path / "baseline", _values(0.1, 0.3, 0.5))
    candidate = _write_run(tmp_path / "candidate", _values(0.3, 0.5, 0.7))
    sources = [aggregate_metrics_path_for(Path(path)) for path in (baseline, candidate)]
    before = [path.read_bytes() for path in sources]
    path = _manifest(
        tmp_path,
        benchmarks={"birdbench": {"metrics": [{"name": "mean/reward"}]}},
        models={
            "llama": {
                "birdbench": {
                    "baseline": {"rollouts": "baseline/rollouts.jsonl"},
                    "candidate": {"rollouts": "candidate/rollouts.jsonl"},
                }
            }
        },
    )
    monkeypatch.chdir(tmp_path.parent)
    result = _compare(path)
    cell = result.cells[0]
    assert cell.metrics[0].low == 0
    assert cell.metrics[0].high == 1
    assert cell.metrics[0].higher_is_better is True
    assert cell.metrics[0].definition_source.endswith("benchmarks/birdbench/manifest.yaml:reward")
    assert cell.baseline.rollouts == Path(baseline)
    assert cell.baseline.agent == AGENT
    assert cell.statistics.delta == pytest.approx(20)
    assert cell.statistics.confidence_level == 0.9
    assert [source.read_bytes() for source in sources] == before
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_complete_scale_override_reverses_lower_is_better_metric(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"birdbench": {"metrics": [_scale(low=0, high=10, higher=False)]}},
        models={"model": {"birdbench": _pair(tmp_path, "run", _values(4, 6), _values(2, 4))}},
    )
    cell = build_suite_comparison(SuiteComparisonConfig(suite_fpath=str(path)), "direct comparison").cells[0]
    assert cell.baseline.repeat_scores == pytest.approx([60, 40])
    assert cell.candidate.repeat_scores == pytest.approx([80, 60])
    assert cell.statistics.delta == pytest.approx(20)
    assert cell.metrics[0].definition_source == "suite manifest override"


def test_direct_comparison_requires_scale_or_catalog_resolver(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"birdbench": {"metrics": [{"name": "mean/reward"}]}},
        models={"model": {"birdbench": _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))}},
    )
    with pytest.raises(ConfigError, match="no catalog reward resolver.*theoretical low, high, and higher_is_better"):
        build_suite_comparison(SuiteComparisonConfig(suite_fpath=str(path)), "direct comparison")


def test_adjustable_confidence_changes_interval_only_and_supports_unequal_repeats(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={"model": {"test": _pair(tmp_path, "run", _values(0.1, 0.4, 0.7), _values(0.2, 0.5))}},
    )
    result90 = _compare(path).overall.statistics
    result95 = _compare(path, 0.95).overall.statistics
    assert result95.delta == result90.delta
    assert result95.standard_error == result90.standard_error
    assert result95.degrees_of_freedom == result90.degrees_of_freedom
    assert result95.ci_low < result90.ci_low < result90.ci_high < result95.ci_high


def test_single_repeat_is_excluded_as_pair_and_weights_recompute(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in ("single", "included")},
        models={
            "model": {
                "single": _pair(tmp_path, "single", _values(0.0), _values(0.5, 1.0)),
                "included": _pair(tmp_path, "included", _values(0.2, 0.4), _values(0.4, 0.6)),
            }
        },
    )
    result = _compare(path)
    excluded = next(cell for cell in result.cells if cell.benchmark == "single")
    included = next(cell for cell in result.cells if cell.benchmark == "included")
    assert excluded.eligible is False
    assert excluded.baseline.repeat_count == 1
    assert excluded.statistics is None
    assert excluded.overall_weight == 0
    assert included.model_weight == included.overall_weight == 1
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.cell_ids == ["model/included"]
    assert any("single" in warning and "at least two repeats" in warning for warning in result.warnings)


def test_all_single_repeat_cells_fail_with_exclusion_reason(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={"model": {"test": _pair(tmp_path, "run", _values(0.1), _values(0.3))}},
    )
    with pytest.raises(ConfigError, match="No eligible.*at least two repeats"):
        _compare(path)


def test_multiple_agents_require_explicit_selection(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    metrics = aggregate_metrics_path_for(Path(pair["baseline"]["rollouts"]))
    entries = orjson.loads(metrics.read_bytes())
    other = dict(entries[0]) | {"agent_ref": {"name": "other_agent"}}
    metrics.write_bytes(orjson.dumps([*entries, other]))
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    with pytest.raises(ConfigError, match="multiple aggregate entries.*set 'agent'.*'environment_server'"):
        _compare(path)
    pair["baseline"]["agent"] = AGENT
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    assert _compare(path).cells[0].baseline.agent == AGENT


@pytest.mark.parametrize(
    ("selectors", "expected_agent", "expected_server"),
    [
        ({"agent": "bird_agent"}, "bird_agent", "bird_env"),
        ({"environment_server": "mmlu_env"}, "mcqa_agent", "mmlu_env"),
        ({"agent": "bird_agent", "environment_server": "bird_env"}, "bird_agent", "bird_env"),
    ],
)
def test_agent_and_environment_selectors_retain_both_identities(
    tmp_path: Path, selectors: dict[str, str], expected_agent: str, expected_server: str
) -> None:
    rollouts = _write_routed_run(
        tmp_path,
        [
            ("bird_env", "bird_agent", (0, 1), _values(0.2, 0.4)),
            ("mmlu_env", "mcqa_agent", (2, 3), _values(0.6, 0.8)),
        ],
    )
    run = load_suite_run(SuiteRunSource(rollouts=Path(rollouts), **selectors), directory=tmp_path)
    assert run.agent == expected_agent
    assert run.environment_server == expected_server
    expected_tasks = frozenset((0, 1) if expected_server == "bird_env" else (2, 3))
    assert run.planned_tasks_by_repeat == {0: expected_tasks, 1: expected_tasks}
    assert [row["mean/reward"] for row in run.repeat_rows] == pytest.approx(
        [0.2, 0.4] if expected_server == "bird_env" else [0.6, 0.8]
    )


def test_single_server_is_inferred_without_selectors(tmp_path: Path) -> None:
    rollouts = _write_routed_run(tmp_path, [("bird_env", "bird_agent", (0, 1), _values(0.2, 0.4))])
    run = load_suite_run(SuiteRunSource(rollouts=Path(rollouts)), directory=tmp_path)
    assert run.agent == "bird_agent"
    assert run.environment_server == "bird_env"
    assert run.planned_tasks_by_repeat == {0: frozenset((0, 1)), 1: frozenset((0, 1))}


@pytest.mark.parametrize(
    "selectors",
    [
        {},
        {"agent": "absent_agent"},
        {"environment_server": "absent_env"},
        {"agent": "bird_agent", "environment_server": "mmlu_env"},
    ],
)
def test_ambiguous_unknown_or_conflicting_selectors_list_available_entries(
    tmp_path: Path, selectors: dict[str, str]
) -> None:
    rollouts = _write_routed_run(
        tmp_path,
        [
            ("bird_env", "bird_agent", (0, 1), _values(0.2, 0.4)),
            ("mmlu_env", "mcqa_agent", (2, 3), _values(0.6, 0.8)),
        ],
    )
    with pytest.raises(ConfigError, match="Available entries:") as error:
        load_suite_run(SuiteRunSource(rollouts=Path(rollouts), **selectors), directory=tmp_path)
    message = str(error.value)
    for identity in ("bird_agent", "bird_env", "mcqa_agent", "mmlu_env"):
        assert identity in message
    if not selectors:
        assert "set 'agent' or 'environment_server'" in message
    else:
        assert "No aggregate entry" in message


def test_shared_harness_server_derived_labels_are_retained(tmp_path: Path) -> None:
    # When servers share a harness, aggregation uses server-derived agent labels.
    rollouts = _write_routed_run(
        tmp_path,
        [
            ("bird_env", "bird_env", (0, 1), _values(0.2, 0.4)),
            ("mmlu_env", "mmlu_env", (2, 3), _values(0.6, 0.8)),
        ],
    )
    selected = load_suite_run(
        SuiteRunSource(rollouts=Path(rollouts), environment_server="mmlu_env"), directory=tmp_path
    )
    assert selected.agent == "mmlu_env"
    assert selected.environment_server == "mmlu_env"
    assert [row["mean/reward"] for row in selected.repeat_rows] == pytest.approx([0.6, 0.8])
    assert selected.planned_tasks_by_repeat == {0: frozenset((2, 3)), 1: frozenset((2, 3))}


@pytest.mark.parametrize("overlapping_indices", [False, True])
def test_taskset_servers_share_a_file_without_contaminating_coverage_or_statistics(
    tmp_path: Path, overlapping_indices: bool
) -> None:
    second_tasks = (0, 1) if overlapping_indices else (2, 3)
    baseline = _write_routed_run(
        tmp_path / "baseline",
        [
            ("bird_simple_env", "bird_simple_agent", (0, 1), _values(0.2, 0.4)),
            ("mmlu_simple_env", "mcqa_simple_agent", second_tasks, _values(0.6, 0.8)),
        ],
    )
    candidate = _write_routed_run(
        tmp_path / "candidate",
        [
            ("bird_hermes_env", "bird_hermes_agent", (0, 1), _values(0.4, 0.6)),
            ("mmlu_hermes_env", "mcqa_hermes_agent", second_tasks, _values(0.7, 0.9)),
        ],
    )
    path = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in ("bird", "mmlu")},
        models={
            "model": {
                name: {
                    "baseline": {"rollouts": baseline, "environment_server": f"{name}_simple_env"},
                    "candidate": {"rollouts": candidate, "environment_server": f"{name}_hermes_env"},
                }
                for name in ("bird", "mmlu")
            }
        },
    )
    result = _compare(path)
    assert result.warnings == []
    assert {cell.benchmark: cell.statistics.delta for cell in result.cells} == pytest.approx({"bird": 20, "mmlu": 10})
    for cell in result.cells:
        assert cell.baseline.coverage.status == cell.candidate.coverage.status == "verified"
        assert cell.baseline.coverage.planned_task_count == cell.candidate.coverage.planned_task_count == 2
        assert cell.baseline.environment_server == f"{cell.benchmark}_simple_env"
        assert cell.candidate.environment_server == f"{cell.benchmark}_hermes_env"
        assert cell.model_weight == cell.overall_weight == 0.5
        assert cell.statistics.ci_low is not None
        assert cell.statistics.ci_high is not None
    assert result.overall.statistics.baseline_mean == pytest.approx(50)
    assert result.overall.statistics.candidate_mean == pytest.approx(65)
    assert result.overall.statistics.delta == pytest.approx(15)
    assert result.overall.weights == [0.5, 0.5]
    assert result.overall.statistics.ci_low is not None
    assert result.per_model[0].statistics.ci_high is not None
    report_dir = tmp_path / "reports"
    write_suite_reports(result, report_dir, "both")
    markdown = (report_dir / "suite_compare_report.md").read_text()
    for name in ("bird", "mmlu"):
        for role in ("simple", "hermes"):
            assert f"Environment server: `{name}_{role}_env`." in markdown
    restored = SuiteComparisonResult.model_validate_json((report_dir / "suite_compare_report.json").read_bytes())
    assert restored == result


@pytest.mark.parametrize("missing_server", ["selected", "other"])
def test_server_filtering_reports_missing_tasks_only_for_selected_server(tmp_path: Path, missing_server: str) -> None:
    baseline = _write_routed_run(
        tmp_path / "baseline",
        [
            ("bird_env", "bird_agent", (0, 1), _values(0.2, 0.4)),
            ("mmlu_env", "mcqa_agent", (2, 3), _values(0.6, 0.8)),
        ],
    )
    candidate = _write_routed_run(
        tmp_path / "candidate",
        [
            ("bird_env", "bird_agent", (0, 1), _values(0.4, 0.6)),
            ("mmlu_env", "mcqa_agent", (2, 3), _values(0.7, 0.9)),
        ],
    )
    aggregate_path = aggregate_metrics_path_for(Path(candidate))
    entries = orjson.loads(aggregate_path.read_bytes())
    entries[0 if missing_server == "selected" else 1]["group_level_metrics"].pop()
    aggregate_path.write_bytes(orjson.dumps(entries))
    path = _manifest(
        tmp_path,
        benchmarks={"bird": {"metrics": [_scale()]}},
        models={
            "model": {
                "bird": {
                    "baseline": {"rollouts": baseline, "environment_server": "bird_env"},
                    "candidate": {"rollouts": candidate, "environment_server": "bird_env"},
                }
            }
        },
    )
    result = _compare(path)
    cell = result.cells[0]
    assert cell.statistics.delta == pytest.approx(20)
    if missing_server == "other":
        assert cell.baseline.coverage.status == "verified"
        assert cell.candidate.coverage.status == "verified"
        assert result.warnings == []
        assert cell.statistics.ci_low is not None
        assert result.per_model[0].statistics.ci_high is not None
        assert result.overall.statistics.ci_low is not None
        return
    assert cell.baseline.coverage.status == "incomplete"
    assert cell.candidate.coverage.status == "incomplete"
    assert cell.candidate.coverage.planned_task_count == 2
    assert "Results are missing 1 of 2 planned tasks (task indices: 1)." in cell.candidate.coverage.notes
    assert "Results are missing 2 of 4 planned rollouts." in cell.candidate.coverage.notes
    assert cell.statistics.ci_low is None
    assert result.per_model[0].statistics.ci_high is None
    assert result.overall.statistics.ci_low is None


def test_stamped_rows_use_server_while_unstamped_rows_keep_agent_fallback(tmp_path: Path) -> None:
    rollouts = _write_routed_run(tmp_path, [("bird_env", "bird_agent", (0, 1), _values(0.2, 0.4))])
    rows = [
        # Server ownership takes precedence even when the agent label differs.
        {"_ng_environment_server": "bird_env", "agent_ref": {"name": "different_label"}, "_ng_task_index": 0},
        {"agent_ref": {"name": "bird_agent"}, "_ng_task_index": 1},
        {"_ng_task_index": 2},
        # Neither another server nor another unstamped agent belongs to this entry.
        {"_ng_environment_server": "mmlu_env", "agent_ref": {"name": "bird_agent"}, "_ng_task_index": 3},
        {"agent_ref": {"name": "mcqa_agent"}, "_ng_task_index": 4},
    ]
    materialized_path_for(Path(rollouts)).write_bytes(
        b"".join(orjson.dumps(row | {"_ng_rollout_index": repeat}) + b"\n" for row in rows for repeat in (0, 1))
    )
    run = load_suite_run(SuiteRunSource(rollouts=Path(rollouts)), directory=tmp_path)
    assert run.planned_tasks_by_repeat == {0: frozenset((0, 1, 2)), 1: frozenset((0, 1, 2))}


def test_unstamped_aggregate_cannot_claim_server_stamped_inputs(tmp_path: Path) -> None:
    rollouts = _write_run(tmp_path, _values(0.2, 0.4))
    inputs = materialized_path_for(Path(rollouts))
    rows = [orjson.loads(line) | {"_ng_environment_server": "bird_env"} for line in inputs.read_bytes().splitlines()]
    inputs.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    with pytest.raises(ConfigError, match="Cannot match stamped materialized inputs.*has no _ng_environment_server"):
        load_suite_run(SuiteRunSource(rollouts=Path(rollouts)), directory=tmp_path)


def test_different_selector_fields_cannot_reuse_the_same_source_samples(tmp_path: Path) -> None:
    rollouts = _write_routed_run(tmp_path / "source", [("bird_env", "bird_agent", (0, 1), _values(0.2, 0.4))])
    path = _manifest(
        tmp_path,
        benchmarks={"bird": {"metrics": [_scale()]}},
        models={
            "model": {
                "bird": {
                    "baseline": {"rollouts": rollouts, "environment_server": "bird_env"},
                    "candidate": {"rollouts": rollouts, "agent": "bird_agent"},
                }
            }
        },
    )
    with pytest.raises(ConfigError, match="reuses the same aggregate file and agent.*independent source"):
        _compare(path)


def test_missing_repeat_metrics_errors_without_reconstructing(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    _edit_entry(pair["baseline"]["rollouts"], lambda entry: entry.pop("repeat_level_metrics"))
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    with pytest.raises(ConfigError, match="lacks repeat_level_metrics.*gym eval aggregate"):
        _compare(path)


@pytest.mark.parametrize("value", [None, "missing"])
def test_missing_or_nonnumeric_selected_metric_fails_without_dropping_it(tmp_path: Path, value: object) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    _edit_entry(pair["baseline"]["rollouts"], lambda entry: entry["repeat_level_metrics"][0].update({"custom": value}))
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale(), _scale("custom")]}},
        models={"model": {"test": pair}},
    )
    with pytest.raises(ConfigError, match="requires numeric metric 'custom'"):
        _compare(path)


def test_custom_metric_requires_explicit_theoretical_definition(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [{"name": "accuracy"}]}},
        models={
            "model": {
                "test": _pair(tmp_path, "run", [{"accuracy": 1}, {"accuracy": 2}], [{"accuracy": 3}, {"accuracy": 4}])
            }
        },
    )
    with pytest.raises(ConfigError, match="no general Gym scale definition.*theoretical low"):
        _compare(path)


@pytest.mark.parametrize("mode", ["missing_task", "partial_repeat", "different_task_sets", "masked_rollouts"])
def test_known_incomplete_coverage_suppresses_dependent_intervals(tmp_path: Path, mode: str) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    baseline = pair["baseline"]["rollouts"]
    if mode == "missing_task":
        _edit_entry(baseline, lambda entry: entry["group_level_metrics"].pop())
    elif mode == "partial_repeat":

        def incomplete(entry):
            entry["group_level_metrics"][1]["rollout_infos"].pop()
            entry["group_level_metrics"][1]["missing_num_rollouts"] = 1
            entry["repeat_level_metrics"][1].update({"sample_count": 1, "missing_count": 1})

        _edit_entry(baseline, incomplete)
    elif mode == "different_task_sets":
        pair["candidate"]["rollouts"] = _write_run(tmp_path / "different", _values(0.4, 0.6), tasks=(4, 5))
    else:
        _edit_entry(baseline, lambda entry: entry["agent_metrics"].update({"coverage/masked_rollouts": 1}))
    path = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in ("incomplete", "complete")},
        models={
            "model": {
                "incomplete": pair,
                "complete": _pair(tmp_path, "complete", _values(0.1, 0.3), _values(0.2, 0.4)),
            }
        },
    )
    result = _compare(path)
    incomplete = next(cell for cell in result.cells if cell.benchmark == "incomplete")
    complete = next(cell for cell in result.cells if cell.benchmark == "complete")
    assert incomplete.eligible is True
    assert incomplete.statistics.delta == pytest.approx(20)
    assert incomplete.statistics.ci_low is None
    assert incomplete.statistics.ci_high is None
    assert (
        incomplete.statistics.interval_unavailable_reason == "Known incomplete task coverage in an included benchmark."
    )
    assert complete.statistics.ci_low is not None
    assert result.overall.statistics.ci_low is None
    assert result.per_model[0].statistics.ci_high is None
    assert result.overall.weights == [0.5, 0.5]
    assert result.warnings
    if mode == "missing_task":
        assert "Results are missing 1 of 2 planned tasks (task indices: 1)." in incomplete.baseline.coverage.notes
        assert "Results are missing 2 of 4 planned rollouts." in incomplete.baseline.coverage.notes
    elif mode == "partial_repeat":
        assert "Results are missing 1 of 4 planned rollouts." in incomplete.baseline.coverage.notes
        assert not any("planned tasks (task indices:" in note for note in incomplete.baseline.coverage.notes)


@pytest.mark.parametrize(
    ("recorded_tasks", "planned_tasks", "expected_notes"),
    [
        ((0,), (0, 1), ["Results are missing 1 of 2 planned tasks (task indices: 1)."]),
        (
            (0, 2),
            (0, 1),
            [
                "Results are missing 1 of 2 planned tasks (task indices: 1).",
                "Results include 1 unplanned task (task indices: 2).",
                "Results are missing 2 of 4 planned rollouts.",
                "Results include 2 unplanned rollouts.",
            ],
        ),
        (
            (0, 4, 2),
            (0,),
            ["Results include 2 unplanned tasks (task indices: 2, 4).", "Results include 4 unplanned rollouts."],
        ),
        (
            (0,),
            tuple(range(12, -1, -1)),
            [
                "Results are missing 12 of 13 planned tasks "
                "(task indices: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... (+2 more))."
            ],
        ),
    ],
)
def test_coverage_warnings_identify_missing_and_unplanned_tasks(
    tmp_path: Path,
    recorded_tasks: tuple[int, ...],
    planned_tasks: tuple[int, ...],
    expected_notes: list[str],
) -> None:
    pair = {
        "baseline": {"rollouts": _write_run(tmp_path / "baseline", _values(0.2, 0.4), tasks=planned_tasks)},
        "candidate": {
            "rollouts": _write_run(
                tmp_path / "candidate", _values(0.4, 0.6), tasks=recorded_tasks, planned_tasks=planned_tasks
            )
        },
    }
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    result = _compare(path)
    cell = result.cells[0]
    assert cell.candidate.coverage.status == "incomplete"
    assert cell.candidate.coverage.recorded_task_count == len(recorded_tasks)
    assert cell.candidate.coverage.planned_task_count == len(planned_tasks)
    for note in expected_notes:
        assert note in cell.candidate.coverage.notes
        assert f"model/test candidate: {note}" in result.warnings
    assert cell.statistics.delta == pytest.approx(20)
    assert cell.statistics.ci_low is None
    assert result.overall.statistics.ci_high is None


def test_without_materialized_inputs_coverage_is_explicitly_unverified(tmp_path: Path) -> None:
    pair = {
        "baseline": {"rollouts": _write_run(tmp_path / "baseline", _values(0.2, 0.4), write_inputs=False)},
        "candidate": {"rollouts": _write_run(tmp_path / "candidate", _values(0.4, 0.6), write_inputs=False)},
    }
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    result = _compare(path)
    assert result.cells[0].baseline.coverage.status == "unverified"
    assert result.cells[0].baseline.coverage.planned_task_count is None
    assert result.overall.statistics.ci_low is not None
    assert any("materialized inputs are unavailable" in warning for warning in result.warnings)


def test_reusing_source_samples_is_rejected(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    pair["candidate"] = dict(pair["baseline"])
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    with pytest.raises(ConfigError, match="reuses the same aggregate file and agent.*independent source"):
        _compare(path)


def test_duplicate_repeat_indices_are_rejected(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    _edit_entry(
        pair["baseline"]["rollouts"], lambda entry: entry["repeat_level_metrics"][1].update({"_ng_rollout_index": 0})
    )
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    with pytest.raises(ConfigError, match="duplicate repeat indices"):
        _compare(path)


@pytest.mark.parametrize("field", ["aggregate_metrics", "materialized_inputs"])
@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("source_list", [False, True])
def test_source_artifact_overrides_are_rejected(tmp_path: Path, field: str, role: str, source_list: bool) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    pair[role][field] = "unsupported.json"
    if source_list:
        pair[role] = [pair[role]]
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    with pytest.raises(ConfigError, match="Cannot load suite manifest") as error:
        _compare(path)

    cause = error.value.__cause__
    assert isinstance(cause, ValidationError)
    errors = cause.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "extra_forbidden"
    assert errors[0]["loc"] == ("models", "model", "test", role, 0, field)
    assert {file: file.read_bytes() for file in sources} == sources


@pytest.mark.parametrize("companions_beside_alias", [False, True])
def test_rollout_alias_reads_same_companions_as_run_comparison(tmp_path: Path, companions_beside_alias: bool) -> None:
    target = Path(_write_run(tmp_path / "blob", _values(0.2, 0.4)))
    candidate = _write_run(tmp_path / "candidate", _values(0.4, 0.6))
    alias = tmp_path / "baseline" / "rollouts.jsonl"
    alias.parent.mkdir()
    alias.symlink_to(target)
    if companions_beside_alias:
        for companion in (aggregate_metrics_path_for, materialized_path_for):
            companion(target).replace(companion(alias))
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={
            "model": {
                "test": {
                    "baseline": {"rollouts": str(alias.relative_to(tmp_path))},
                    "candidate": {"rollouts": candidate},
                }
            }
        },
    )
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    result = _compare(path)

    cell = result.cells[0]
    expected_rollouts = alias if companions_beside_alias else target
    assert cell.baseline.rollouts == expected_rollouts
    assert cell.baseline.aggregate_metrics == resolve_aggregate_metrics_fpath(str(alias), None)
    assert cell.baseline.aggregate_metrics == aggregate_metrics_path_for(expected_rollouts)
    assert cell.baseline.materialized_inputs == materialized_path_for(expected_rollouts)
    assert cell.baseline.coverage.status == cell.candidate.coverage.status == "verified"
    assert cell.baseline.coverage.planned_task_count == 2
    assert cell.baseline.repeat_scores == pytest.approx([20, 40])
    assert cell.candidate.repeat_scores == pytest.approx([40, 60])
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.statistics.ci_low is not None
    assert result.warnings == []
    assert {file: file.read_bytes() for file in sources} == sources


def test_rollout_alias_rejects_competing_companion_layouts_like_run_comparison(tmp_path: Path) -> None:
    target = Path(_write_run(tmp_path / "blob", _values(0.8, 0.9)))
    alias = Path(_write_run(tmp_path / "alias", _values(0.2, 0.4)))
    alias.unlink()
    alias.symlink_to(target)
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    with pytest.raises(ConfigError, match="beside the rollout alias"):
        resolve_aggregate_metrics_fpath(str(alias), None)
    with pytest.raises(ConfigError, match="beside the rollout alias"):
        load_suite_run(SuiteRunSource(rollouts=alias), directory=tmp_path)

    assert {file: file.read_bytes() for file in sources} == sources


def test_rollout_aliases_cannot_reuse_physical_source_with_distinct_aggregate_companions(tmp_path: Path) -> None:
    target = tmp_path / "blob" / "data.jsonl"
    target.parent.mkdir()
    baseline = Path(_write_run(tmp_path / "baseline", _values(0.2, 0.4)))
    baseline.replace(target)
    baseline.symlink_to(target)
    candidate = Path(_write_run(tmp_path / "candidate", _values(0.2, 0.4)))
    candidate.unlink()
    candidate.symlink_to(target)
    sources = {
        role: {"rollouts": str(alias.relative_to(tmp_path))}
        for role, alias in (("baseline", baseline), ("candidate", candidate))
    }
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": sources}})

    with pytest.raises(ConfigError, match="reuses the same rollouts file and agent.*independent source"):
        _compare(path)


def test_companion_inventory_follows_relative_rollout_path_and_ignores_other_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = {
        "baseline": {"rollouts": _write_run(tmp_path / "baseline", _values(0.2, 0.4))},
        "candidate": {"rollouts": _write_run(tmp_path / "candidate", _values(0.4, 0.6))},
    }
    inventory = materialized_path_for(Path(pair["baseline"]["rollouts"]))
    pair["baseline"]["rollouts"] = "baseline/rollouts.jsonl"
    # The foreign agent reuses the same indices and has extra tasks: all of its rows
    # must be filtered before duplicate detection and coverage verification.
    rows = [
        {"agent_ref": {"name": agent}, "_ng_task_index": task, "_ng_rollout_index": repeat}
        for agent, tasks in ((AGENT, (0, 1)), ("other_agent", (0, 1, 50)))
        for task in tasks
        for repeat in (0, 1)
    ]
    inventory.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    unrelated_directory = tmp_path / "unrelated"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)
    result = _compare(path)
    cell = result.cells[0]
    assert cell.baseline.materialized_inputs == inventory.resolve()
    assert cell.baseline.coverage.status == cell.candidate.coverage.status == "verified"
    assert cell.baseline.coverage.recorded_task_count == cell.baseline.coverage.planned_task_count == 2
    assert cell.baseline.repeat_scores == pytest.approx([20, 40])
    assert cell.candidate.repeat_scores == pytest.approx([40, 60])
    assert cell.statistics.delta == pytest.approx(20)
    assert cell.statistics.ci_low is not None
    assert result.per_model[0].statistics.ci_high is not None
    assert result.overall.statistics.ci_low is not None
    assert result.warnings == []


def test_malformed_companion_inventory_is_rejected(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    inventory = materialized_path_for(Path(pair["baseline"]["rollouts"]))
    inventory.write_text("not valid JSON\n")
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    with pytest.raises(ConfigError, match="Cannot read materialized inputs.*rollouts_materialized_inputs.jsonl"):
        _compare(path)


def test_different_plans_suppress_intervals_when_recorded_task_groups_are_unavailable(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "different_plan", _values(0.2, 0.4), _values(0.4, 0.6))
    for source in pair.values():
        _edit_entry(source["rollouts"], lambda entry: entry.update({"group_level_metrics": []}))
    path = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in ("different_plan", "complete")},
        models={
            "model": {
                "different_plan": pair,
                "complete": _pair(tmp_path, "complete", _values(0.1, 0.3), _values(0.2, 0.4)),
            }
        },
    )
    control = _compare(path)
    control_cell = next(cell for cell in control.cells if cell.benchmark == "different_plan")
    assert control_cell.baseline.coverage.status == control_cell.candidate.coverage.status == "unverified"
    assert control_cell.statistics.ci_low is not None
    inventory = materialized_path_for(Path(pair["candidate"]["rollouts"]))
    rows = [
        {"agent_ref": {"name": AGENT}, "_ng_task_index": task, "_ng_rollout_index": repeat}
        for task in (2, 3)
        for repeat in (0, 1)
    ]
    inventory.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    result = _compare(path)
    cell = next(cell for cell in result.cells if cell.benchmark == "different_plan")
    complete = next(cell for cell in result.cells if cell.benchmark == "complete")
    note = "Baseline and candidate planned task sets differ."
    assert cell.baseline.coverage.status == cell.candidate.coverage.status == "incomplete"
    assert cell.baseline.coverage.recorded_task_count == cell.candidate.coverage.recorded_task_count == 0
    assert cell.baseline.coverage.planned_task_count == cell.candidate.coverage.planned_task_count == 2
    assert set(cell.baseline.coverage.notes) == set(control_cell.baseline.coverage.notes) | {note}
    assert set(cell.candidate.coverage.notes) == set(control_cell.candidate.coverage.notes) | {note}
    for role in ("baseline", "candidate"):
        assert f"model/different_plan {role}: {note}" in result.warnings
    assert not any("Candidate and baseline have results for different tasks" in warning for warning in result.warnings)
    assert cell.statistics.baseline_mean == pytest.approx(30)
    assert cell.statistics.candidate_mean == pytest.approx(50)
    assert cell.statistics.delta == pytest.approx(20)
    assert cell.model_weight == cell.overall_weight == 0.5
    assert complete.baseline.coverage.status == complete.candidate.coverage.status == "verified"
    assert complete.statistics.ci_low is not None
    for statistics in (cell.statistics, result.per_model[0].statistics, result.overall.statistics):
        assert statistics.ci_low is None
        assert statistics.ci_high is None
    assert result.overall.statistics.delta == control.overall.statistics.delta == pytest.approx(15)
    assert result.overall.statistics.variance == control.overall.statistics.variance
    assert result.overall.weights == control.overall.weights == [0.5, 0.5]


def test_differing_repeat_plans_suppress_intervals_with_partial_recorded_identities(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "different_repeats", _values(0.2, 0.4), _values(0.4, 0.6))

    def without_rollout_identities(entry):
        for group in entry["group_level_metrics"]:
            group.pop("rollout_infos")

    for source in pair.values():
        _edit_entry(source["rollouts"], without_rollout_identities)
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={"model": {"test": pair}},
    )
    control = _compare(path)
    assert control.cells[0].baseline.coverage.status == control.cells[0].candidate.coverage.status == "unverified"
    assert control.overall.statistics.ci_low is not None
    for source in pair.values():
        inventory = materialized_path_for(Path(source["rollouts"]))
        rows = [
            row
            for line in inventory.read_bytes().splitlines()
            if (row := orjson.loads(line))["_ng_task_index"] != 1 or row["_ng_rollout_index"] != 0
        ]
        inventory.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    result = _compare(path)
    cell = result.cells[0]
    note = "Planned task sets differ between repeats."
    assert cell.baseline.coverage.status == cell.candidate.coverage.status == "incomplete"
    assert cell.baseline.coverage.recorded_task_count == cell.baseline.coverage.planned_task_count == 2
    for role in ("baseline", "candidate"):
        run = getattr(cell, role)
        assert set(run.coverage.notes) == set(getattr(control.cells[0], role).coverage.notes) | {note}
        assert f"model/test {role}: {note}" in result.warnings
    assert not any("Recorded task sets differ between repeats." in warning for warning in result.warnings)
    assert not any("Baseline and candidate planned task sets differ." in warning for warning in result.warnings)
    assert cell.baseline.repeat_scores == control.cells[0].baseline.repeat_scores == pytest.approx([20, 40])
    assert cell.candidate.repeat_scores == control.cells[0].candidate.repeat_scores == pytest.approx([40, 60])
    assert cell.statistics.delta == pytest.approx(20)
    assert cell.model_weight == cell.overall_weight == 1
    for statistics in (cell.statistics, result.per_model[0].statistics, result.overall.statistics):
        assert statistics.ci_low is None
        assert statistics.ci_high is None
    assert result.overall.statistics.variance == control.overall.statistics.variance
    assert result.overall.weights == control.overall.weights == [1]


@pytest.mark.parametrize(
    "metrics",
    [
        [{"name": "mean/reward", "low": 0}],
        [_scale(high=0)],
        [_scale(), _scale()],
    ],
)
def test_invalid_metric_definitions_are_actionable_errors(tmp_path: Path, metrics: list[dict]) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": metrics}},
        models={"model": {"test": _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))}},
    )
    with pytest.raises(ConfigError, match="require low, high|greater than low|same metric more than once"):
        _compare(path)


def test_model_with_all_sources_excluded_is_retained_without_statistics(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={
            "excluded": {"test": _pair(tmp_path, "single", _values(0.1), _values(0.3))},
            "included": {"test": _pair(tmp_path, "repeated", _values(0.2, 0.4), _values(0.4, 0.6))},
        },
    )
    result = _compare(path)
    excluded = next(model for model in result.per_model if model.name == "excluded")
    assert excluded.statistics is None
    assert excluded.cell_ids == []
    assert excluded.weights == []
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.weights == [1]


@pytest.mark.parametrize("evidence", ["sample_counts", "reported_repeats", "missing_group_rollouts"])
def test_incomplete_coverage_evidence_without_rollout_identities_suppresses_ci(tmp_path: Path, evidence: str) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))

    def incomplete(entry):
        for group in entry["group_level_metrics"]:
            group.pop("rollout_infos")
        if evidence == "sample_counts":
            entry["repeat_level_metrics"][1]["sample_count"] = 1
        elif evidence == "reported_repeats":
            entry["agent_metrics"]["num_repeats"] = 3
        else:
            entry["group_level_metrics"][0]["missing_num_rollouts"] = 1

    _edit_entry(pair["baseline"]["rollouts"], incomplete)
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    result = _compare(path)
    assert result.cells[0].baseline.coverage.status == "incomplete"
    assert result.overall.statistics.ci_low is None
    assert result.overall.statistics.delta == pytest.approx(20)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_processing_rejects_nonfinite_repeat_scores_with_cell_context(tmp_path: Path, value: float) -> None:
    # Standard JSON cannot encode these values; exercise the processing boundary directly.
    source = _write_run(tmp_path / "baseline", _values(0.2, 0.4))
    baseline = load_suite_run(SuiteRunSource(rollouts=Path(source)), directory=tmp_path)
    candidate_source = _write_run(tmp_path / "candidate", _values(0.4, 0.6))
    candidate = load_suite_run(SuiteRunSource(rollouts=Path(candidate_source)), directory=tmp_path)
    baseline = replace(
        baseline, repeat_rows=(dict(baseline.repeat_rows[0]) | {"mean/reward": value}, baseline.repeat_rows[1])
    )
    with pytest.raises(ConfigError, match="Cannot normalize 'model/test'.*must be finite"):
        compare_suite_cell(
            model="model",
            benchmark="test",
            baseline=baseline,
            candidate=candidate,
            metrics=[
                SuiteMetricResult(name="mean/reward", low=0, high=1, higher_is_better=True, definition_source="test")
            ],
            confidence_level=0.9,
        )


def test_unknown_recorded_task_coverage_does_not_claim_verification(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))

    def without_identities(entry):
        for group in entry["group_level_metrics"]:
            group.pop("rollout_infos")

    _edit_entry(pair["baseline"]["rollouts"], without_identities)
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    result = _compare(path)
    assert result.cells[0].baseline.coverage.status == "unverified"
    assert result.cells[0].baseline.coverage.planned_task_count == 2
    assert result.overall.statistics.ci_low is not None
    assert any("recorded task identities are insufficient" in warning for warning in result.warnings)


def test_duplicate_source_across_cells_is_rejected(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={"model-a": {"test": pair}, "model-b": {"test": pair}},
    )
    with pytest.raises(ConfigError, match="reuses the same aggregate file and agent.*independent source"):
        _compare(path)


def test_recorded_group_task_mismatch_without_rollout_infos_suppresses_ci(tmp_path: Path) -> None:
    pair = {
        "baseline": {"rollouts": _write_run(tmp_path / "baseline", _values(0.2, 0.4), write_inputs=False)},
        "candidate": {
            "rollouts": _write_run(tmp_path / "candidate", _values(0.4, 0.6), tasks=(9, 10), write_inputs=False)
        },
    }

    def without_identities(entry):
        for group in entry["group_level_metrics"]:
            group.pop("rollout_infos")

    for source in pair.values():
        _edit_entry(source["rollouts"], without_identities)
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    result = _compare(path)
    assert result.cells[0].baseline.coverage.status == "incomplete"
    assert result.cells[0].candidate.coverage.status == "incomplete"
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.statistics.ci_low is None
    assert any("Candidate and baseline have results for different tasks" in warning for warning in result.warnings)


def test_recorded_group_subset_of_planned_tasks_without_rollout_infos_suppresses_ci(tmp_path: Path) -> None:
    pair = {
        "baseline": {
            "rollouts": _write_run(tmp_path / "baseline", _values(0.2, 0.4), tasks=(0,), planned_tasks=(0, 1))
        },
        "candidate": {
            "rollouts": _write_run(tmp_path / "candidate", _values(0.4, 0.6), tasks=(0,), planned_tasks=(0, 1))
        },
    }

    def without_identities(entry):
        for group in entry["group_level_metrics"]:
            group.pop("rollout_infos")

    for source in pair.values():
        _edit_entry(source["rollouts"], without_identities)
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    result = _compare(path)
    assert result.cells[0].baseline.coverage.recorded_task_count == 1
    assert result.cells[0].baseline.coverage.planned_task_count == 2
    assert result.cells[0].baseline.coverage.status == "incomplete"
    assert result.overall.statistics.ci_low is None
    assert "Results are missing 1 of 2 planned tasks (task indices: 1)." in result.cells[0].baseline.coverage.notes


def test_distinct_rollouts_cannot_reuse_physical_aggregate_companion(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    baseline_metrics = aggregate_metrics_path_for(Path(pair["baseline"]["rollouts"]))
    candidate_metrics = aggregate_metrics_path_for(Path(pair["candidate"]["rollouts"]))
    candidate_metrics.unlink()
    candidate_metrics.symlink_to(baseline_metrics)
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    with pytest.raises(ConfigError, match="reuses the same aggregate file and agent.*independent source"):
        _compare(path)
