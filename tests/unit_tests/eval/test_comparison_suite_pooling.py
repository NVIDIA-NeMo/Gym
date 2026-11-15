# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pooled suite repeats against independent numerical and artifact evidence."""

from pathlib import Path
from statistics import mean, variance

import orjson
import pytest
from scipy.stats import ttest_ind

from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.suite.report import render_suite_markdown, write_suite_reports
from nemo_gym.eval.comparison.suite.schema import SuiteComparisonResult
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, materialized_path_for
from tests.unit_tests.eval.test_comparison_suite import (
    _compare,
    _edit_entry,
    _manifest,
    _scale,
    _values,
    _write_routed_run,
)
from tests.unit_tests.eval.test_comparison_suite_failures import _coverage


def _source(
    directory: Path,
    scores: list[float],
    *,
    tasks: tuple[int, ...] = (0, 1),
    repeat_offset: int = 0,
    agent: str = "agent",
    server: str = "benchmark",
) -> dict[str, str]:
    rollouts = _write_routed_run(directory, [(server, agent, tasks, _values(*scores))])
    if repeat_offset:

        def offset(entry):
            for row in entry["repeat_level_metrics"]:
                row["_ng_rollout_index"] += repeat_offset
            for group in entry["group_level_metrics"]:
                for row in group["rollout_infos"]:
                    row["_ng_rollout_index"] += repeat_offset

        _edit_entry(rollouts, offset)
        path = materialized_path_for(Path(rollouts))
        rows = [orjson.loads(line) for line in path.read_bytes().splitlines()]
        for row in rows:
            row["_ng_rollout_index"] += repeat_offset
        path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    return {"rollouts": rollouts, "agent": agent, "environment_server": server}


def _suite(tmp_path: Path, baseline: list[dict[str, str]], candidate: list[dict[str, str]]) -> Path:
    return _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale()]}},
        models={"model": {"test": {"baseline": baseline, "candidate": candidate}}},
    )


@pytest.mark.parametrize("counts", [((8, 8), (8, 8, 8)), ((2, 5), (2, 3, 4))])
def test_pooled_repeats_match_scipy_welch_without_averaging_run_means(
    tmp_path: Path, counts: tuple[tuple[int, ...], tuple[int, ...]]
) -> None:
    baseline_values = [
        [0.1 + 0.1 * run + 0.02 * repeat for repeat in range(count)] for run, count in enumerate(counts[0])
    ]
    candidate_values = [
        [0.3 + 0.1 * run + 0.01 * repeat for repeat in range(count)] for run, count in enumerate(counts[1])
    ]
    baseline = [_source(tmp_path / f"baseline-{index}", scores) for index, scores in enumerate(baseline_values)]
    candidate = [
        _source(tmp_path / f"candidate-{index}", scores, repeat_offset=10 if index else 0)
        for index, scores in enumerate(candidate_values)
    ]
    path = _suite(tmp_path, baseline, candidate)
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    result = _compare(path)

    cell = result.cells[0]
    b = [100 * score for scores in baseline_values for score in scores]
    c = [100 * score for scores in candidate_values for score in scores]
    oracle = ttest_ind(c, b, equal_var=False)
    interval = oracle.confidence_interval(confidence_level=0.9)
    assert cell.baseline.repeat_count == cell.baseline.summary.count == len(b)
    assert cell.candidate.repeat_count == cell.candidate.summary.count == len(c)
    assert cell.baseline.repeat_scores == pytest.approx(b)
    assert cell.candidate.repeat_scores == pytest.approx(c)
    assert cell.baseline.summary.mean == pytest.approx(mean(b))
    assert cell.baseline.summary.sample_variance == pytest.approx(variance(b))
    assert cell.candidate.summary.sample_variance == pytest.approx(variance(c))
    assert cell.statistics.delta == pytest.approx(mean(c) - mean(b))
    assert cell.statistics.degrees_of_freedom == pytest.approx(oracle.df)
    assert (cell.statistics.ci_low, cell.statistics.ci_high) == pytest.approx((interval.low, interval.high))
    assert result.overall.statistics == cell.statistics
    assert cell.baseline.repeat_indices == cell.candidate.repeat_indices == []
    assert cell.baseline.sources[0].repeat_indices == list(range(counts[0][0]))
    assert cell.baseline.sources[1].repeat_indices == list(range(counts[0][1]))
    assert cell.candidate.sources[1].repeat_indices == list(range(10, 10 + counts[1][1]))
    assert cell.metrics[0].baseline_raw == pytest.approx([score for scores in baseline_values for score in scores])
    assert cell.baseline.coverage.status == cell.candidate.coverage.status == "verified"
    assert result.warnings == []
    assert {file: file.read_bytes() for file in before} == before
    if counts[0][0] != counts[0][1]:
        assert cell.baseline.summary.mean != pytest.approx(mean([100 * mean(scores) for scores in baseline_values]))

    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result
    markdown = render_suite_markdown(restored)
    assert f"Baseline: 2 pooled source runs; {len(b)} total repeats" in markdown
    assert f"Candidate: 3 pooled source runs; {len(c)} total repeats" in markdown
    for side in (baseline, candidate):
        assert all(str(aggregate_metrics_path_for(Path(source["rollouts"]))) in markdown for source in side)


@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_empty_source_collection_is_rejected(tmp_path: Path, role: str) -> None:
    baseline = [_source(tmp_path / "baseline", [0.2, 0.4])]
    candidate = [_source(tmp_path / "candidate", [0.4, 0.6])]
    path = _suite(tmp_path, [] if role == "baseline" else baseline, [] if role == "candidate" else candidate)
    with pytest.raises(ConfigError, match="List should have at least 1 item"):
        _compare(path)


@pytest.mark.parametrize("across_sides", [False, True])
def test_duplicate_aggregate_aliases_cannot_add_repeat_observations(tmp_path: Path, across_sides: bool) -> None:
    baseline = _source(tmp_path / "baseline", [0.2, 0.4])
    candidate = _source(tmp_path / "candidate", [0.4, 0.6])
    duplicated = _source(tmp_path / "duplicated", [0.4, 0.6])
    alias = aggregate_metrics_path_for(Path(duplicated["rollouts"]))
    alias.unlink()
    alias.symlink_to(aggregate_metrics_path_for(Path(baseline["rollouts"])))
    path = _suite(
        tmp_path, [baseline] if across_sides else [baseline, duplicated], [duplicated] if across_sides else [candidate]
    )
    with pytest.raises(ConfigError, match="reuses the same aggregate file and agent.*independent source"):
        _compare(path)


def test_every_source_can_select_a_different_explicit_agent_and_server(tmp_path: Path) -> None:
    baseline = [
        _source(tmp_path / "baseline-1", [0.1, 0.3], agent="first", server="first-server"),
        _source(tmp_path / "baseline-2", [0.2, 0.4], agent="second", server="second-server"),
    ]
    candidate = [_source(tmp_path / "candidate", [0.4, 0.6])]
    cell = _compare(_suite(tmp_path, baseline, candidate)).cells[0]
    assert cell.baseline.repeat_scores == pytest.approx([10, 30, 20, 40])
    assert [source.agent for source in cell.baseline.sources] == ["first", "second"]
    assert [source.environment_server for source in cell.baseline.sources] == ["first-server", "second-server"]
    assert cell.statistics.delta == pytest.approx(25)
    assert cell.statistics.ci_low is not None


@pytest.mark.parametrize("evidence", ["masked", "missing", "unscored", "recorded_tasks", "planned_tasks"])
def test_incomplete_later_source_suppresses_all_dependent_intervals(tmp_path: Path, evidence: str) -> None:
    baseline = [_source(tmp_path / "baseline-1", [0.1, 0.3]), _source(tmp_path / "baseline-2", [0.2, 0.4])]
    candidate = [_source(tmp_path / "candidate", [0.4, 0.6])]
    later = baseline[1]["rollouts"]
    if evidence == "recorded_tasks":
        baseline[1] = _source(tmp_path / "other-taskset", [0.2, 0.4], tasks=(2, 3))
    elif evidence == "planned_tasks":
        inventory = materialized_path_for(Path(later))
        rows = [orjson.loads(line) for line in inventory.read_bytes().splitlines()]
        for row in rows:
            row["_ng_task_index"] += 10
        inventory.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        # Leave only planned identities to detect a mismatch across independent source plans.
        for source in (*baseline, *candidate):
            _edit_entry(source["rollouts"], lambda entry: entry.update({"group_level_metrics": []}))
    else:

        def incomplete(entry):
            if evidence == "masked":
                entry["agent_metrics"]["coverage/masked_rollouts"] = 1
            elif evidence == "missing":
                entry["repeat_level_metrics"][1]["missing_count"] = 1
            else:
                entry["repeat_level_metrics"][1] = {"_ng_rollout_index": 1, "mean_completion_tokens": 10}
                for group in entry["group_level_metrics"]:
                    group["rollout_infos"].pop()

        _edit_entry(later, incomplete)
    result = _compare(_suite(tmp_path, baseline, candidate))
    cell = result.cells[0]
    assert cell.baseline.coverage.status == "incomplete"
    expected_source_status = (
        "verified" if evidence == "recorded_tasks" else "unverified" if evidence == "planned_tasks" else "incomplete"
    )
    assert cell.baseline.sources[1].coverage.status == expected_source_status
    assert any("source 2" in warning.lower() for warning in result.warnings)
    for statistics in (cell.statistics, result.per_model[0].statistics, result.overall.statistics):
        assert statistics.ci_low is statistics.ci_high is None
        assert statistics.interval_unavailable_reason == "Known incomplete task coverage in an included benchmark."
    if evidence == "unscored":
        assert cell.baseline.sources[1].unscored_repeat_indices == [1]
        assert cell.baseline.repeat_scores == pytest.approx([10, 30, 20])


def test_single_repeat_sources_require_saved_repeat_metrics(tmp_path: Path) -> None:
    baseline = [_source(tmp_path / "baseline-1", [0.2]), _source(tmp_path / "baseline-2", [0.4])]
    candidate = [_source(tmp_path / "candidate", [0.4, 0.6])]
    for source, reward in zip(baseline, (0.2, 0.4), strict=True):
        produced = compute_aggregate_metrics(
            [{"_ng_task_index": task, "_ng_rollout_index": 0, "reward": reward} for task in (0, 1)]
        ).model_dump(mode="json")
        _edit_entry(source["rollouts"], lambda entry: entry.update(produced))
    with pytest.raises(ConfigError, match="baseline-1.*no stored repeat rows.*Pooling requires stored metrics"):
        _compare(_suite(tmp_path, baseline, candidate))


@pytest.mark.parametrize(
    ("version", "total", "malformed"), [(1, 2, False), (2, 2, False), (2, 0, False), (2, 2, True)]
)
def test_later_source_saved_failures_preserve_statistics_and_source_provenance(
    tmp_path: Path, version: int, total: int, malformed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [_source(tmp_path / "baseline-1", [0.1, 0.3]), _source(tmp_path / "baseline-2", [0.2, 0.4])]
    candidate = [_source(tmp_path / "candidate-1", [0.4, 0.6]), _source(tmp_path / "candidate-2", [0.5, 0.7])]
    path = _suite(tmp_path, baseline, candidate)
    control = _compare(path)
    zero = _coverage(Path(baseline[0]["rollouts"]), total=0)
    source = _coverage(Path(candidate[1]["rollouts"]), version=version, total=total)
    if malformed:
        source.write_bytes(orjson.dumps(orjson.loads(source.read_bytes()) | {"failed": True}))
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    rollouts = {Path(item["rollouts"]) for item in (*baseline, *candidate)}
    opened_rollouts = []
    open_file = Path.open

    def record_open(file: Path, *args, **kwargs):
        if file in rollouts:
            opened_rollouts.append(file)
        return open_file(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", record_open)
        result = _compare(path)

    assert opened_rollouts == []
    cell = result.cells[0]
    assert cell.baseline.failures is cell.candidate.failures is None
    assert cell.baseline.sources[0].failures.source == zero
    assert cell.baseline.sources[0].failures.counts.total == 0
    assert cell.baseline.sources[1].failures is cell.candidate.sources[0].failures is None
    failures = cell.candidate.sources[1].failures
    assert failures.source == source
    assert failures.scope == "rollout_file"
    if malformed:
        assert failures.counts is None
        assert "at failed: Input should be a valid integer" in failures.unavailable_reason
        assert len(result.warnings) == 1
        assert "candidate source 2: Saved failures (entire rollout file): unavailable" in result.warnings[0]
    else:
        assert failures.counts.total == total
        assert failures.counts.by_class == ({"agent_run_error": total} if total else {})
        assert failures.counts.counted_as_zero == (1 if total else 0)
        assert failures.counts.final_zeros is None
        assert failures.unavailable_reason is None
        assert result.warnings == (
            [
                "model/test candidate source 2: Saved failures (entire rollout file): 2 "
                "(1 scored as zero, 1 not in the score; agent_run_error=2)."
            ]
            if total
            else []
        )
    assert result.overall == control.overall
    assert result.per_model == control.per_model
    assert cell.statistics == control.cells[0].statistics
    assert cell.metrics == control.cells[0].metrics
    assert (cell.eligible, cell.model_weight, cell.overall_weight) == (
        control.cells[0].eligible,
        control.cells[0].model_weight,
        control.cells[0].overall_weight,
    )
    for role in ("baseline", "candidate"):
        previous, current = getattr(control.cells[0], role), getattr(cell, role)
        assert current.repeat_scores == previous.repeat_scores
        assert current.summary == previous.summary
        assert current.coverage == previous.coverage
    assert cell.statistics.ci_low is not None
    assert {file: file.read_bytes() for file in before} == before
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result
    output = tmp_path / "reports"
    write_suite_reports(restored, output, "both")
    payload = orjson.loads((output / "suite_compare_report.json").read_bytes())
    assert payload["cells"][0]["candidate"]["failures"] is None
    assert payload["cells"][0]["candidate"]["sources"][1]["failures"]["source"] == str(source)
    markdown = (output / "suite_compare_report.md").read_text()
    assert str(source) in markdown
    assert str(zero) in markdown
    assert {file: file.read_bytes() for file in before} == before


def test_shared_pooled_source_keeps_one_failure_snapshot_and_groups_warning_usages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [_source(tmp_path / "baseline-1", [0.1, 0.3]), _source(tmp_path / "baseline-2", [0.2, 0.4])]
    shared = _write_routed_run(
        tmp_path / "candidate",
        [
            ("first_env", "first_agent", (0, 1), _values(0.4, 0.6)),
            ("second_env", "second_agent", (0, 1), _values(0.5, 0.7)),
        ],
    )
    candidate = [
        {"rollouts": shared, "agent": agent, "environment_server": server}
        for server, agent in (("first_env", "first_agent"), ("second_env", "second_agent"))
    ]
    path = _suite(tmp_path, baseline, candidate)
    control = _compare(path)
    snapshot = _coverage(Path(shared))
    payload = orjson.loads(snapshot.read_bytes()) | {"final_zeros": 1}
    snapshot.write_bytes(orjson.dumps(payload))
    updated = payload | {
        "failed": 0,
        "failed_final": 0,
        "failed_not_final": 0,
        "failed_by_class": {},
        "failures_counted_as_zero": 0,
        "final_zeros": 0,
    }
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file() and file != snapshot}
    read_bytes = Path.read_bytes

    def read_while_snapshot_is_replaced(file: Path) -> bytes:
        saved = read_bytes(file)
        if file == snapshot:
            replacement = snapshot.with_suffix(".next.json")
            replacement.write_bytes(orjson.dumps(updated))
            replacement.replace(snapshot)
        return saved

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_while_snapshot_is_replaced)
        result = _compare(path)

    assert orjson.loads(snapshot.read_bytes())["failed"] == 0
    assert result.warnings == [
        "model/test candidate source 1, model/test candidate source 2: Saved failures (entire rollout file): 2 "
        "(1 scored as zero, 1 not in the score; agent_run_error=2)."
    ]
    cell = result.cells[0]
    assert cell.candidate.failures is None
    summaries = [source.failures for source in cell.candidate.sources]
    assert len(summaries) == 2
    assert all(summary.source == snapshot and summary.scope == "rollout_file" for summary in summaries)
    assert all(summary.counts.total == 2 for summary in summaries)
    assert all(summary.counts.counted_as_zero == summary.counts.final_zeros == 1 for summary in summaries)
    assert result.overall == control.overall
    assert result.per_model == control.per_model
    assert cell.statistics == control.cells[0].statistics
    assert cell.statistics.ci_low is not None
    assert cell.candidate.coverage == control.cells[0].candidate.coverage
    assert {file: file.read_bytes() for file in before} == before
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert len(restored.warnings) == 1
    assert all(source.failures.counts.final_zeros == 1 for source in restored.cells[0].candidate.sources)


@pytest.mark.parametrize("artifact", ["aggregate", "coverage"])
def test_report_writer_protects_later_pooled_source_artifacts(tmp_path: Path, artifact: str) -> None:
    baseline = [_source(tmp_path / "baseline-1", [0.1, 0.3]), _source(tmp_path / "baseline-2", [0.2, 0.4])]
    candidate = [_source(tmp_path / "candidate", [0.4, 0.6])]
    later = Path(baseline[1]["rollouts"])
    source = aggregate_metrics_path_for(later) if artifact == "aggregate" else _coverage(later)
    result = _compare(_suite(tmp_path, baseline, candidate))
    report_target = tmp_path / "suite_compare_report.json"
    report_target.symlink_to(source)
    before = source.read_bytes()
    with pytest.raises(ConfigError, match="would overwrite a source artifact"):
        write_suite_reports(result, tmp_path, "both")
    assert source.read_bytes() == before
    assert not (tmp_path / "suite_compare_report.md").exists()


def test_coverage_compares_available_recorded_and_planned_sets_across_sources(tmp_path: Path) -> None:
    baseline = [
        _source(tmp_path / "baseline-1", [0.1, 0.3]),
        _source(tmp_path / "baseline-2", [0.2, 0.4], tasks=(2, 3)),
    ]
    candidate = [_source(tmp_path / "candidate", [0.4, 0.6])]
    # One source supplies only recorded identities; another supplies only planned identities.
    materialized_path_for(Path(baseline[0]["rollouts"])).unlink()
    materialized_path_for(Path(candidate[0]["rollouts"])).unlink()
    _edit_entry(baseline[1]["rollouts"], lambda entry: entry.update({"group_level_metrics": []}))

    result = _compare(_suite(tmp_path, baseline, candidate))

    cell = result.cells[0]
    assert cell.baseline.coverage.status == "incomplete"
    assert cell.statistics.delta == pytest.approx(25)
    assert cell.statistics.ci_low is cell.statistics.ci_high is None
    assert any(
        "Recorded and planned task sets differ across pooled source runs" in warning for warning in result.warnings
    )


@pytest.mark.parametrize("candidate_sources", [1, 3])
def test_collection_mismatch_preserves_source_evidence_and_reports_each_problem_once(
    tmp_path: Path, candidate_sources: int
) -> None:
    baseline = [
        _source(tmp_path / "baseline-1", [0.1, 0.3]),
        _source(tmp_path / "baseline-2", [0.2, 0.4], tasks=(5, 6)),
    ]
    candidate = [_source(tmp_path / f"candidate-{index}", [0.4, 0.6]) for index in range(candidate_sources)]

    result = _compare(_suite(tmp_path, baseline, candidate))

    cell = result.cells[0]
    mismatch = str(aggregate_metrics_path_for(Path(baseline[1]["rollouts"])))
    assert len(result.warnings) == 4
    for side in (cell.baseline, cell.candidate):
        assert side.coverage.status == "incomplete"
        assert len(side.coverage.notes) == 2
        assert all("baseline source 2" in note and mismatch in note for note in side.coverage.notes)
        assert all(source.coverage.status == "verified" for source in side.sources)
        assert all(source.coverage.notes == [] for source in side.sources)
    for statistics in (cell.statistics, result.per_model[0].statistics, result.overall.statistics):
        assert statistics.ci_low is statistics.ci_high is None
        assert statistics.interval_unavailable_reason == "Known incomplete task coverage in an included benchmark."


def test_selected_metrics_are_normalized_within_each_pooled_repeat(tmp_path: Path) -> None:
    baseline = [_source(tmp_path / "baseline-1", [0.1, 0.3]), _source(tmp_path / "baseline-2", [0.2, 0.4])]
    candidate = [_source(tmp_path / "candidate-1", [0.4, 0.5]), _source(tmp_path / "candidate-2", [0.6, 0.8])]
    for source, latencies in zip((*baseline, *candidate), ([90, 50], [70, 30], [80, 60], [20, 40]), strict=True):

        def add_latency(entry):
            for row, latency in zip(entry["repeat_level_metrics"], latencies, strict=True):
                row["latency"] = latency

        _edit_entry(source["rollouts"], add_latency)
    path = _manifest(
        tmp_path,
        benchmarks={"test": {"metrics": [_scale(), _scale("latency", high=100, higher=False)]}},
        models={"model": {"test": {"baseline": baseline, "candidate": candidate}}},
    )

    cell = _compare(path).cells[0]

    baseline_scores = [10, 40, 25, 55]
    candidate_scores = [30, 45, 70, 70]
    oracle = ttest_ind(candidate_scores, baseline_scores, equal_var=False)
    interval = oracle.confidence_interval(confidence_level=0.9)
    assert cell.baseline.repeat_scores == pytest.approx(baseline_scores)
    assert cell.candidate.repeat_scores == pytest.approx(candidate_scores)
    assert cell.baseline.sources[0].repeat_scores == pytest.approx([10, 40])
    assert cell.baseline.sources[1].repeat_scores == pytest.approx([25, 55])
    assert cell.metrics[1].baseline_raw == [90, 50, 70, 30]
    assert cell.metrics[1].baseline_scaled == pytest.approx([10, 50, 30, 70])
    assert cell.metrics[1].candidate_scaled == pytest.approx([20, 40, 80, 60])
    assert cell.statistics.delta == pytest.approx(mean(candidate_scores) - mean(baseline_scores))
    assert (cell.statistics.ci_low, cell.statistics.ci_high) == pytest.approx((interval.low, interval.high))
