# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved failure accounting without recomputing outcomes or trajectory health."""

from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError

from nemo_gym.cli.main import main
from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.suite.failures import read_saved_failures, saved_failure_warning
from nemo_gym.eval.comparison.suite.report import render_suite_markdown, write_suite_reports
from nemo_gym.eval.comparison.suite.schema import SuiteComparisonResult, SuiteFailureSummary
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.rollouts.rollout_collection import _counted_failure_rows
from nemo_gym.rollouts.rollout_records import coverage_path_for
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, failures_path_for
from tests.unit_tests.eval.test_comparison_suite import (
    _compare,
    _manifest,
    _pair,
    _scale,
    _values,
    _write_routed_run,
)


def _suite(tmp_path: Path) -> tuple[Path, Path, Path]:
    pair = _pair(tmp_path, "run", _values(0.2, 0.4), _values(0.4, 0.6))
    manifest = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    return manifest, Path(pair["baseline"]["rollouts"]), Path(pair["candidate"]["rollouts"])


def _coverage(rollouts: Path, *, version: int = 2, total: int = 2) -> Path:
    path = coverage_path_for(rollouts)
    path.write_bytes(
        orjson.dumps(
            {
                "schema_version": version,
                "failed": total,
                "failed_final": 1 if total else 0,
                "failed_not_final": total - 1 if total else 0,
                "failed_by_class": {"agent_run_error": total} if total else {},
                "expected": 4,
                "successful": 2,
                "failures_counted_as_zero": 1 if total else 0,
                "imputed": 0,
            }
        )
    )
    return path


def _unchanged_statistics(before: SuiteComparisonResult, after: SuiteComparisonResult) -> None:
    assert after.overall == before.overall
    assert after.per_model == before.per_model
    for previous, current in zip(before.cells, after.cells, strict=True):
        assert current.statistics == previous.statistics
        assert current.metrics == previous.metrics
        assert (current.eligible, current.model_weight, current.overall_weight) == (
            previous.eligible,
            previous.model_weight,
            previous.overall_weight,
        )
        for role in ("baseline", "candidate"):
            old_run, new_run = getattr(previous, role), getattr(current, role)
            assert new_run.repeat_scores == old_run.repeat_scores
            assert new_run.summary == old_run.summary
            assert new_run.coverage == old_run.coverage


def _scored_store(
    directory: Path,
    *,
    final_zeros: set[tuple[int, int]],
    other_failures: set[tuple[int, int]],
    counted_classes: list[str],
) -> tuple[RolloutStore, int, int]:
    directory.mkdir()
    output = directory / "rollouts.jsonl"
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": repeat, "agent_ref": {"name": "agent"}}
        for task in (0, 1)
        for repeat in (0, 1)
    ]
    source = directory / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    manifest = RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})
    with RolloutStore.start_or_resume(output, lambda: (rows, manifest), resume=False) as store:
        for row in store.pending():
            store.record_dispatch(row)
            key = row["_ng_task_index"], row["_ng_rollout_index"]
            if key in final_zeros or key in other_failures:
                store.record_outcome(
                    row
                    | {
                        "_ng_failure_class": "agent_run_error",
                        "_ng_failure_terminal": key in final_zeros,
                        "_ng_failure_counts_as_zero": key in final_zeros,
                    }
                )
            else:
                store.record_outcome(row | {"reward": 0.4 + 0.2 * row["_ng_rollout_index"], "response": {}})
        counted = _counted_failure_rows(store.failures(), counted_classes)
        scored = store.selected("success") + counted
    entry = {
        "agent_ref": {"name": "agent"},
        **compute_aggregate_metrics(scored).model_dump(mode="json", exclude_none=True),
    }
    aggregate_metrics_path_for(output).write_bytes(orjson.dumps([entry]))
    return store, len(counted), len(scored)


@pytest.mark.parametrize("version", [1, 2])
def test_saved_failures_warn_without_changing_statistics_or_reading_rollouts(
    tmp_path: Path, version: int, monkeypatch
) -> None:
    path, baseline, candidate = _suite(tmp_path)
    control = _compare(path)
    source = _coverage(candidate, version=version)
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    opened_rollouts = []
    open_file = Path.open

    def record_open(path: Path, *args, **kwargs):
        if path in (baseline, candidate):
            opened_rollouts.append(path)
        return open_file(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", record_open)
        result = _compare(path)

    assert opened_rollouts == []
    summary = result.cells[0].candidate.failures
    assert summary.source == source
    assert summary.scope == "rollout_file"
    assert summary.counts.model_dump() == {
        "total": 2,
        "final": 1,
        "not_final": 1,
        "by_class": {"agent_run_error": 2},
        "counted_as_zero": 1,
        "final_zeros": None,
    }
    assert summary.unavailable_reason is None
    assert result.cells[0].baseline.failures is None
    assert result.warnings == [
        "model/test candidate: Saved failures (entire rollout file): 2 "
        "(1 scored as zero, 1 not in the score; agent_run_error=2)."
    ]
    _unchanged_statistics(control, result)
    assert result.overall.statistics.ci_low is not None
    assert baseline.read_text() == candidate.read_text() == "trajectory contents are deliberately unavailable\n"
    assert {file: file.read_bytes() for file in sources} == sources
    write_suite_reports(result, tmp_path / "reports", "both")
    saved = orjson.loads((tmp_path / "reports/suite_compare_report.json").read_bytes())
    assert saved["cells"][0]["candidate"]["failures"]["counts"]["total"] == 2
    assert saved["cells"][0]["baseline"]["failures"] is None
    markdown = (tmp_path / "reports/suite_compare_report.md").read_text()
    assert "## Warnings" in markdown
    assert result.warnings[0] in markdown
    assert str(source) in markdown


def test_missing_accounting_is_unknown_and_saved_zero_counts_remain_quiet(tmp_path: Path) -> None:
    path, baseline, candidate = _suite(tmp_path)
    source = _coverage(baseline, total=0)

    result = _compare(path)

    assert result.cells[0].baseline.failures.source == source
    assert result.cells[0].baseline.failures.counts.total == 0
    assert result.cells[0].baseline.failures.counts.by_class == {}
    assert result.cells[0].candidate.failures is None
    assert not coverage_path_for(candidate).exists()
    assert result.warnings == []
    assert "## Warnings" not in render_suite_markdown(result)


@pytest.mark.parametrize("payload", [{"schema_version": 1}, {"schema_version": 1, "failed": None}])
def test_unrecorded_counts_are_reported_as_unavailable_not_zero(tmp_path: Path, payload: dict) -> None:
    path, _, candidate = _suite(tmp_path)
    control = _compare(path)
    source = coverage_path_for(candidate)
    source.write_bytes(orjson.dumps(payload))

    result = _compare(path)

    assert result.cells[0].candidate.failures.source == source
    assert result.cells[0].candidate.failures.counts is None
    assert "failure counts were not recorded" in result.cells[0].candidate.failures.unavailable_reason
    assert len(result.warnings) == 1
    assert "Saved failures (entire rollout file): unavailable" in result.warnings[0]
    _unchanged_statistics(control, result)


@pytest.mark.parametrize(
    ("change", "expected_reason"),
    [
        (
            {"failed": True, "failed_final": 1, "failed_not_final": 0, "failed_by_class": {"agent_run_error": 1}},
            "at failed: Input should be a valid integer",
        ),
        (
            {"failed_by_class": {"agent_run_error": 3, "timeout": -1}},
            "at failed_by_class.timeout: Input should be greater than or equal to 0",
        ),
        ({"failed_final": 2}, "Final and not-final failure counts must sum to the total."),
        ({"failed_by_class": {"agent_run_error": 1}}, "Failure class counts must sum to the total."),
        ({"schema_version": 3}, "unsupported coverage schema version 3"),
    ],
)
def test_invalid_saved_counts_remain_advisory(tmp_path: Path, change: dict, expected_reason: str) -> None:
    path, _, candidate = _suite(tmp_path)
    control = _compare(path)
    source = _coverage(candidate)
    source.write_bytes(orjson.dumps(orjson.loads(source.read_bytes()) | change))
    original = source.read_bytes()

    result = _compare(path)

    assert result.cells[0].candidate.failures.counts is None
    assert expected_reason in result.cells[0].candidate.failures.unavailable_reason
    assert len(result.warnings) == 1
    assert "Saved failures (entire rollout file): unavailable" in result.warnings[0]
    assert expected_reason in result.warnings[0]
    assert "\n" not in result.warnings[0]
    _unchanged_statistics(control, result)
    write_suite_reports(result, tmp_path / "reports", "both")
    assert source.read_bytes() == original
    assert (tmp_path / "reports/suite_compare_report.md").is_file()
    assert (tmp_path / "reports/suite_compare_report.json").is_file()


@pytest.mark.parametrize("directory", [False, True])
def test_unreadable_accounting_remains_advisory(tmp_path: Path, directory: bool) -> None:
    path, _, candidate = _suite(tmp_path)
    control = _compare(path)
    source = coverage_path_for(candidate)
    if directory:
        source.mkdir()
    else:
        source.write_bytes(b"not JSON\n")

    result = _compare(path)

    assert result.cells[0].candidate.failures.source == source
    assert result.cells[0].candidate.failures.counts is None
    assert result.cells[0].candidate.failures.unavailable_reason
    assert len(result.warnings) == 1
    _unchanged_statistics(control, result)


def test_counts_from_real_store_exclude_a_failure_recovered_by_a_later_attempt(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": repeat, "agent_ref": {"name": "agent"}}
        for task in (0, 1)
        for repeat in (0, 1)
    ]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    manifest = RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})
    with RolloutStore.start_or_resume(output, lambda: (rows, manifest), resume=False) as store:
        for row in store.pending():
            store.record_dispatch(row)
            if row["_ng_task_index"] == 0 and row["_ng_rollout_index"] == 0:
                store.record_outcome(row | {"_ng_failure_class": "agent_run_error", "_ng_failure_terminal": False})
                retry = row | {"_ng_attempt_index": 1}
                store.record_dispatch(retry)
                store.record_outcome(retry | {"reward": 1.0, "response": {}})
            elif row["_ng_task_index"] == 1:
                store.record_outcome(
                    row
                    | {
                        "_ng_failure_class": "agent_run_error",
                        "_ng_failure_terminal": row["_ng_rollout_index"] == 0,
                    }
                )
            else:
                store.record_outcome(row | {"reward": 1.0, "response": {}})
        store.write_coverage()
    before = {file: file.read_bytes() for file in tmp_path.iterdir() if file.is_file()}

    summary = read_saved_failures(output)

    assert len(failures_path_for(output).read_bytes().splitlines()) == 3
    assert summary.counts.model_dump() == {
        "total": 2,
        "final": 1,
        "not_final": 1,
        "by_class": {"agent_run_error": 2},
        "counted_as_zero": None,
        "final_zeros": 0,
    }
    assert summary.source == coverage_path_for(output)
    assert summary.unavailable_reason is None
    assert {file: file.read_bytes() for file in before} == before


@pytest.mark.parametrize("record_aggregation_count", [False, True])
def test_complete_scored_run_identifies_terminal_policy_zeros(tmp_path: Path, record_aggregation_count: bool) -> None:
    store, counted, scored = _scored_store(
        tmp_path / "candidate",
        final_zeros={(0, 0), (0, 1), (1, 0)},
        other_failures=set(),
        counted_classes=[],
    )
    pair = _pair(tmp_path, "comparison", _values(0.2, 0.4), _values(0.4, 0.6))
    pair["candidate"] = {"rollouts": str(store.output)}
    path = _manifest(tmp_path, benchmarks={"test": {"metrics": [_scale()]}}, models={"model": {"test": pair}})
    control = _compare(path)
    reporting = {"scored": scored}
    if record_aggregation_count:
        reporting["failures_counted_as_zero"] = counted
    store.write_coverage(**reporting)
    coverage = orjson.loads(coverage_path_for(store.output).read_bytes())
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    result = _compare(path)

    assert coverage["complete"] is True
    assert coverage["scored"] == coverage["expected"] == 4
    assert coverage["final_zeros"] == counted == 3
    counts = result.cells[0].candidate.failures.counts
    assert counts.total == counts.final_zeros == 3
    assert counts.counted_as_zero == (3 if record_aggregation_count else None)
    assert result.cells[0].candidate.coverage.status == "verified"
    assert result.cells[0].statistics.ci_low is not None
    assert len(result.warnings) == 1
    expected = "3 scored as zero, 0 not in the score" if record_aggregation_count else "3 final policy zeros"
    assert expected in result.warnings[0]
    assert "3 not in the score" not in result.warnings[0]
    _unchanged_statistics(control, result)
    assert {file: file.read_bytes() for file in sources} == sources
    write_suite_reports(result, tmp_path / "reports", "both")
    saved = orjson.loads((tmp_path / "reports/suite_compare_report.json").read_bytes())
    assert saved["cells"][0]["candidate"]["failures"]["counts"]["final_zeros"] == 3
    assert expected in (tmp_path / "reports/suite_compare_report.md").read_text()


def test_aggregation_zero_count_includes_nonterminal_failures_beyond_final_policy_zeros(tmp_path: Path) -> None:
    store, counted, scored = _scored_store(
        tmp_path / "candidate",
        final_zeros={(1, 0)},
        other_failures={(1, 1)},
        counted_classes=["agent_run_error"],
    )
    store.write_coverage(scored=scored, failures_counted_as_zero=counted)
    original = coverage_path_for(store.output).read_bytes()

    summary = read_saved_failures(store.output)

    assert summary.counts.total == counted == 2
    assert summary.counts.final_zeros == 1
    assert summary.counts.counted_as_zero == 2
    assert orjson.loads(original)["scored"] == 4
    assert "2 scored as zero, 0 not in the score" in saved_failure_warning(summary)
    assert coverage_path_for(store.output).read_bytes() == original


def test_partial_final_zero_count_does_not_classify_other_failures_as_unscored(tmp_path: Path) -> None:
    _, _, candidate = _suite(tmp_path)
    source = _coverage(candidate)
    payload = orjson.loads(source.read_bytes())
    payload.pop("failures_counted_as_zero")
    payload["final_zeros"] = 1
    source.write_bytes(orjson.dumps(payload))

    summary = read_saved_failures(candidate)

    assert summary.counts.counted_as_zero is None
    assert summary.counts.final_zeros == 1
    warning = saved_failure_warning(summary)
    assert "1 final policy zeros" in warning
    assert "1 other failures (score inclusion unknown)" in warning
    assert "not in the score" not in warning


@pytest.mark.parametrize("version", [1, 2])
def test_missing_zero_count_split_preserves_known_failures_with_unknown_score_inclusion(
    tmp_path: Path, version: int
) -> None:
    path, _, candidate = _suite(tmp_path)
    control = _compare(path)
    source = _coverage(candidate, version=version)
    payload = orjson.loads(source.read_bytes())
    payload.pop("failures_counted_as_zero")
    source.write_bytes(orjson.dumps(payload))

    result = _compare(path)

    counts = result.cells[0].candidate.failures.counts
    assert counts.total == 2
    assert counts.counted_as_zero is counts.final_zeros is None
    assert len(result.warnings) == 1
    assert "1 final, 1 not final; score inclusion unknown" in result.warnings[0]
    assert "not in the score" not in result.warnings[0]
    _unchanged_statistics(control, result)


@pytest.mark.parametrize("both_present", [False, True])
def test_failure_summary_requires_exactly_one_result(tmp_path: Path, both_present: bool) -> None:
    values = (
        {"counts": {"total": 0, "final": 0, "not_final": 0, "by_class": {}}, "unavailable_reason": "unknown"}
        if both_present
        else {}
    )

    with pytest.raises(ValidationError, match="exactly one"):
        SuiteFailureSummary(source=tmp_path / "rollouts_coverage.json", **values)


@pytest.mark.parametrize(
    ("change", "expected_reason"),
    [
        ({"failures_counted_as_zero": True}, "at failures_counted_as_zero: Input should be a valid integer"),
        (
            {"failures_counted_as_zero": -1},
            "at failures_counted_as_zero: Input should be greater than or equal to 0",
        ),
        ({"failures_counted_as_zero": 3}, "Failures counted as zero must not exceed the total failure count."),
        (
            {"final_zeros": 2, "failures_counted_as_zero": 2},
            "Final policy zeros must not exceed the final failure count.",
        ),
        (
            {"final_zeros": 1, "failures_counted_as_zero": 0},
            "Failures counted as zero must include all final policy zeros.",
        ),
    ],
)
def test_invalid_zero_count_splits_remain_advisory(tmp_path: Path, change: dict, expected_reason: str) -> None:
    path, _, candidate = _suite(tmp_path)
    control = _compare(path)
    source = _coverage(candidate)
    source.write_bytes(orjson.dumps(orjson.loads(source.read_bytes()) | change))
    original = source.read_bytes()

    result = _compare(path)

    assert result.cells[0].candidate.failures.counts is None
    assert expected_reason in result.cells[0].candidate.failures.unavailable_reason
    assert len(result.warnings) == 1
    assert "Saved failures (entire rollout file): unavailable" in result.warnings[0]
    _unchanged_statistics(control, result)
    assert source.read_bytes() == original


def test_shared_rollout_file_keeps_one_saved_snapshot_across_selected_agents(tmp_path: Path, monkeypatch) -> None:
    baseline = _write_routed_run(
        tmp_path / "baseline",
        [
            ("first_env", "first_agent", (0,), _values(0.2, 0.4)),
            ("second_env", "second_agent", (1,), _values(0.3, 0.5)),
        ],
    )
    candidate = _write_routed_run(
        tmp_path / "candidate",
        [
            ("first_env", "first_agent", (0,), _values(0.4, 0.6)),
            ("second_env", "second_agent", (1,), _values(0.5, 0.7)),
        ],
    )
    source = _coverage(Path(candidate))
    cells = {
        name: {
            "baseline": {"rollouts": baseline, "environment_server": server},
            "candidate": {"rollouts": candidate, "environment_server": server},
        }
        for name, server in (("first", "first_env"), ("second", "second_env"))
    }
    manifest = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in cells},
        models={"model": cells},
    )
    updated = orjson.loads(source.read_bytes()) | {
        "failed": 0,
        "failed_final": 0,
        "failed_not_final": 0,
        "failed_by_class": {},
        "failures_counted_as_zero": 0,
    }
    read_bytes = Path.read_bytes

    def read_while_collector_replaces_snapshot(path: Path) -> bytes:
        saved = read_bytes(path)
        if path == source:
            replacement = source.with_suffix(".next.json")
            replacement.write_bytes(orjson.dumps(updated))
            replacement.replace(source)
        return saved

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_while_collector_replaces_snapshot)
        result = _compare(manifest)

    assert orjson.loads(source.read_bytes())["failed"] == 0
    assert result.warnings == [
        "model/first candidate, model/second candidate: Saved failures (entire rollout file): 2 "
        "(1 scored as zero, 1 not in the score; agent_run_error=2)."
    ]
    for cell in result.cells:
        assert cell.candidate.failures.scope == "rollout_file"
        assert cell.candidate.failures.source == source
        assert cell.candidate.failures.counts.total == 2
        assert cell.candidate.failures.counts.by_class == {"agent_run_error": 2}
        assert cell.candidate.coverage.status == "verified"
        assert cell.statistics.ci_low is not None


def test_real_cli_reports_saved_failures(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest, _, candidate = _suite(tmp_path)
    _coverage(candidate)
    output = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv", ["gym", "eval", "compare-suite", "--suite", str(manifest), "--output-dir", str(output)]
    )

    main()

    result = SuiteComparisonResult.model_validate_json((output / "suite_compare_report.json").read_bytes())
    assert result.cells[0].candidate.failures.counts.total == 2
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.statistics.ci_low is not None
    assert "Saved failures (entire rollout file): 2" in capsys.readouterr().out


def test_report_cannot_overwrite_saved_failure_accounting_through_a_symlink(tmp_path: Path) -> None:
    manifest, _, candidate = _suite(tmp_path)
    coverage = _coverage(candidate)
    output = tmp_path / "reports"
    output.mkdir()
    target = output / "suite_compare_report.json"
    coverage.rename(target)
    coverage.symlink_to(target)
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    result = _compare(manifest)

    with pytest.raises(ConfigError, match="would overwrite a source artifact"):
        write_suite_reports(result, output, "both")

    assert coverage.is_symlink()
    assert not (output / "suite_compare_report.md").exists()
    assert {file: file.read_bytes() for file in before} == before


def test_invalid_later_comparison_preserves_saved_accounting(tmp_path: Path) -> None:
    pair = _pair(tmp_path, "first", _values(0.2, 0.4), _values(0.4, 0.6))
    later = _pair(tmp_path, "later", _values(0.2, 0.4), _values(0.4, 0.6))
    coverage = _coverage(Path(pair["candidate"]["rollouts"]))
    original = coverage.read_bytes()
    aggregate_metrics_path_for(Path(later["candidate"]["rollouts"])).unlink()
    path = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in ("first", "later")},
        models={"model": {"first": pair, "later": later}},
    )
    with pytest.raises(ConfigError, match="Cannot read aggregate metrics"):
        _compare(path)

    assert coverage.read_bytes() == original
