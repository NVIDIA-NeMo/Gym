# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read saved failure accounting without recomputing outcomes or trajectory health."""

from pathlib import Path

import orjson
import pytest

from nemo_gym.cli.main import main
from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.suite.failures import read_saved_failures
from nemo_gym.eval.comparison.suite.report import render_suite_markdown, write_suite_reports
from nemo_gym.eval.comparison.suite.schema import SuiteComparisonResult
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
                "failures_counted_as_zero": 1,
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
    assert summary.counts.model_dump() == {"total": 2, "final": 1, "not_final": 1, "by_class": {"agent_run_error": 2}}
    assert summary.unavailable_reason is None
    assert result.cells[0].baseline.failures is None
    assert result.warnings == [
        "model/test candidate: Saved failures (entire rollout file): 2 (1 final, 1 not final; agent_run_error=2)."
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
    "change",
    [
        {"failed": True},
        {"failed_by_class": {"agent_run_error": -2}},
        {"failed_final": 2},
        {"failed_by_class": {"agent_run_error": 1}},
        {"schema_version": 3},
    ],
)
def test_invalid_saved_counts_remain_advisory(tmp_path: Path, change: dict) -> None:
    path, _, candidate = _suite(tmp_path)
    control = _compare(path)
    source = _coverage(candidate)
    source.write_bytes(orjson.dumps(orjson.loads(source.read_bytes()) | change))
    original = source.read_bytes()

    result = _compare(path)

    assert result.cells[0].candidate.failures.counts is None
    assert result.cells[0].candidate.failures.unavailable_reason
    assert len(result.warnings) == 1
    assert "Saved failures (entire rollout file): unavailable" in result.warnings[0]
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
    assert summary.counts.model_dump() == {"total": 2, "final": 1, "not_final": 1, "by_class": {"agent_run_error": 2}}
    assert summary.source == coverage_path_for(output)
    assert summary.unavailable_reason is None
    assert {file: file.read_bytes() for file in before} == before


def test_shared_rollout_file_keeps_whole_file_counts_and_is_read_once(tmp_path: Path, monkeypatch) -> None:
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
    reads = []
    read_bytes = Path.read_bytes

    def record_read(path: Path) -> bytes:
        if path == source:
            reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)

    result = _compare(manifest)

    assert reads == [source]
    assert len(result.warnings) == 2
    for cell in result.cells:
        assert cell.candidate.failures.scope == "rollout_file"
        assert cell.candidate.failures.source == source
        assert cell.candidate.failures.counts.total == 2
        assert cell.candidate.failures.counts.by_class == {"agent_run_error": 2}
        assert cell.candidate.coverage.status == "verified"
        assert cell.statistics.ci_low is not None


def test_real_cli_reports_saved_failures_without_enabling_a_health_scan(tmp_path: Path, monkeypatch, capsys) -> None:
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
    assert "--check-health" not in result.command
    assert "health_check_enabled" not in result.model_dump()


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


def test_invalid_later_comparison_is_validated_before_saved_accounting_is_read(tmp_path: Path, monkeypatch) -> None:
    pair = _pair(tmp_path, "first", _values(0.2, 0.4), _values(0.4, 0.6))
    later = _pair(tmp_path, "later", _values(0.2, 0.4), _values(0.4, 0.6))
    _coverage(Path(pair["candidate"]["rollouts"]))
    aggregate_metrics_path_for(Path(later["candidate"]["rollouts"])).unlink()
    path = _manifest(
        tmp_path,
        benchmarks={name: {"metrics": [_scale()]} for name in ("first", "later")},
        models={"model": {"first": pair, "later": later}},
    )
    reads = []
    read_bytes = Path.read_bytes

    def record_read(path: Path) -> bytes:
        if path.name.endswith("_coverage.json"):
            reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)

    with pytest.raises(ConfigError, match="Cannot read aggregate metrics"):
        _compare(path)

    assert reads == []
