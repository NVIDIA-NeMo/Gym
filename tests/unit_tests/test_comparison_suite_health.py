# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise optional suite health checks against actual persisted trajectory evidence."""

import builtins
import shutil
from pathlib import Path

import orjson
import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.suite.health import check_rollout_file_health, rollout_health_warning
from nemo_gym.eval.comparison.suite.report import render_suite_markdown, write_suite_reports
from nemo_gym.eval.comparison.suite.runner import build_suite_comparison
from nemo_gym.eval.comparison.suite.schema import SuiteComparisonConfig, SuiteComparisonResult
from nemo_gym.rollouts.rollout_records import coverage_path_for, journal_path_for
from nemo_gym.rollouts.rollout_recovery import RunManifest, manifest_path_for
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.utils.path_utils import (
    aggregate_metrics_path_for,
    aggregation_sources_path_for,
    attempt_extension_path_for,
    failures_path_for,
    materialized_path_for,
    rerun_masked_journal_path_for,
    resolved_config_path_for,
    resubmit_offset_path_for,
)


def _record(repeat: int, *, state: str = "healthy", task: int = 0) -> dict:
    call_id = f"call-{task}-{repeat}"
    tokens = 0 if state == "unhealthy" else 2
    row = {
        "agent_ref": {"name": "agent"},
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "response": {
            "output": [{"type": "message", "role": "assistant", "content": "ok"}],
            "usage": {"input_tokens": 3, "output_tokens": tokens},
        },
        "ng_trajectory": {
            "task_id": str(task),
            "rollout_id": f"{task}-{repeat}",
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": str(task),
                    "rollout_id": f"{task}-{repeat}",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "answer": "ok",
                    "step_count": 1,
                    "model_calls": [{"model_call_id": call_id}],
                }
            ],
            "model_calls": [
                {
                    "model_call_id": call_id,
                    "request": {"input": "question"},
                    "response": {"output_text": "ok"},
                    "response_metadata": {
                        "response_id": f"response-{task}-{repeat}",
                        "status_code": 200,
                        "response_status": "completed",
                        "finish_reason": "stop",
                    },
                    "token_stats": {"prompt_tokens": 3, "completion_tokens": tokens},
                }
            ],
        },
    }
    if state == "unobserved":
        del row["ng_trajectory"]
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))


def _run(directory: Path, *, state: str = "healthy", offset: float = 0) -> Path:
    directory.mkdir()
    path = directory / "rollouts.jsonl"
    _write_rows(path, [_record(repeat, state=state) for repeat in range(2)])
    _write_rows(
        materialized_path_for(path),
        [{"agent_ref": {"name": "agent"}, "_ng_task_index": 0, "_ng_rollout_index": repeat} for repeat in range(2)],
    )
    aggregate_metrics_path_for(path).write_bytes(
        orjson.dumps(
            [
                {
                    "agent_ref": {"name": "agent"},
                    "agent_metrics": {"num_repeats": 2},
                    "repeat_level_metrics": [
                        {"_ng_rollout_index": repeat, "sample_count": 1, "mean/reward": score + offset}
                        for repeat, score in enumerate((0.2, 0.4))
                    ],
                    "group_level_metrics": [
                        {
                            "_ng_task_index": 0,
                            "expected_num_rollouts": 2,
                            "missing_num_rollouts": 0,
                            "rollout_infos": [
                                {"_ng_task_index": 0, "_ng_rollout_index": repeat} for repeat in range(2)
                            ],
                        }
                    ],
                }
            ]
        )
    )
    return path


def _suite(tmp_path: Path, *, candidate_state: str = "healthy") -> tuple[Path, Path, Path]:
    baseline = _run(tmp_path / "baseline")
    candidate = _run(tmp_path / "candidate", state=candidate_state, offset=0.2)
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmarks": {
                    "test": {"metrics": [{"name": "mean/reward", "low": 0, "high": 1, "higher_is_better": True}]}
                },
                "models": {
                    "model": {
                        "test": {"baseline": {"rollouts": str(baseline)}, "candidate": {"rollouts": str(candidate)}}
                    }
                },
            }
        )
    )
    return path, baseline, candidate


def _compare(
    path: Path, *, check_health: bool = False, health_output_dir: Path | None = None
) -> SuiteComparisonResult:
    return build_suite_comparison(
        SuiteComparisonConfig(
            suite_fpath=str(path),
            check_health=check_health,
            health_check_workers=1,
            health_check_output_dirpath=str(health_output_dir) if health_output_dir is not None else None,
        ),
        "gym eval compare-suite",
    )


@pytest.fixture
def health_scans(monkeypatch) -> list[Path]:
    import nemo_gym.rollouts.rollout_health as health

    paths = []
    original_check = health.run_health_checks

    def track_scan(rollout_path, **kwargs):
        paths.append(rollout_path)
        return original_check(rollout_path, **kwargs)

    monkeypatch.setattr(health, "run_health_checks", track_scan)
    return paths


def test_later_invalid_cell_is_rejected_before_any_health_scan(tmp_path: Path, health_scans: list[Path]) -> None:
    path, _, _ = _suite(tmp_path)
    later_baseline = _run(tmp_path / "later-baseline")
    later_candidate = _run(tmp_path / "later-candidate")
    manifest = yaml.safe_load(path.read_text())
    manifest["benchmarks"]["later"] = manifest["benchmarks"]["test"]
    missing_aggregate = tmp_path / "mistyped-aggregate.json"
    manifest["models"]["model"]["later"] = {
        "baseline": {"rollouts": str(later_baseline)},
        "candidate": {"rollouts": str(later_candidate), "aggregate_metrics": str(missing_aggregate)},
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    output = tmp_path / "health-reports"

    with pytest.raises(ConfigError, match="mistyped-aggregate.json"):
        _compare(path, check_health=True, health_output_dir=output)

    assert health_scans == []
    assert not output.exists()
    assert {file: file.read_bytes() for file in sources} == sources


def test_no_eligible_cells_is_rejected_before_any_health_scan(tmp_path: Path, health_scans: list[Path]) -> None:
    path, baseline, candidate = _suite(tmp_path)
    for rollouts in (baseline, candidate):
        aggregate = aggregate_metrics_path_for(rollouts)
        entries = orjson.loads(aggregate.read_bytes())
        entry = entries[0]
        entry["agent_metrics"]["num_repeats"] = 1
        entry["repeat_level_metrics"] = entry["repeat_level_metrics"][:1]
        group = entry["group_level_metrics"][0]
        group["expected_num_rollouts"] = 1
        group["rollout_infos"] = group["rollout_infos"][:1]
        aggregate.write_bytes(orjson.dumps(entries))
        _write_rows(rollouts, [_record(0)])
        _write_rows(
            materialized_path_for(rollouts),
            [{"agent_ref": {"name": "agent"}, "_ng_task_index": 0, "_ng_rollout_index": 0}],
        )
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    output = tmp_path / "health-reports"

    with pytest.raises(ConfigError, match="No eligible model-benchmark comparisons remain"):
        _compare(path, check_health=True, health_output_dir=output)

    assert health_scans == []
    assert not output.exists()
    assert {file: file.read_bytes() for file in sources} == sources


@pytest.mark.parametrize("failure", ["file", "unwritable"])
def test_invalid_health_output_root_is_rejected_before_any_scan(
    tmp_path: Path, monkeypatch, health_scans: list[Path], failure: str
) -> None:
    import nemo_gym.eval.comparison.suite.runner as runner

    path, _, _ = _suite(tmp_path)
    output = tmp_path / "health-output"
    if failure == "file":
        output.write_bytes(b"preserve existing output file")
    else:
        output.mkdir()

        def deny_output_probe(*args, **kwargs):
            assert Path(kwargs["dir"]) == output
            raise PermissionError("test health output access denied")

        # Patch the filesystem boundary: the comparison and checker remain real.
        monkeypatch.setattr(runner, "TemporaryFile", deny_output_probe)
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}

    with pytest.raises(ConfigError, match="--health-check-output-dir"):
        _compare(path, check_health=True, health_output_dir=output)

    assert health_scans == []
    assert {file: file.read_bytes() for file in sources} == sources
    if failure == "unwritable":
        assert list(output.iterdir()) == []


def test_disabled_health_neither_imports_checker_nor_reads_rollout_files(tmp_path: Path, monkeypatch) -> None:
    path, baseline, candidate = _suite(tmp_path)
    baseline.unlink()
    candidate.unlink()
    original_import = builtins.__import__

    def forbid_health_import(name, *args, **kwargs):
        assert name not in ("nemo_gym.rollouts.rollout_health", "nemo_gym.eval.comparison.suite.health")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_health_import)
    result = _compare(path)

    assert result.health_check_enabled is False
    assert result.cells[0].baseline.health is result.cells[0].candidate.health is None
    assert result.overall.statistics.delta == pytest.approx(20)
    assert not result.warnings


@pytest.mark.parametrize("state", ["healthy", "unhealthy", "unobserved"])
def test_health_verdicts_are_computed_from_current_records_without_altering_statistics(
    tmp_path: Path, state: str
) -> None:
    path, baseline, candidate = _suite(tmp_path, candidate_state=state)
    # An existing summary may be stale or may describe a different JSONL in this folder.
    (candidate.parent / "quality_summary.json").write_text('"stale report: healthy"')
    (candidate.parent / "rollout_verdicts.jsonl").write_text("do not overwrite\n")
    # A merged-run sidecar must not redirect this check to different source files.
    aggregation_sources_path_for(candidate).write_text('["missing-shard.jsonl"]')
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    without = _compare(path)
    result = _compare(path, check_health=True)

    assert result.health_check_enabled is True
    assert result.cells[0].baseline.health.healthy == 2
    summary = result.cells[0].candidate.health
    assert summary.record_count == 2
    assert getattr(summary, state) == 2
    assert summary.healthy + summary.unhealthy + summary.unobserved == 2
    assert summary.run_healthy is True
    assert summary.failed is None  # No failure sidecar means counts are unknown, rather than known zero.
    assert summary.summary_path is summary.verdicts_path is None
    assert result.overall.statistics == without.overall.statistics
    assert result.per_model == without.per_model
    assert result.cells[0].statistics == without.cells[0].statistics
    assert result.cells[0].candidate.repeat_scores == without.cells[0].candidate.repeat_scores
    assert result.cells[0].candidate.coverage == without.cells[0].candidate.coverage
    assert {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()} == before
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result
    if state == "healthy":
        assert not summary.issues
        assert not result.warnings
    elif state == "unhealthy":
        assert summary.issues["model_call_zero_completion_tokens"] == 2
        assert all(count > 0 for count in summary.issues.values())
        assert len(result.warnings) == 1
        assert "model/test candidate: Rollout file health: 0 healthy, 2 unhealthy" in result.warnings[0]
        assert "model_call_zero_completion_tokens=2" in result.warnings[0]
    else:
        assert summary.unhealthy == 0
        assert not summary.issues
        assert len(result.warnings) == 1
        assert "2 unobserved of 2 records" in result.warnings[0]
        assert "insufficient evidence" in result.warnings[0]
    assert all("\n" not in warning for warning in result.warnings)


def test_missing_raw_records_warn_even_when_aggregate_coverage_and_remaining_records_are_healthy(
    tmp_path: Path,
) -> None:
    path, _, candidate = _suite(tmp_path)
    _write_rows(candidate, [_record(0)])
    result = _compare(path, check_health=True)

    summary = result.cells[0].candidate.health
    assert summary.healthy == summary.record_count == 1
    assert summary.unhealthy == 0
    assert summary.run_healthy is False
    assert summary.run_unhealthy_reasons == ["missing 1 of 2"]
    assert result.cells[0].candidate.coverage.status == "verified"
    assert result.overall.statistics.ci_low is not None
    assert len(result.warnings) == 1
    assert "run: missing 1 of 2" in result.warnings[0]


@pytest.mark.parametrize("terminal", [False, True], ids=["retryable", "final"])
def test_any_unrecovered_failure_warns_even_when_saved_records_are_healthy(tmp_path: Path, terminal: bool) -> None:
    path, _, candidate = _suite(tmp_path)
    _write_rows(candidate, [_record(1)])
    _write_rows(
        failures_path_for(candidate),
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "_ng_failure_class": "agent_run_error",
                "_ng_failure_terminal": terminal,
            }
        ],
    )
    # Keep the aggregate sidecar unchanged: health findings must not alter the comparison's statistics.
    without = _compare(path)
    result = _compare(path, check_health=True, health_output_dir=tmp_path / "health-reports")

    cell = result.cells[0]
    summary = cell.candidate.health
    assert summary.record_count == summary.healthy == 1
    assert summary.unhealthy == summary.unobserved == 0
    assert summary.run_healthy is False
    assert summary.issues == {"run_excessive_failures": 1}
    group = "final" if terminal else "not final"
    reason = f"{group} failures 1 of 2 (50.0%), limit 0%"
    assert summary.run_unhealthy_reasons == [reason]
    assert len(result.warnings) == 1
    assert "model/test candidate: Rollout file health: 1 healthy, 0 unhealthy" in result.warnings[0]
    assert f"run: {reason}" in result.warnings[0]
    assert "\n" not in result.warnings[0]
    assert result.overall.statistics == without.overall.statistics
    assert result.per_model == without.per_model
    assert cell.statistics == without.cells[0].statistics
    assert cell.candidate.repeat_count == without.cells[0].candidate.repeat_count == 2
    assert cell.candidate.repeat_scores == without.cells[0].candidate.repeat_scores
    assert cell.candidate.coverage == without.cells[0].candidate.coverage
    saved = orjson.loads(summary.summary_path.read_bytes())["run"]["artifacts"]
    assert saved["expected"] == 2
    assert saved["missing"] == 0
    assert saved["failed"] == {
        "total": 1,
        "final": int(terminal),
        "not_final": int(not terminal),
        "by_class": {"agent_run_error": 1},
    }
    assert summary.failed.model_dump() == saved["failed"]
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored.cells[0].candidate.health.failed == summary.failed
    warning = result.warnings[0]
    assert "failed rollout" in warning
    assert f"({int(terminal)} final, {int(not terminal)} retryable; agent_run_error=1)" in warning
    assert saved["limits"]["max_not_final_failure_fraction"] == 0
    assert saved["limits"]["max_final_failure_fraction"] == 0


@pytest.mark.parametrize("recovered_failure", [False, True], ids=["empty-sidecar", "recovered-failure"])
def test_zero_failure_limits_allow_no_unrecovered_failures(tmp_path: Path, recovered_failure: bool) -> None:
    path, _, candidate = _suite(tmp_path)
    _write_rows(
        failures_path_for(candidate),
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "_ng_failure_class": "agent_run_error",
                "_ng_failure_terminal": False,
            }
        ]
        if recovered_failure
        else [],
    )
    result = _compare(path, check_health=True, health_output_dir=tmp_path / "health-reports")

    summary = result.cells[0].candidate.health
    assert summary.record_count == summary.healthy == 2
    assert summary.unhealthy == summary.unobserved == 0
    assert summary.run_healthy is True
    assert summary.run_unhealthy_reasons == []
    assert summary.issues == {}
    assert result.warnings == []
    saved = orjson.loads(summary.summary_path.read_bytes())["run"]["artifacts"]
    assert saved["failed"] == {"total": 0, "final": 0, "not_final": 0, "by_class": {}}
    assert summary.failed.model_dump() == saved["failed"]
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored.cells[0].candidate.health.failed.total == 0
    assert saved["limits"]["max_not_final_failure_fraction"] == 0
    assert saved["limits"]["max_final_failure_fraction"] == 0
    assert saved["coverage"]["run_excessive_failures"] == {"evaluated": 1, "unobserved": 0, "ignored": 0}


def test_recorded_failures_warn_without_inventory_even_when_saved_trajectories_are_healthy(tmp_path: Path) -> None:
    path, _, candidate = _suite(tmp_path)
    materialized_path_for(candidate).unlink()
    _write_rows(
        failures_path_for(candidate),
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": repeat,
                "_ng_failure_class": "agent_run_error",
                "_ng_failure_terminal": terminal,
            }
            for repeat, terminal in ((2, False), (3, True))
        ],
    )
    without = _compare(path)
    result = _compare(path, check_health=True, health_output_dir=tmp_path / "health-reports")

    summary = result.cells[0].candidate.health
    assert summary.record_count == summary.healthy == 2
    assert summary.unhealthy == summary.unobserved == 0
    assert summary.run_healthy is None
    assert summary.run_unhealthy_reasons == []
    assert summary.issues == {}
    expected_failed = {"total": 2, "final": 1, "not_final": 1, "by_class": {"agent_run_error": 2}}
    assert summary.failed.model_dump() == expected_failed
    saved = orjson.loads(summary.summary_path.read_bytes())["run"]["artifacts"]
    assert saved["failed"] == expected_failed
    assert saved["expected"] is None
    assert saved["coverage"]["run_excessive_failures"] == {"evaluated": 0, "unobserved": 1, "ignored": 0}
    warnings = [warning for warning in result.warnings if "Rollout file health:" in warning]
    assert len(warnings) == 1
    assert "2 healthy, 0 unhealthy, 0 unobserved of 2 records" in warnings[0]
    assert "2 failed rollouts without saved results (1 final, 1 retryable; agent_run_error=2)" in warnings[0]
    assert "run_excessive_failures" not in warnings[0]
    assert "\n" not in warnings[0]
    assert result.cells[0].statistics == without.cells[0].statistics
    assert result.overall.statistics == without.overall.statistics
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored.cells[0].candidate.health.failed.model_dump() == expected_failed
    assert "2 failed rollouts without saved results" in render_suite_markdown(restored)


def test_empty_raw_file_keeps_failure_counts_in_json_and_warning(tmp_path: Path) -> None:
    path, _, candidate = _suite(tmp_path)
    candidate.write_bytes(b"")
    materialized_path_for(candidate).unlink()
    _write_rows(
        failures_path_for(candidate),
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": repeat,
                "_ng_failure_class": "agent_run_error",
                "_ng_failure_terminal": terminal,
            }
            for repeat, terminal in ((0, False), (1, True))
        ],
    )
    result = _compare(path, check_health=True)

    summary = result.cells[0].candidate.health
    assert summary.record_count == summary.healthy == summary.unhealthy == summary.unobserved == 0
    assert summary.failed.model_dump() == {
        "total": 2,
        "final": 1,
        "not_final": 1,
        "by_class": {"agent_run_error": 2},
    }
    warning = next(warning for warning in result.warnings if "Rollout file health:" in warning)
    assert "rollout file is empty" in warning
    assert "no trajectory health could be checked" in warning
    assert "2 failed rollouts without saved results (1 final, 1 retryable; agent_run_error=2)" in warning
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored.cells[0].candidate.health.failed == summary.failed


def test_empty_raw_file_warns_instead_of_appearing_healthy(tmp_path: Path) -> None:
    path, _, candidate = _suite(tmp_path)
    candidate.write_bytes(b"")
    result = _compare(path, check_health=True)

    summary = result.cells[0].candidate.health
    assert summary.record_count == summary.healthy == summary.unhealthy == summary.unobserved == 0
    assert summary.run_healthy is False
    assert len(result.warnings) == 1
    assert "rollout file is empty" in result.warnings[0]
    assert "no trajectory health could be checked" in result.warnings[0]
    assert result.overall.statistics.ci_low is not None


@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_unavailable_health_warns_and_preserves_comparison(tmp_path: Path, monkeypatch, failure: str) -> None:
    path, _, candidate = _suite(tmp_path)
    without = _compare(path)
    if failure == "missing":
        candidate.unlink()
    else:
        original_open = Path.open

        def open_unless_denied(file: Path, *args, **kwargs):
            if file == candidate:
                raise PermissionError("test rollout access denied")
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_unless_denied)
    result = _compare(path, check_health=True)

    summary = result.cells[0].candidate.health
    assert summary.record_count is None
    assert summary.unavailable_reason
    assert len(result.warnings) == 1
    assert "Rollout file health: unavailable" in result.warnings[0]
    assert "readable" in result.warnings[0]
    assert result.overall.statistics == without.overall.statistics


def test_malformed_failure_sidecar_warns_without_losing_comparison_reports(tmp_path: Path) -> None:
    path, _, candidate = _suite(tmp_path)
    _write_rows(failures_path_for(candidate), [{"_ng_task_index": [], "_ng_rollout_index": 2}])
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    without = _compare(path)
    result = _compare(path, check_health=True)

    cell = result.cells[0]
    baseline_health = cell.baseline.health
    assert baseline_health.record_count == baseline_health.healthy == 2
    assert baseline_health.unhealthy == baseline_health.unobserved == 0
    assert rollout_health_warning(baseline_health) is None
    candidate_health = cell.candidate.health
    assert candidate_health.record_count is None
    assert candidate_health.failed is None
    assert "unhashable type: 'list'" in candidate_health.unavailable_reason
    assert len(result.warnings) == 1
    assert "model/test candidate: Rollout file health: unavailable" in result.warnings[0]
    assert "unhashable type: 'list'" in result.warnings[0]
    assert result.overall == without.overall
    assert result.per_model == without.per_model
    assert cell.statistics == without.cells[0].statistics
    assert cell.model_weight == without.cells[0].model_weight
    assert cell.overall_weight == without.cells[0].overall_weight
    for arm in ("baseline", "candidate"):
        source, original = getattr(cell, arm), getattr(without.cells[0], arm)
        assert source.repeat_count == original.repeat_count == 2
        assert source.repeat_scores == original.repeat_scores
        assert source.coverage == original.coverage
    assert {file: file.read_bytes() for file in sources} == sources

    output = tmp_path / "comparison"
    written = write_suite_reports(result, output, "both")
    assert set(written) == {output / "suite_compare_report.md", output / "suite_compare_report.json"}
    restored = SuiteComparisonResult.model_validate_json((output / "suite_compare_report.json").read_bytes())
    assert restored == result
    assert result.warnings[0] in (output / "suite_compare_report.md").read_text()


def test_superseded_run_history_warns_without_losing_comparison_reports(tmp_path: Path) -> None:
    path, _, original_candidate = _suite(tmp_path)
    candidate = tmp_path / "managed-candidate" / "rollouts.jsonl"
    candidate.parent.mkdir()
    inputs = candidate.parent / "tasks.jsonl"
    rows = [{"agent_ref": {"name": "agent"}, "_ng_task_index": 0, "_ng_rollout_index": i} for i in range(2)]
    _write_rows(inputs, rows)

    def prepare() -> tuple[list[dict], RunManifest]:
        return rows, RunManifest.create(inputs, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})

    with RolloutStore.start_or_resume(candidate, prepare, resume=False) as store:
        pending = store.pending(3)
        for row in pending:
            store.record_dispatch(row)
            repeat = row["_ng_rollout_index"]
            store.record_outcome(row | _record(repeat) | {"reward": 0.4 + repeat * 0.2})
        store.allocate_attempt(pending[0])
        store.record_outcome(pending[0] | _record(0) | {"reward": 0.4})
    aggregate_metrics_path_for(candidate).write_bytes(aggregate_metrics_path_for(original_candidate).read_bytes())
    manifest = yaml.safe_load(path.read_text())
    manifest["models"]["model"]["test"]["candidate"] = {"rollouts": str(candidate)}
    path.write_text(yaml.safe_dump(manifest))
    sources = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    without = _compare(path)

    result = _compare(path, check_health=True)

    cell = result.cells[0]
    assert cell.baseline.health.record_count == cell.baseline.health.healthy == 2
    assert rollout_health_warning(cell.baseline.health) is None
    assert cell.candidate.health.record_count is None
    assert cell.candidate.health.failed is None
    assert "superseded outcomes" in cell.candidate.health.unavailable_reason
    assert "gym eval aggregate" in cell.candidate.health.unavailable_reason
    assert len(result.warnings) == 1
    assert "model/test candidate: Rollout file health: unavailable" in result.warnings[0]
    assert result.overall == without.overall
    assert result.per_model == without.per_model
    assert cell.statistics == without.cells[0].statistics
    assert cell.model_weight == without.cells[0].model_weight
    assert cell.overall_weight == without.cells[0].overall_weight
    for arm in ("baseline", "candidate"):
        source, original = getattr(cell, arm), getattr(without.cells[0], arm)
        assert source.repeat_count == original.repeat_count == 2
        assert source.repeat_scores == original.repeat_scores
        assert source.coverage == original.coverage
    assert {file: file.read_bytes() for file in sources} == sources
    output = tmp_path / "comparison"
    written = write_suite_reports(result, output, "both")
    assert set(written) == {output / "suite_compare_report.md", output / "suite_compare_report.json"}
    restored = SuiteComparisonResult.model_validate_json((output / "suite_compare_report.json").read_bytes())
    assert restored == result
    markdown = (output / "suite_compare_report.md").read_text()
    assert "model/test candidate: Rollout file health: unavailable" in markdown
    assert "superseded outcomes" in markdown
    assert "gym eval aggregate" in markdown


def test_health_summary_contract_errors_are_not_reported_as_unavailable(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.rollouts.rollout_health as health

    path = _run(tmp_path / "run")
    original_check = health.run_health_checks

    def check_with_invalid_record_count(*args, **kwargs):
        result = original_check(*args, **kwargs)
        result.summary["run"]["artifacts"]["records"] = -1
        return result

    monkeypatch.setattr(health, "run_health_checks", check_with_invalid_record_count)
    with pytest.raises(ValidationError, match="record_count"):
        check_rollout_file_health(path, workers=1)


def test_multiple_agents_in_one_file_are_checked_once_for_the_entire_file(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.rollouts.rollout_health as health

    path, baseline, _ = _suite(tmp_path)
    entries = orjson.loads(aggregate_metrics_path_for(baseline).read_bytes())
    entries.append({**entries[0], "agent_ref": {"name": "other_agent"}})
    aggregate_metrics_path_for(baseline).write_bytes(orjson.dumps(entries))
    inputs = [orjson.loads(line) for line in materialized_path_for(baseline).read_bytes().splitlines()]
    _write_rows(
        materialized_path_for(baseline),
        [*inputs, *[{**row, "agent_ref": {"name": "other_agent"}} for row in inputs]],
    )
    manifest = yaml.safe_load(path.read_text())
    pair = manifest["models"]["model"]["test"]
    pair["baseline"]["agent"] = "agent"
    pair["candidate"] = {"rollouts": str(baseline), "agent": "other_agent"}
    path.write_text(yaml.safe_dump(manifest))
    rows = [_record(repeat) for repeat in range(2)]
    rows.append({**_record(0, state="unhealthy"), "agent_ref": {"name": "other_agent"}})
    _write_rows(baseline, rows)
    calls = []
    original_check = health.run_health_checks

    def checked_source(rollout_path, **kwargs):
        calls.append(rollout_path)
        return original_check(rollout_path, **kwargs)

    monkeypatch.setattr(health, "run_health_checks", checked_source)
    output = tmp_path / "health-reports"
    result = _compare(path, check_health=True, health_output_dir=output)

    assert calls == [baseline.resolve()]
    assert result.cells[0].baseline.health == result.cells[0].candidate.health
    assert result.cells[0].baseline.health.record_count == 3
    assert result.cells[0].baseline.health.unhealthy == 2
    assert result.cells[0].baseline.health.issues["model_call_zero_completion_tokens"] == 1
    assert result.cells[0].baseline.health.summary_path.is_file()
    assert len(list(output.glob("*/quality_summary.json"))) == 1
    assert len(result.warnings) == 2
    assert all("Rollout file health:" in warning and "2 unhealthy" in warning for warning in result.warnings)


def test_health_checker_worker_count_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SuiteComparisonConfig(suite_fpath=str(tmp_path / "suite.yaml"), health_check_workers=0)


@pytest.mark.parametrize("under_source", [False, True])
def test_full_health_reports_persist_per_source_without_changing_inputs(tmp_path: Path, under_source: bool) -> None:
    path, baseline, _ = _suite(tmp_path, candidate_state="unhealthy")
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    output = baseline.parent if under_source else tmp_path / "saved-health"
    result = _compare(path, check_health=True, health_output_dir=output)
    cell = result.cells[0]
    directories = set()
    for run in (cell.baseline, cell.candidate):
        summary = run.health
        assert summary.summary_path.is_file()
        assert summary.verdicts_path.is_file()
        assert summary.summary_path.parent == summary.verdicts_path.parent
        assert summary.summary_path.parent.parent == output
        directories.add(summary.summary_path.parent)
        saved = orjson.loads(summary.summary_path.read_bytes())
        assert saved["run"]["verdicts"] == {
            "healthy": summary.healthy,
            "unhealthy": summary.unhealthy,
            "unobserved": summary.unobserved,
        }
        verdicts = [orjson.loads(line)["verdict"] for line in summary.verdicts_path.read_bytes().splitlines()]
        assert verdicts == (["healthy", "healthy"] if run is cell.baseline else ["unhealthy", "unhealthy"])
    assert len(directories) == 2
    assert {file: file.read_bytes() for file in before} == before
    markdown = render_suite_markdown(result)
    assert all(str(run.health.summary_path) in markdown for run in (cell.baseline, cell.candidate))
    assert all(str(run.health.verdicts_path) in markdown for run in (cell.baseline, cell.candidate))
    restored = SuiteComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_omitted_health_output_directory_cleans_up_temporary_reports(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.eval.comparison.suite.health as suite_health

    path, _, _ = _suite(tmp_path)
    temporary_directories = []
    original_temporary_directory = suite_health.TemporaryDirectory

    def track_temporary_directory(**kwargs):
        context = original_temporary_directory(**kwargs)
        temporary_directories.append(Path(context.name))
        return context

    monkeypatch.setattr(suite_health, "TemporaryDirectory", track_temporary_directory)
    result = _compare(path, check_health=True)
    assert len(temporary_directories) == 2
    assert all(not directory.exists() for directory in temporary_directories)
    assert result.cells[0].baseline.health.healthy == result.cells[0].candidate.health.healthy == 2
    for run in (result.cells[0].baseline, result.cells[0].candidate):
        assert run.health.summary_path is run.health.verdicts_path is None


def test_output_directory_alone_does_not_enable_health_checks(tmp_path: Path) -> None:
    path, baseline, candidate = _suite(tmp_path)
    baseline.unlink()
    candidate.unlink()
    output = tmp_path / "unused-health-output"
    result = _compare(path, health_output_dir=output)
    assert result.health_check_enabled is False
    assert not output.exists()
    assert result.cells[0].baseline.health is result.cells[0].candidate.health is None
    assert not result.warnings


@pytest.mark.parametrize(
    "collision",
    [
        "directory",
        "aggregate",
        "quality_summary.json",
        "rollout_verdicts.jsonl",
        "manifest",
        "journal",
        "coverage",
        "resolved-config",
        "attempt-extension",
        "rerun-masked-journal",
        "resubmit-offset",
    ],
)
def test_health_report_symlinks_cannot_overwrite_another_source(tmp_path: Path, collision: str) -> None:
    path, _, candidate = _suite(tmp_path)
    (candidate.parent / "quality_summary.json").write_text('{"source_health": "preserve"}')
    (candidate.parent / "rollout_verdicts.jsonl").write_text('{"source_verdict": "preserve"}\n')
    output = tmp_path / "health-output"
    result = _compare(path, check_health=True, health_output_dir=output)
    summary = result.cells[0].baseline.health
    if collision == "directory":
        directory = summary.summary_path.parent
        shutil.rmtree(directory)
        directory.symlink_to(candidate.parent, target_is_directory=True)
    else:
        summary.summary_path.unlink()
        metadata = {
            "manifest": manifest_path_for(candidate),
            "journal": journal_path_for(candidate),
            "coverage": coverage_path_for(candidate),
            "resolved-config": resolved_config_path_for(candidate),
            "attempt-extension": attempt_extension_path_for(candidate),
            "rerun-masked-journal": rerun_masked_journal_path_for(candidate),
            "resubmit-offset": resubmit_offset_path_for(candidate),
        }
        if collision in metadata:
            destination = metadata[collision]
            destination.write_text('{"source_metadata": "preserve"}\n')
        else:
            destination = (
                aggregate_metrics_path_for(candidate) if collision == "aggregate" else candidate.parent / collision
            )
        summary.summary_path.symlink_to(destination)
    sources = [file for file in tmp_path.rglob("*") if file.is_file()]
    before = {file: file.read_bytes() for file in sources}
    with pytest.raises(ConfigError, match="choose another --health-check-output-dir"):
        _compare(path, check_health=True, health_output_dir=output)
    assert {file: file.read_bytes() for file in sources} == before


def test_suite_report_cannot_overwrite_retained_health_report(tmp_path: Path) -> None:
    path, _, _ = _suite(tmp_path)
    result = _compare(path, check_health=True, health_output_dir=tmp_path / "health")
    health_summary = result.cells[0].baseline.health.summary_path
    before = health_summary.read_bytes()
    report_dir = tmp_path / "comparison"
    report_dir.mkdir()
    (report_dir / "suite_compare_report.json").symlink_to(health_summary)
    with pytest.raises(ConfigError, match="would overwrite a source artifact"):
        write_suite_reports(result, report_dir, "json")
    assert health_summary.read_bytes() == before


def test_different_sources_cannot_share_a_health_report_through_file_symlinks(tmp_path: Path) -> None:
    path, _, _ = _suite(tmp_path)
    output = tmp_path / "health"
    result = _compare(path, check_health=True, health_output_dir=output)
    cell = result.cells[0]
    baseline_summary = cell.baseline.health.summary_path
    candidate_summary = cell.candidate.health.summary_path
    baseline_summary.unlink()
    baseline_summary.symlink_to(candidate_summary)
    before = candidate_summary.read_bytes()
    with pytest.raises(ConfigError, match="overlap other reports"):
        _compare(path, check_health=True, health_output_dir=output)
    assert candidate_summary.read_bytes() == before


def test_health_warning_limits_check_ids_but_preserves_full_typed_findings(tmp_path: Path) -> None:
    path = _run(tmp_path / "run")
    rows = [_record(repeat, state="unhealthy") for repeat in range(2)]
    for row in rows:
        row["ng_trajectory"]["turns"][0]["answer"] = ""
        row["ng_trajectory"]["turns"][0]["step_count"] = 0
        row["ng_trajectory"]["model_calls"][0]["response_metadata"]["status_code"] = 500
        row["ng_trajectory"]["model_calls"][0]["response_metadata"]["response_status"] = "failed"
    _write_rows(path, rows)
    summary = check_rollout_file_health(path, workers=1)
    warning = rollout_health_warning(summary)

    assert len(summary.issues) > 3
    assert f"+{len(summary.issues) - 3} more check types" in warning
    assert "\n" not in warning
