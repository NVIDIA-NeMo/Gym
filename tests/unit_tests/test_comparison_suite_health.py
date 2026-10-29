# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise optional suite health checks against actual persisted trajectory evidence."""

import builtins
from pathlib import Path

import orjson
import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.comparison.suite.health import check_rollout_file_health, rollout_health_warning
from nemo_gym.comparison.suite.runner import build_suite_comparison
from nemo_gym.comparison.suite.schema import SuiteComparisonConfig, SuiteComparisonResult
from nemo_gym.path_utils import aggregate_metrics_path_for, aggregation_sources_path_for, materialized_path_for


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


def _compare(path: Path, *, check_health: bool = False) -> SuiteComparisonResult:
    return build_suite_comparison(
        SuiteComparisonConfig(suite_fpath=str(path), check_health=check_health, health_check_workers=1),
        "gym eval compare-suite",
    )


def test_disabled_health_neither_imports_checker_nor_reads_rollout_files(tmp_path: Path, monkeypatch) -> None:
    path, baseline, candidate = _suite(tmp_path)
    baseline.unlink()
    candidate.unlink()
    original_import = builtins.__import__

    def forbid_health_import(name, *args, **kwargs):
        assert name not in ("nemo_gym.rollout_health", "nemo_gym.comparison.suite.health")
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


def test_multiple_agents_in_one_file_are_checked_once_for_the_entire_file(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.rollout_health as health

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
    result = _compare(path, check_health=True)

    assert calls == [baseline.resolve()]
    assert result.cells[0].baseline.health == result.cells[0].candidate.health
    assert result.cells[0].baseline.health.record_count == 3
    assert result.cells[0].baseline.health.unhealthy == 2
    assert result.cells[0].baseline.health.issues["model_call_zero_completion_tokens"] == 1
    assert len(result.warnings) == 2
    assert all("Rollout file health:" in warning and "2 unhealthy" in warning for warning in result.warnings)


def test_health_checker_worker_count_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SuiteComparisonConfig(suite_fpath=str(tmp_path / "suite.yaml"), health_check_workers=0)


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
