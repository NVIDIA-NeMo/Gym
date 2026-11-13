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
"""Verify the contract between Gym's metric producer and suite comparison."""

from pathlib import Path

import orjson
import pytest
import yaml

from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.suite.report import write_suite_reports
from nemo_gym.eval.comparison.suite.runner import build_suite_comparison
from nemo_gym.eval.comparison.suite.schema import SuiteComparisonConfig, SuiteComparisonResult
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, materialized_path_for


def _write_masked_source(
    directory: Path,
    *,
    offset: float,
    scored_repeats: tuple[int, ...],
    masked_repeats: tuple[int, ...] = (),
    with_perf: bool = False,
    missing_reward_repeat: int | None = None,
) -> dict[str, str]:
    directory.mkdir(parents=True)
    server = directory.name + "_env"
    rows = []
    inputs = []
    for task in range(3):
        for repeat in sorted((*scored_repeats, *masked_repeats)):
            identity = {"_ng_task_index": task, "_ng_rollout_index": repeat, "_ng_environment_server": server}
            inputs.append(identity)
            row = identity | {"reward": 0.1 + 0.2 * task + 0.1 * repeat + offset}
            if repeat in masked_repeats:
                row.update({"mask_sample": True, "reward": 0.99})
            if repeat == missing_reward_repeat:
                row.pop("reward")
            if with_perf:
                row["ng_perf"] = {"completion_tokens": 10 * (repeat + 1), "num_turns": 2}
            rows.append(row)
    rollouts = directory / "rollouts.jsonl"
    rollouts.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    entry = {
        "agent_ref": {"name": directory.name + "_agent"},
        "_ng_environment_server": server,
        **compute_aggregate_metrics(rows).model_dump(mode="json", exclude_none=True),
    }
    aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps([entry]))
    materialized_path_for(rollouts).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in inputs))
    return {"rollouts": str(rollouts), "environment_server": server}


def _write_manifest(directory: Path, benchmarks: dict[str, dict[str, dict[str, str]]]) -> Path:
    manifest = directory / "suite.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmarks": {
                    name: {"metrics": [{"name": "mean/reward", "low": 0, "high": 1, "higher_is_better": True}]}
                    for name in benchmarks
                },
                "models": {"model": benchmarks},
            }
        )
    )
    return manifest


def _compare(manifest: Path) -> SuiteComparisonResult:
    return build_suite_comparison(SuiteComparisonConfig(suite_fpath=str(manifest)), "gym eval compare-suite")


def test_suite_compares_metrics_generated_from_rollout_records(tmp_path: Path) -> None:
    def write_source(*, role: str, offset: float) -> dict[str, str]:
        directory = tmp_path / role
        directory.mkdir()
        rollouts = directory / "rollouts.jsonl"
        server = f"{role}_env"
        rows = [
            {
                "_ng_task_index": task,
                "_ng_rollout_index": repeat,
                "_ng_environment_server": server,
                "reward": 0.1 + 0.2 * task + 0.1 * repeat + offset,
            }
            for task in range(3)
            for repeat in range(3)
        ]
        rollouts.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
        # Only attach the identities the collector adds around the real producer output.
        entry = {
            "agent_ref": {"name": f"{role}_agent"},
            "_ng_environment_server": server,
            **compute_aggregate_metrics(rows).model_dump(mode="json", exclude_none=True),
        }
        aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps([entry]))
        inputs = [{key: value for key, value in row.items() if key != "reward"} for row in rows]
        materialized_path_for(rollouts).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in inputs))
        return {"rollouts": str(rollouts), "environment_server": server}

    manifest = tmp_path / "suite.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "benchmarks": {
                    "benchmark": {"metrics": [{"name": "mean/reward", "low": 0, "high": 1, "higher_is_better": True}]}
                },
                "models": {
                    "model": {
                        "benchmark": {
                            "baseline": write_source(role="baseline", offset=0),
                            "candidate": write_source(role="candidate", offset=0.2),
                        }
                    }
                },
            }
        )
    )
    result = build_suite_comparison(SuiteComparisonConfig(suite_fpath=str(manifest)), "gym eval compare-suite")
    cell = result.cells[0]
    assert cell.baseline.repeat_scores == pytest.approx([30, 40, 50])
    assert cell.candidate.repeat_scores == pytest.approx([50, 60, 70])
    for role in ("baseline", "candidate"):
        run = getattr(cell, role)
        assert run.environment_server == f"{role}_env"
        assert run.repeat_indices == [0, 1, 2]
        assert run.repeat_count == 3
        assert run.coverage.status == "verified"
        assert run.coverage.recorded_task_count == run.coverage.planned_task_count == 3
        assert run.coverage.notes == []
    assert result.warnings == []
    for statistics in (cell.statistics, result.per_model[0].statistics, result.overall.statistics):
        assert statistics.baseline_mean == pytest.approx(40)
        assert statistics.candidate_mean == pytest.approx(60)
        assert statistics.delta == pytest.approx(20)
        assert statistics.ci_low < statistics.delta < statistics.ci_high
    assert cell.model_weight == cell.overall_weight == 1

    reports = tmp_path / "reports"
    write_suite_reports(result, reports, "both")
    restored = SuiteComparisonResult.model_validate_json((reports / "suite_compare_report.json").read_bytes())
    assert restored == result


def test_masked_repeat_token_accounting_preserves_scores_and_coverage(tmp_path: Path) -> None:
    results = []
    for with_perf in (False, True):
        directory = tmp_path / ("with_perf" if with_perf else "without_perf")
        sources = {
            benchmark: {
                role: _write_masked_source(
                    directory / benchmark / role,
                    offset=offset,
                    scored_repeats=(0, 2),
                    masked_repeats=(1,) if benchmark == "masked" else (),
                    with_perf=with_perf,
                )
                for role, offset in (("baseline", 0), ("candidate", 0.2))
            }
            for benchmark in ("masked", "complete")
        }
        aggregate_paths = [
            aggregate_metrics_path_for(Path(source["rollouts"]))
            for pair in sources.values()
            for source in pair.values()
        ]
        before = {path: path.read_bytes() for path in aggregate_paths}
        result = _compare(_write_manifest(directory, sources))
        masked = next(cell for cell in result.cells if cell.benchmark == "masked")
        complete = next(cell for cell in result.cells if cell.benchmark == "complete")
        assert masked.eligible is True
        assert masked.baseline.repeat_scores == pytest.approx([30, 50])
        assert masked.candidate.repeat_scores == pytest.approx([50, 70])
        for role in ("baseline", "candidate"):
            run = getattr(masked, role)
            assert run.repeat_indices == [0, 2]
            assert run.repeat_count == 2
            assert run.coverage.status == "incomplete"
            assert any("coverage/masked_rollouts=3" in note for note in run.coverage.notes)
            entry = orjson.loads(run.aggregate_metrics.read_bytes())[0]
            if with_perf:
                token_only = next(row for row in entry["repeat_level_metrics"] if row["_ng_rollout_index"] == 1)
                assert token_only["mean_completion_tokens"] == 20
                assert token_only["mean_tokens_per_turn"] == 10
                assert "sample_count" not in token_only
                assert "mean/reward" not in token_only
                assert run.unscored_repeat_indices == [1]
        assert masked.statistics.delta == pytest.approx(20)
        assert masked.model_weight == masked.overall_weight == 0.5
        assert complete.baseline.coverage.status == complete.candidate.coverage.status == "verified"
        assert complete.statistics.ci_low is not None
        for statistics in (masked.statistics, result.per_model[0].statistics, result.overall.statistics):
            assert statistics.ci_low is None
            assert statistics.ci_high is None
        assert result.overall.statistics.delta == pytest.approx(20)
        assert result.overall.weights == [0.5, 0.5]
        reports = directory / "reports"
        write_suite_reports(result, reports, "both")
        markdown = (reports / "suite_compare_report.md").read_text()
        assert "## Warnings" in markdown
        assert "coverage/masked_rollouts=3" in markdown
        restored = SuiteComparisonResult.model_validate_json((reports / "suite_compare_report.json").read_bytes())
        assert restored == result
        assert {path: path.read_bytes() for path in aggregate_paths} == before
        results.append(result)
    without_perf, with_perf = results
    assert without_perf.overall.statistics == with_perf.overall.statistics
    for before_cell, after_cell in zip(without_perf.cells, with_perf.cells, strict=True):
        assert before_cell.benchmark == after_cell.benchmark
        assert before_cell.statistics == after_cell.statistics
        assert before_cell.overall_weight == after_cell.overall_weight


def test_unscored_repeat_identity_alone_marks_coverage_incomplete(tmp_path: Path) -> None:
    sources = {
        "benchmark": {
            role: _write_masked_source(
                tmp_path / role,
                offset=offset,
                scored_repeats=(0, 2),
                masked_repeats=(1,),
                with_perf=True,
            )
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
    }
    with_token_rows = {}
    for source in sources["benchmark"].values():
        rollouts = Path(source["rollouts"])
        aggregate = aggregate_metrics_path_for(rollouts)
        entries = orjson.loads(aggregate.read_bytes())
        entry = entries[0]
        # Remove the independent completeness evidence so only the omitted token-only
        # repeat can suppress the interval; scored task summaries remain unchanged.
        entry["agent_metrics"] = {
            key: value
            for key, value in entry["agent_metrics"].items()
            if not key.startswith("coverage/") and key != "num_repeats"
        }
        materialized_path_for(rollouts).unlink()
        with_token_rows[aggregate] = orjson.dumps(entries)
        entry["repeat_level_metrics"] = [row for row in entry["repeat_level_metrics"] if row["_ng_rollout_index"] != 1]
        aggregate.write_bytes(orjson.dumps(entries))
    manifest = _write_manifest(tmp_path, sources)
    control = _compare(manifest)
    for run in (control.cells[0].baseline, control.cells[0].candidate):
        assert run.coverage.status == "unverified"
        assert run.unscored_repeat_indices == []
    assert control.overall.statistics.ci_low is not None

    for path, contents in with_token_rows.items():
        path.write_bytes(contents)
    result = _compare(manifest)
    cell = result.cells[0]
    note = "Repeats with no scored task observations are excluded from score calculations (indices: 1)."
    for role in ("baseline", "candidate"):
        run = getattr(cell, role)
        assert run.repeat_indices == [0, 2]
        assert run.repeat_count == 2
        assert run.unscored_repeat_indices == [1]
        assert run.coverage.status == "incomplete"
        assert set(run.coverage.notes) == set(getattr(control.cells[0], role).coverage.notes) | {note}
        assert f"model/benchmark {role}: {note}" in result.warnings
    assert cell.baseline.repeat_scores == pytest.approx([30, 50])
    assert cell.candidate.repeat_scores == pytest.approx([50, 70])
    for statistics in (cell.statistics, result.per_model[0].statistics, result.overall.statistics):
        assert statistics.delta == pytest.approx(20)
        assert statistics.ci_low is None
        assert statistics.ci_high is None
    assert result.overall.statistics.variance == control.overall.statistics.variance
    assert result.overall.weights == control.overall.weights == [1]
    assert {path: path.read_bytes() for path in with_token_rows} == with_token_rows


@pytest.mark.parametrize("with_perf", [False, True])
def test_single_scored_repeat_with_masked_repeats_is_excluded_and_reweights_suite(
    tmp_path: Path, with_perf: bool
) -> None:
    sources = {
        benchmark: {
            role: _write_masked_source(
                tmp_path / benchmark / role,
                offset=offset,
                scored_repeats=(0,) if benchmark == "single_scored" else (0, 2),
                masked_repeats=(1, 2) if benchmark == "single_scored" else (),
                with_perf=with_perf,
            )
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
        for benchmark in ("single_scored", "complete")
    }
    aggregate_paths = [
        aggregate_metrics_path_for(Path(source["rollouts"])) for pair in sources.values() for source in pair.values()
    ]
    before = {path: path.read_bytes() for path in aggregate_paths}
    result = _compare(_write_manifest(tmp_path, sources))
    excluded = next(cell for cell in result.cells if cell.benchmark == "single_scored")
    included = next(cell for cell in result.cells if cell.benchmark == "complete")
    assert excluded.eligible is False
    assert excluded.baseline.repeat_count == excluded.candidate.repeat_count == 1
    assert excluded.statistics is None
    assert excluded.model_weight == excluded.overall_weight == 0
    assert "at least two repeats" in excluded.exclusion_reason
    assert any("single_scored" in warning and "at least two repeats" in warning for warning in result.warnings)
    assert included.model_weight == included.overall_weight == 1
    assert result.overall.cell_ids == ["model/complete"]
    assert result.overall.weights == [1]
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.statistics.ci_low is not None
    write_suite_reports(result, tmp_path / "reports", "both")
    assert {path: path.read_bytes() for path in aggregate_paths} == before


@pytest.mark.parametrize("with_perf", [False, True])
def test_entirely_masked_producer_entry_is_excluded_without_regeneration(tmp_path: Path, with_perf: bool) -> None:
    sources = {
        benchmark: {
            role: _write_masked_source(
                tmp_path / benchmark / role,
                offset=offset,
                scored_repeats=() if benchmark == "unscored" else (0, 2),
                masked_repeats=(0, 1) if benchmark == "unscored" else (),
                with_perf=with_perf,
            )
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
        for benchmark in ("unscored", "complete")
    }
    aggregate_paths = [
        aggregate_metrics_path_for(Path(source["rollouts"])) for pair in sources.values() for source in pair.values()
    ]
    before = {path: path.read_bytes() for path in aggregate_paths}
    for source in sources["unscored"].values():
        entry = orjson.loads(aggregate_metrics_path_for(Path(source["rollouts"])).read_bytes())[0]
        assert entry["repeat_level_metrics"] == []
        assert entry["group_level_metrics"] == []
        assert entry["agent_metrics"]["coverage/measured_rollouts"] == 0
        assert entry["agent_metrics"]["coverage/masked_rollouts"] == 6
    result = _compare(_write_manifest(tmp_path, sources))
    excluded = next(cell for cell in result.cells if cell.benchmark == "unscored")
    assert excluded.eligible is False
    assert excluded.baseline.repeat_count == excluded.candidate.repeat_count == 0
    assert excluded.baseline.coverage.status == excluded.candidate.coverage.status == "incomplete"
    assert excluded.statistics is None
    assert excluded.model_weight == excluded.overall_weight == 0
    assert "at least two repeats" in excluded.exclusion_reason
    assert result.overall.cell_ids == ["model/complete"]
    assert result.overall.weights == [1]
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.statistics.ci_low is not None
    reports = tmp_path / "reports"
    write_suite_reports(result, reports, "both")
    restored = SuiteComparisonResult.model_validate_json((reports / "suite_compare_report.json").read_bytes())
    assert restored == result
    assert {path: path.read_bytes() for path in aggregate_paths} == before


def test_performance_metric_keeps_valid_token_only_repeat_observations(tmp_path: Path) -> None:
    sources = {
        "benchmark": {
            role: _write_masked_source(
                tmp_path / role,
                offset=offset,
                scored_repeats=(0, 2),
                masked_repeats=(1,),
                with_perf=True,
            )
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
    }
    manifest = _write_manifest(tmp_path, sources)
    payload = yaml.safe_load(manifest.read_text())
    payload["benchmarks"]["benchmark"]["metrics"] = [
        {"name": "mean_completion_tokens", "low": 0, "high": 100, "higher_is_better": True}
    ]
    manifest.write_text(yaml.safe_dump(payload))
    result = _compare(manifest)
    cell = result.cells[0]
    assert cell.eligible is True
    for run in (cell.baseline, cell.candidate):
        assert run.repeat_indices == [0, 1, 2]
        assert run.repeat_count == 3
        assert run.repeat_scores == pytest.approx([10, 20, 30])
        assert run.unscored_repeat_indices == []
        assert run.coverage.status == "incomplete"
    assert cell.statistics.delta == pytest.approx(0)
    assert cell.statistics.variance > 0
    assert result.overall.weights == [1]
    assert result.overall.statistics.ci_low is None


def test_single_valid_repeat_with_performance_is_excluded_with_verified_coverage(tmp_path: Path) -> None:
    sources = {
        benchmark: {
            role: _write_masked_source(
                tmp_path / benchmark / role,
                offset=offset,
                scored_repeats=(0,) if benchmark == "single" else (0, 2),
                with_perf=True,
            )
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
        for benchmark in ("single", "complete")
    }
    result = _compare(_write_manifest(tmp_path, sources))
    single = next(cell for cell in result.cells if cell.benchmark == "single")
    assert single.eligible is False
    for run in (single.baseline, single.candidate):
        assert run.repeat_count == 1
        assert run.unscored_repeat_indices == []
        assert run.coverage.status == "verified"
        assert run.coverage.notes == []
    assert "at least two repeats" in single.exclusion_reason
    assert result.overall.cell_ids == ["model/complete"]
    assert result.overall.weights == [1]
    assert result.overall.statistics.ci_low is not None


def test_score_rows_without_sample_counts_remain_valid_observations(tmp_path: Path) -> None:
    sources = {
        "benchmark": {
            role: _write_masked_source(tmp_path / role, offset=offset, scored_repeats=(0, 2), with_perf=True)
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
    }
    # Custom/older writers may omit counts while still storing genuine score metrics.
    for source in sources["benchmark"].values():
        aggregate = aggregate_metrics_path_for(Path(source["rollouts"]))
        entries = orjson.loads(aggregate.read_bytes())
        for row in entries[0]["repeat_level_metrics"]:
            row.pop("sample_count")
        aggregate.write_bytes(orjson.dumps(entries))
    result = _compare(_write_manifest(tmp_path, sources))
    cell = result.cells[0]
    assert cell.baseline.repeat_scores == pytest.approx([30, 50])
    assert cell.candidate.repeat_scores == pytest.approx([50, 70])
    for run in (cell.baseline, cell.candidate):
        assert run.repeat_count == 2
        assert run.repeat_indices == [0, 2]
        assert run.unscored_repeat_indices == []
        assert run.coverage.status == "verified"
    assert result.warnings == []
    assert result.overall.statistics.delta == pytest.approx(20)
    assert result.overall.statistics.ci_low is not None


def test_missing_reward_metric_on_scored_producer_row_still_fails(tmp_path: Path) -> None:
    sources = {
        "benchmark": {
            role: _write_masked_source(
                tmp_path / role,
                offset=offset,
                scored_repeats=(0, 1, 2),
                with_perf=True,
                missing_reward_repeat=2 if role == "candidate" else None,
            )
            for role, offset in (("baseline", 0), ("candidate", 0.2))
        }
    }
    candidate_path = aggregate_metrics_path_for(Path(sources["benchmark"]["candidate"]["rollouts"]))
    entry = orjson.loads(candidate_path.read_bytes())[0]
    missing = next(row for row in entry["repeat_level_metrics"] if row["_ng_rollout_index"] == 2)
    assert missing["sample_count"] == 3
    assert "mean/reward" not in missing
    with pytest.raises(ConfigError, match="repeat 2 requires numeric metric 'mean/reward'"):
        _compare(_write_manifest(tmp_path, sources))
