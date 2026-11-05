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

from nemo_gym.eval.comparison.suite.report import write_suite_reports
from nemo_gym.eval.comparison.suite.runner import build_suite_comparison
from nemo_gym.eval.comparison.suite.schema import SuiteComparisonConfig, SuiteComparisonResult
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, materialized_path_for


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
