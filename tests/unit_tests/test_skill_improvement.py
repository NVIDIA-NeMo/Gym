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

import json
import shutil
from pathlib import Path

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.skill_improvement import (
    ImproveHistoryConfig,
    ImproveInitConfig,
    ImproveKeepConfig,
    ImproveRecordConfig,
    ImproveRevertConfig,
    ImproveRoundConfig,
    RoundStatus,
    compare_key_metrics,
    init_workspace,
    keep_round,
    list_history,
    load_aggregate_metrics_key_metrics,
    record_round,
    revert_live_skills,
    start_round,
)


def _write_skill(skills_dir: Path, name: str, body: str = "# Body\n") -> None:
    skill_dir = skills_dir / name
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test skill.\n---\n{body}")


def _write_aggregate_metrics(path: Path, mean_reward: float) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "agent_ref": {"name": "my_agent"},
                    "agent_metrics": {"mean/reward": mean_reward},
                    "key_metrics": {"mean/reward": mean_reward},
                    "group_level_metrics": [],
                }
            ]
        )
    )


class TestMetricHelpers:
    def test_load_and_compare_key_metrics(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline_aggregate_metrics.json"
        _write_aggregate_metrics(baseline, 0.5)
        loaded = load_aggregate_metrics_key_metrics(baseline)
        assert loaded["mean/reward"] == pytest.approx(0.5)

        comparisons = compare_key_metrics({"mean/reward": 0.5}, {"mean/reward": 0.75})
        assert comparisons[0].delta == pytest.approx(0.25)


class TestImproveWorkspace:
    def test_init_round_record_keep_revert(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        live_skills = tmp_path / "live_skills"
        _write_skill(live_skills, "baseline_skill")

        baseline_metrics = tmp_path / "baseline_metrics.json"
        _write_aggregate_metrics(baseline_metrics, 0.4)

        init_workspace(
            ImproveInitConfig(
                workspace=str(workspace),
                skills=str(live_skills),
                baseline_metrics=str(baseline_metrics),
            )
        )

        assert (workspace / "improve.yaml").exists()
        assert (workspace / "accepted" / "skills" / "baseline_skill" / "SKILL.md").exists()

        # Edit live skills for round 1
        _write_skill(live_skills, "baseline_skill", body="# Improved body\n")
        round_id, skills_ref = start_round(ImproveRoundConfig(workspace=str(workspace)))
        assert round_id == "0001"

        round_metrics = tmp_path / "round_metrics.json"
        _write_aggregate_metrics(round_metrics, 0.8)
        manifest = record_round(
            ImproveRecordConfig(
                workspace=str(workspace),
                aggregate_metrics=str(round_metrics),
            )
        )
        assert manifest.comparison is not None
        assert manifest.comparison.metrics[0].delta == pytest.approx(0.4)

        keep_round(
            ImproveKeepConfig(
                workspace=str(workspace),
                round_id=round_id,
                copy_live=True,
            )
        )
        kept = json.loads((workspace / "rounds" / round_id / "manifest.json").read_text())
        assert kept["status"] == RoundStatus.KEPT.value

        # Mutate live skills, then revert
        _write_skill(live_skills, "baseline_skill", body="# Bad edit\n")
        revert_live_skills(ImproveRevertConfig(workspace=str(workspace), force=True))
        reverted_body = (live_skills / "baseline_skill" / "SKILL.md").read_text()
        assert "Improved body" in reverted_body

    def test_keep_refuses_dirty_live_skills_without_force(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        live_skills = tmp_path / "live_skills"
        _write_skill(live_skills, "s")

        baseline_metrics = tmp_path / "baseline_metrics.json"
        _write_aggregate_metrics(baseline_metrics, 0.1)

        init_workspace(
            ImproveInitConfig(
                workspace=str(workspace),
                skills=str(live_skills),
                baseline_metrics=str(baseline_metrics),
            )
        )

        round_id, _ = start_round(ImproveRoundConfig(workspace=str(workspace)))
        round_metrics = tmp_path / "round_metrics.json"
        _write_aggregate_metrics(round_metrics, 0.2)
        record_round(ImproveRecordConfig(workspace=str(workspace), aggregate_metrics=str(round_metrics)))

        _write_skill(live_skills, "s", body="# changed after round\n")
        with pytest.raises(ConfigError, match="differ from round"):
            keep_round(ImproveKeepConfig(workspace=str(workspace), round_id=round_id, copy_live=True))

    def test_history_lists_rounds(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        live_skills = tmp_path / "live_skills"
        _write_skill(live_skills, "s")
        baseline_metrics = tmp_path / "baseline_metrics.json"
        _write_aggregate_metrics(baseline_metrics, 0.1)

        init_workspace(
            ImproveInitConfig(
                workspace=str(workspace),
                skills=str(live_skills),
                baseline_metrics=str(baseline_metrics),
            )
        )
        start_round(ImproveRoundConfig(workspace=str(workspace)))

        rounds = list_history(ImproveHistoryConfig(workspace=str(workspace)))
        assert len(rounds) == 1
        assert rounds[0].round_id == "0001"

    def test_init_rejects_nonempty_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "existing.txt").write_text("x")

        live_skills = tmp_path / "live_skills"
        _write_skill(live_skills, "s")
        baseline_metrics = tmp_path / "baseline_metrics.json"
        _write_aggregate_metrics(baseline_metrics, 0.1)

        with pytest.raises(ConfigError, match="not empty"):
            init_workspace(
                ImproveInitConfig(
                    workspace=str(workspace),
                    skills=str(live_skills),
                    baseline_metrics=str(baseline_metrics),
                )
            )
