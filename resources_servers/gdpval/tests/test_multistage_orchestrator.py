# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the standard-flow multi-stage ELO orchestrator (no servers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    build_stage_rows,
    find_gdpval_reference_elos,
    index_rows_by_task,
    parse_multistage_config,
    row_task_id,
    run_multistage_stages,
    tag_results,
    write_rollouts,
)


REF_ELOS = {"a": 1000.0, "b": 1200.0, "c": 1400.0, "d": 1600.0}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestParseConfig:
    def test_parses_mapping_stages(self) -> None:
        cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [{"num_tasks": 5}, {"num_tasks": 88, "num_models": 4, "seed": 7}],
                "nested_tasks": True,
                "column": "sector",
            }
        )
        assert cfg.enabled is True
        assert [(s.num_tasks, s.num_models, s.seed) for s in cfg.stages] == [(5, None, None), (88, 4, 7)]
        assert cfg.nested_tasks is True
        assert cfg.column == ["sector"]
        # Deliverable reuse across stages is on by default.
        assert cfg.reuse_cached_deliverables is True

    def test_reuse_cached_deliverables_can_be_disabled(self) -> None:
        cfg = parse_multistage_config({"enabled": True, "stages": ["5"], "reuse_cached_deliverables": False})
        assert cfg.reuse_cached_deliverables is False

    def test_parses_string_stages(self) -> None:
        cfg = parse_multistage_config({"enabled": True, "stages": ["5", "88:4", "100:2:9"]})
        assert [(s.num_tasks, s.num_models, s.seed) for s in cfg.stages] == [
            (5, None, None),
            (88, 4, None),
            (100, 2, 9),
        ]

    def test_empty_stages_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_multistage_config({"enabled": True, "stages": []})


class TestFindReferenceElos:
    def test_extracts_from_nel_style_config(self) -> None:
        global_config = {
            "some_model_server": {"responses_api_models": {"vllm_model": {"model": "x"}}},
            "gdpval_resources_server": {
                "resources_servers": {
                    "gdpval": {
                        "reward_mode": "comparison",
                        "reference_models": {
                            "glm51": {"deliverables_dir": "/d/glm", "elo": 1535},
                            "kimi_k25": {"deliverables_dir": "/d/kimi", "elo": 1284},
                        },
                    }
                }
            },
        }
        assert find_gdpval_reference_elos(global_config) == {"glm51": 1535.0, "kimi_k25": 1284.0}

    def test_returns_empty_when_absent(self) -> None:
        assert find_gdpval_reference_elos({"foo": {"bar": 1}}) == {}


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _materialized_rows(task_ids: List[str], repeats: int = 1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t_idx, tid in enumerate(task_ids):
        for r_idx in range(repeats):
            rows.append(
                {
                    TASK_INDEX_KEY_NAME: t_idx,
                    ROLLOUT_INDEX_KEY_NAME: r_idx,
                    AGENT_REF_KEY_NAME: {"name": "gdpval_stirrup_agent"},
                    "task_id": tid,
                    "responses_create_params": {"input": [], "metadata": {"task_id": tid}},
                }
            )
    return rows


class TestRowHelpers:
    def test_row_task_id_top_level_and_metadata(self) -> None:
        assert row_task_id({"task_id": "x"}) == "x"
        assert row_task_id({"responses_create_params": {"metadata": {"task_id": "y"}}}) == "y"
        assert row_task_id({"responses_create_params": {}}) is None

    def test_index_rows_by_task_groups_repeats(self) -> None:
        rows = _materialized_rows(["t0", "t1"], repeats=2)
        by_task = index_rows_by_task(rows)
        assert set(by_task) == {"t0", "t1"}
        assert len(by_task["t0"]) == 2

    def test_build_stage_rows_tags_and_preserves_indices(self) -> None:
        by_task = index_rows_by_task(_materialized_rows(["t0", "t1"], repeats=2))
        rows = build_stage_rows(by_task, ["t0", "t1"], ["b", "c"], stage_index=2)
        assert len(rows) == 4  # 2 tasks x 2 repeats
        for row in rows:
            assert row["reference_ids"] == ["b", "c"]
            assert row["stage_index"] == 2
        # Indices are preserved (no per-stage offset) so the rollout index keeps
        # matching the on-disk deliverable repeat dir; stage_index is the
        # disambiguator across stages.
        assert {(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in rows} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }

    def test_build_stage_rows_skips_unknown_tasks(self) -> None:
        by_task = index_rows_by_task(_materialized_rows(["t0"]))
        rows = build_stage_rows(by_task, ["t0", "missing"], ["a"], stage_index=0)
        assert len(rows) == 1

    def test_build_stage_rows_tags_reuse_for_produced(self) -> None:
        by_task = index_rows_by_task(_materialized_rows(["t0", "t1"], repeats=2))
        # t0's two repeats were already produced; t1 is new this stage.
        produced = {("t0", 0), ("t0", 1)}
        rows = build_stage_rows(by_task, ["t0", "t1"], ["a"], stage_index=1, produced=produced)
        reuse = {(r["task_id"], r.get("reuse_cached_deliverable", False)) for r in rows}
        assert ("t0", True) in reuse
        assert ("t1", False) in reuse

    def test_build_stage_rows_no_reuse_without_produced(self) -> None:
        by_task = index_rows_by_task(_materialized_rows(["t0"]))
        rows = build_stage_rows(by_task, ["t0"], ["a"], stage_index=0)
        assert all("reuse_cached_deliverable" not in r for r in rows)

    def test_tag_results_stamps_identity(self) -> None:
        row = {
            TASK_INDEX_KEY_NAME: 3,
            ROLLOUT_INDEX_KEY_NAME: 7,
            AGENT_REF_KEY_NAME: {"name": "ag"},
            "task_id": "t3",
        }
        result = {"per_reference": {}, "reward": 1.0}
        tagged = tag_results([(row, result)], stage_index=1)
        assert tagged[0][TASK_INDEX_KEY_NAME] == 3
        assert tagged[0][ROLLOUT_INDEX_KEY_NAME] == 7
        assert tagged[0]["stage_index"] == 1
        assert tagged[0]["task_id"] == "t3"


# ---------------------------------------------------------------------------
# Staged loop
# ---------------------------------------------------------------------------


def _distribution(task_ids: List[str]) -> Dict[str, Dict[str, object]]:
    return {"grp": {"percentage": 1.0, "task_ids": list(task_ids)}}


def _fake_run_rollouts_factory(target_elo: float = 1300.0):
    """Eval beats refs below ``target_elo`` and loses to those above ⇒ MLE lands
    near ``target_elo``, so stage-2 reference selection zooms in around it."""

    async def fake_run_rollouts(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for row in rows:
            per_ref: Dict[str, Any] = {}
            for rid in row["reference_ids"]:
                elo = REF_ELOS[rid]
                if elo < target_elo:
                    per_ref[rid] = {"wins": 9, "losses": 1, "ties": 0, "reference_elo": elo}
                else:
                    per_ref[rid] = {"wins": 1, "losses": 9, "ties": 0, "reference_elo": elo}
            result = {
                "task_id": row["task_id"],
                "per_reference": per_ref,
                "total_wins": sum(p["wins"] for p in per_ref.values()),
                "total_losses": sum(p["losses"] for p in per_ref.values()),
                "total_ties": 0,
            }
            pairs.append((row, result))
        return pairs

    return fake_run_rollouts


class TestRunStages:
    async def test_threads_elo_and_shrinks_references(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3", "5:2"]}).stages,
            seed=0,
        )
        all_results, summaries = await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            rows,
            _fake_run_rollouts_factory(),
        )

        # Stage 0 uses all references; stage 1 shrinks to the 2 closest to the
        # stage-0 estimate (~1300 ⇒ b=1200, c=1400).
        assert summaries[0]["reference_ids"] == ["a", "b", "c", "d"]
        assert summaries[1]["reference_ids"] == ["b", "c"]
        assert summaries[0]["eval_elo"] is not None
        assert summaries[1]["eval_elo"] is not None

        # All rollouts accumulated and tagged with their stage.
        assert len(all_results) == summaries[0]["num_rollouts"] + summaries[1]["num_rollouts"]
        assert {r["stage_index"] for r in all_results} == {0, 1}

        # Rows are identified by (stage_index, task_index, rollout_index): the
        # raw (task_index, rollout_index) may recur across stages (same rollout
        # judged against a different reference subset), but adding stage_index
        # makes every row unique. Indices are never offset.
        keys = [(r["stage_index"], r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in all_results]
        assert len(keys) == len(set(keys))

    async def test_reuses_deliverables_across_stages(self) -> None:
        # Nested tasks ⇒ stage 1 ⊇ stage 0, so every stage-0 task recurs in
        # stage 1 and must be reused (not re-run) there.
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3", "6:2"]}).stages,
            seed=0,
            nested_tasks=True,
        )

        seen_reuse: List[Tuple[int, str, bool]] = []
        base_run = _fake_run_rollouts_factory()

        async def recording_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                seen_reuse.append((r["stage_index"], r["task_id"], bool(r.get("reuse_cached_deliverable"))))
            return await base_run(rows_in)

        _, summaries = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, recording_run)

        stage0_tasks = {t for s, t, _ in seen_reuse if s == 0}
        # No reuse in stage 0 (nothing produced yet).
        assert all(not reused for s, _, reused in seen_reuse if s == 0)
        # Every stage-1 row for a stage-0 task is flagged for reuse; brand-new
        # stage-1 tasks are produced fresh.
        for stage, task, reused in seen_reuse:
            if stage == 1:
                assert reused == (task in stage0_tasks)
        assert summaries[1]["num_reused"] == len(stage0_tasks)

    async def test_reuse_disabled_reruns_every_stage(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3", "6:2"]}).stages,
            seed=0,
            nested_tasks=True,
            reuse_cached_deliverables=False,
        )
        _, summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, _fake_run_rollouts_factory()
        )
        assert summaries[0]["num_reused"] == 0
        assert summaries[1]["num_reused"] == 0

    async def test_emits_lifecycle_events(self) -> None:
        task_ids = [f"t{i}" for i in range(6)]
        events: List[str] = []
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["2", "3:2"]}).stages,
            seed=1,
        )
        await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            _materialized_rows(task_ids),
            _fake_run_rollouts_factory(),
            on_event=lambda name, data: events.append(name),
        )
        assert events[0] == "planned"
        assert events.count("stage_start") == 2
        assert events.count("stage_end") == 2


class TestWriteRollouts:
    def test_writes_sorted_jsonl(self, tmp_path: Path) -> None:
        results = [
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t1"},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 5, "task_id": "t0"},
        ]
        out = write_rollouts(results, tmp_path / "rollouts.jsonl")
        lines = [json.loads(line) for line in out.read_text().splitlines()]
        assert [line["task_id"] for line in lines] == ["t0", "t1"]
