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
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

import nemo_gym.rollout_collection as rollout_collection_module
import resources_servers.gdpval.multistage_orchestrator as multistage_module
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
)
from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    StageResume,
    _prepare_resume,
    aggregate_metrics_path_for,
    append_journal_record,
    build_file_resume,
    build_stage_rows,
    compute_fingerprint,
    find_gdpval_reference_elos,
    index_rows_by_task,
    journal_path_for,
    load_failure_attempts,
    load_failure_timings,
    load_gated_keys,
    load_latest_attempt_dispositions,
    load_latest_failures,
    load_persisted_rows,
    load_reuse_cached_keys,
    parse_multistage_config,
    read_journal,
    route_stage_rows,
    row_task_id,
    run_e2e_multistage,
    run_multistage_stages,
    tag_results,
    write_rollouts,
)


REF_ELOS = {"a": 1000.0, "b": 1200.0, "c": 1400.0, "d": 1600.0}


def _runtime_components_with_bind_addresses(host_prefix: str, port_offset: int) -> Dict[str, Any]:
    return {
        "agent": {
            "responses_api_agents": {
                "stirrup": {
                    "host": f"{host_prefix}-agent",
                    "port": 8001 + port_offset,
                    "task": "gdpval",
                    "worker_target": {"host": "sandbox.internal", "port": 9000},
                }
            }
        },
        "policy": {
            "responses_api_models": {
                "vllm": {
                    "host": f"{host_prefix}-policy",
                    "port": 8002 + port_offset,
                    "model": "/checkpoints/policy-a",
                }
            }
        },
        "resources": {
            "resources_servers": {
                "gdpval": {
                    "host": f"{host_prefix}-resources",
                    "port": 8003 + port_offset,
                    "num_comparison_trials": 2,
                }
            }
        },
    }


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

    def test_parses_partial_completion_policy(self) -> None:
        cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [
                    {
                        "num_tasks": 45,
                        "partial_completion": {
                            "min_success_fraction": 0.9,
                            "min_per_reference_success_fraction": 0.5,
                            "min_successful_rows_per_reference": 1,
                        },
                    },
                    {"num_tasks": 220, "num_models": 4},
                ],
            }
        )

        policy = cfg.stages[0].partial_completion
        assert policy is not None
        assert policy.min_success_fraction == 0.9
        assert policy.min_per_reference_success_fraction == 0.5
        assert policy.min_successful_rows_per_reference == 1
        assert cfg.stages[1].partial_completion is None

    @pytest.mark.parametrize(
        "partial_completion",
        [
            {"min_success_fraction": 0},
            {"min_success_fraction": True},
            {"min_per_reference_success_fraction": 1.1},
            {"min_successful_rows_per_reference": 0},
            {"min_successful_rows_per_reference": True},
            {"allowed_failure_classes": ["transient"]},
        ],
    )
    def test_rejects_unsafe_partial_completion_policy(self, partial_completion: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {"num_tasks": 5, "partial_completion": partial_completion},
                        {"num_tasks": 10, "num_models": 2},
                    ],
                }
            )

    def test_rejects_partial_completion_on_final_stage(self) -> None:
        with pytest.raises(ValueError, match="non-final calibration stages"):
            parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {"num_tasks": 5},
                        {"num_tasks": 10, "partial_completion": {"min_success_fraction": 0.9}},
                    ],
                }
            )


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
        # Each task is judged against a single assigned reference.
        rows = build_stage_rows(by_task, {"t0": "b", "t1": "c"}, stage_index=2)
        assert len(rows) == 4  # 2 tasks x 2 repeats
        ref_by_task = {r["task_id"]: r["reference_ids"] for r in rows}
        assert ref_by_task["t0"] == ["b"]
        assert ref_by_task["t1"] == ["c"]
        for row in rows:
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
        rows = build_stage_rows(by_task, {"t0": "a", "missing": "a"}, stage_index=0)
        assert len(rows) == 1

    def test_build_stage_rows_tags_reuse_for_produced(self) -> None:
        by_task = index_rows_by_task(_materialized_rows(["t0", "t1"], repeats=2))
        # t0's two repeats were already produced; t1 is new this stage.
        produced = {("t0", 0), ("t0", 1)}
        rows = build_stage_rows(by_task, {"t0": "a", "t1": "a"}, stage_index=1, produced=produced)
        reuse = {(r["task_id"], r.get("reuse_cached_deliverable", False)) for r in rows}
        assert ("t0", True) in reuse
        assert ("t1", False) in reuse

    def test_build_stage_rows_no_reuse_without_produced(self) -> None:
        by_task = index_rows_by_task(_materialized_rows(["t0"]))
        rows = build_stage_rows(by_task, {"t0": "a"}, stage_index=0)
        assert all("reuse_cached_deliverable" not in r for r in rows)

    def test_tag_results_stamps_identity(self) -> None:
        row = {
            TASK_INDEX_KEY_NAME: 3,
            ROLLOUT_INDEX_KEY_NAME: 7,
            AGENT_REF_KEY_NAME: {"name": "ag"},
            "task_id": "t3",
        }
        result = {"per_reference": {}, "reward": 1.0}
        tagged = tag_results(
            [(row, result)],
            stage_index=1,
            expected_final_stage_index=2,
            expected_stage_row_count=17,
        )
        assert tagged[0][TASK_INDEX_KEY_NAME] == 3
        assert tagged[0][ROLLOUT_INDEX_KEY_NAME] == 7
        assert tagged[0]["stage_index"] == 1
        assert tagged[0]["expected_final_stage_index"] == 2
        assert tagged[0]["expected_stage_row_count"] == 17
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
        task_ids = [f"t{i}" for i in range(20)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            # Full task set each stage (robust ELO); stage 1 narrows to 2 refs.
            stages=parse_multistage_config({"enabled": True, "stages": [{"num_models": 4}, {"num_models": 2}]}).stages,
            seed=0,
        )
        all_results, summaries = await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            rows,
            _fake_run_rollouts_factory(),
        )

        assert all(row["expected_final_stage_index"] == 1 for row in all_results)
        assert all(row["expected_stage_row_count"] == 20 for row in all_results)

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

    async def test_default_full_dataset_and_single_reference_per_task(self) -> None:
        # With num_tasks omitted, every stage judges the FULL task set, and each
        # task's row carries a single reference drawn from the stage's included set.
        task_ids = [f"t{i}" for i in range(40)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            # No num_tasks ⇒ full dataset; stage 1 narrows to 2 references.
            stages=parse_multistage_config({"enabled": True, "stages": [{"num_models": 4}, {"num_models": 2}]}).stages,
            seed=0,
        )

        seen: List[Tuple[int, str, List[str]]] = []
        base_run = _fake_run_rollouts_factory()

        async def recording_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                seen.append((r["stage_index"], r["task_id"], list(r["reference_ids"])))
            return await base_run(rows_in)

        _, summaries = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, recording_run)

        for stage_index, pool in ((0, {"a", "b", "c", "d"}), (1, {"b", "c"})):
            stage_seen = [(t, refs) for s, t, refs in seen if s == stage_index]
            # Full dataset: every task appears exactly once this stage.
            assert {t for t, _ in stage_seen} == set(task_ids)
            # Exactly one reference per task, always from the included pool.
            assert all(len(refs) == 1 for _, refs in stage_seen)
            assert {refs[0] for _, refs in stage_seen}.issubset(pool)
            # With 40 tasks over the pool, every included reference is used.
            assert {refs[0] for _, refs in stage_seen} == pool
            assert summaries[stage_index]["num_tasks"] == len(task_ids)

    async def test_num_tasks_limits_sampled_task_count(self) -> None:
        # An explicit num_tasks samples exactly that many tasks for the stage.
        task_ids = [f"t{i}" for i in range(40)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            # Stage 0 samples 6 tasks; stage 1 defaults to the full 40.
            stages=parse_multistage_config({"enabled": True, "stages": [{"num_tasks": 6}, {"num_models": 2}]}).stages,
            seed=0,
        )

        seen: Dict[int, set] = {0: set(), 1: set()}
        base_run = _fake_run_rollouts_factory()

        async def recording_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                seen[r["stage_index"]].add(r["task_id"])
            return await base_run(rows_in)

        _, summaries = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, recording_run)

        assert len(seen[0]) == 6
        assert summaries[0]["num_tasks"] == 6
        assert len(seen[1]) == 40
        assert summaries[1]["num_tasks"] == 40

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

    def test_fresh_file_uses_normal_umask_permissions(self, tmp_path: Path) -> None:
        control = tmp_path / "normal.jsonl"
        control.write_bytes(b"normal\n")

        out = write_rollouts(
            [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}],
            tmp_path / "rollouts.jsonl",
        )

        assert out.stat().st_mode & 0o777 == control.stat().st_mode & 0o777

    def test_dedupes_by_stage_task_rollout(self, tmp_path: Path) -> None:
        results = [
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "old"},
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "new"},
            {"stage_index": 1, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "other"},
        ]
        out = write_rollouts(results, tmp_path / "rollouts.jsonl")
        lines = [json.loads(line) for line in out.read_text().splitlines()]
        # Dedup keeps the last write per (stage, task, rollout); stage 1 is distinct.
        assert [line["task_id"] for line in lines] == ["new", "other"]

    def test_failed_rewrite_preserves_previous_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "rollouts.jsonl"
        original = b'{"previous":true}\n'
        out.write_bytes(original)
        real_dumps = multistage_module.orjson.dumps
        calls = 0

        def fail_on_second_row(value, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected serialization failure")
            return real_dumps(value, *args, **kwargs)

        monkeypatch.setattr(multistage_module.orjson, "dumps", fail_on_second_row)
        rows = [
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
        ]

        with pytest.raises(RuntimeError, match="injected serialization failure"):
            write_rollouts(rows, out)

        assert out.read_bytes() == original
        assert list(tmp_path.glob(".rollouts.jsonl.merge-*")) == []


# ---------------------------------------------------------------------------
# Resume seam (pure, in-memory)
# ---------------------------------------------------------------------------


class RecordingResume(StageResume):
    """In-memory StageResume that records callback invocations.

    ``gated_keys`` defaults to the successes in ``rows_by_stage``; pass it
    explicitly to model terminal / max-attempt gating from the sidecar.
    """

    def __init__(
        self,
        plans=None,
        outcomes=None,
        rows_by_stage=None,
        gated_keys=None,
        elapsed_by_stage=None,
        reuse_cached_keys=None,
        attempts_by_stage=None,
        latest_failures_by_stage=None,
        latest_attempt_dispositions_by_stage=None,
    ) -> None:
        self.planned: List[Tuple[int, dict]] = []
        self.completed: List[Tuple[int, dict]] = []
        self.appended: Dict[int, List[Dict[str, Any]]] = {}
        self.restarted: List[int] = []
        rows_by_stage = dict(rows_by_stage or {})
        if gated_keys is None:
            gated_keys = {
                i: {(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in rows}
                for i, rows in rows_by_stage.items()
            }
        super().__init__(
            plans=dict(plans or {}),
            outcomes=dict(outcomes or {}),
            rows_by_stage=rows_by_stage,
            gated_keys=dict(gated_keys),
            on_plan=lambda i, p: self.planned.append((i, p)),
            on_outcome=lambda i, o: self.completed.append((i, o)),
            on_rows=lambda i, r: self.appended.setdefault(i, []).extend(r),
            on_restart=self.restarted.append,
            elapsed_by_stage=dict(elapsed_by_stage or {}),
            reuse_cached_keys=dict(reuse_cached_keys or {}),
            attempts_by_stage=dict(attempts_by_stage or {}),
            latest_failures_by_stage=dict(latest_failures_by_stage or {}),
            latest_attempt_dispositions_by_stage=dict(latest_attempt_dispositions_by_stage or {}),
        )


def _two_stage_cfg(seed=0, nested=False) -> MultiStageRunConfig:
    return MultiStageRunConfig(
        enabled=True,
        stages=parse_multistage_config({"enabled": True, "stages": ["3", "5:2"]}).stages,
        seed=seed,
        nested_tasks=nested,
    )


class TestResumeSeam:
    async def test_resume_none_is_backward_compatible(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = _two_stage_cfg()
        run = _fake_run_rollouts_factory()
        base = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, run)
        again = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, run, resume=None)
        # Byte-for-byte identical result rows and summaries.
        assert base[0] == again[0]
        assert base[1] == again[1]

    async def test_complete_stage_skips_dispatch_and_threads_elo(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = _two_stage_cfg()

        # First pass with no resume produces stage-0 tagged rows we can cache.
        full_run = _fake_run_rollouts_factory()
        all_results, base_summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, full_run
        )
        stage0_rows = [r for r in all_results if r["stage_index"] == 0]
        stage0_plan = {
            "stage_index": 0,
            "reference_ids": base_summaries[0]["reference_ids"],
            "task_ids": list(dict.fromkeys(row["task_id"] for row in stage0_rows)),
        }
        resume = RecordingResume(
            plans={0: stage0_plan},
            outcomes={0: {"stage_index": 0, "status": "complete", "eval_elo": base_summaries[0]["eval_elo"]}},
            rows_by_stage={0: stage0_rows},
        )

        dispatched: List[int] = []

        async def counting_run(rows_in: List[Dict[str, Any]]):
            dispatched.append(len(rows_in))
            return await full_run(rows_in)

        _, summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, counting_run, resume=resume
        )

        # Stage 0 was not dispatched; only stage 1 ran.
        assert len(dispatched) == 1
        # Stage 0 ELO was re-fit from cached rows and threaded into stage 1's
        # reference selection (same as the original full run).
        assert summaries[0]["cached"] is True
        assert summaries[1]["reference_ids"] == base_summaries[1]["reference_ids"]

    async def test_interrupted_stage_redispatches_only_missing(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = _two_stage_cfg()
        full_run = _fake_run_rollouts_factory()

        all_results, base_summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, full_run
        )
        # Cache all but one of stage 0's successful rows: the missing one must be
        # re-dispatched; the rest must not.
        stage0_rows = [r for r in all_results if r["stage_index"] == 0]
        stage0_task_ids = list(dict.fromkeys(r["task_id"] for r in stage0_rows))
        cached_stage0 = stage0_rows[:-1]
        missing_key = (stage0_rows[-1][TASK_INDEX_KEY_NAME], stage0_rows[-1][ROLLOUT_INDEX_KEY_NAME])

        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "reference_ids": base_summaries[0]["reference_ids"],
                    "task_ids": stage0_task_ids,
                }
            },
            rows_by_stage={0: cached_stage0},
        )

        dispatched_keys: List[Tuple[int, int]] = []

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                if r["stage_index"] == 0:
                    dispatched_keys.append((r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
            return await full_run(rows_in)

        _, summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, capturing_run, resume=resume
        )

        assert dispatched_keys == [missing_key]
        # Only the newly dispatched row is passed to on_rows.
        assert len(resume.appended[0]) == 1
        # Final stage-0 result count equals the full run (cached + re-dispatched).
        assert summaries[0]["num_rollouts"] == base_summaries[0]["num_rollouts"]

    async def test_completed_cached_stage_stamps_full_expected_row_count(self) -> None:
        task_ids = ["t0", "t1"]
        rows = _materialized_rows(task_ids, repeats=2)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["2"]}).stages,
            seed=0,
        )
        full_results, summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, _fake_run_rollouts_factory()
        )
        # Model a completed stage with one persisted success and the other rows
        # accounted for by terminal/max-attempt sidecars. The success must still
        # declare the full planned cardinality, not the cached success count.
        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "reference_ids": summaries[0]["reference_ids"],
                    "task_ids": task_ids,
                }
            },
            outcomes={0: {"stage_index": 0, "status": "complete"}},
            rows_by_stage={0: [full_results[0]]},
            gated_keys={0: {(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in full_results}},
        )

        async def no_dispatch(rows_in: List[Dict[str, Any]]):
            raise AssertionError("completed cached stage must not dispatch")

        cached_results, _ = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, no_dispatch, resume=resume
        )

        assert len(cached_results) == 1
        assert cached_results[0]["expected_stage_row_count"] == 4
        assert cached_results[0]["expected_final_stage_index"] == 0

    async def test_completed_stage_with_missing_cached_key_is_reopened(self) -> None:
        task_ids = ["t0", "t1"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["2"]}).stages,
            seed=0,
        )
        full_run = _fake_run_rollouts_factory()
        full_results, summaries = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, full_run)
        missing = full_results[-1]
        missing_key = (missing[TASK_INDEX_KEY_NAME], missing[ROLLOUT_INDEX_KEY_NAME])
        cached = full_results[:-1]
        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "reference_ids": summaries[0]["reference_ids"],
                    "task_ids": task_ids,
                }
            },
            outcomes={0: {"stage_index": 0, "status": "complete"}},
            rows_by_stage={0: cached},
        )
        dispatched: List[Tuple[int, int]] = []

        async def capture(rows_in: List[Dict[str, Any]]):
            dispatched.extend((row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in rows_in)
            return await full_run(rows_in)

        results, result_summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, capture, resume=resume
        )

        assert dispatched == [missing_key]
        assert len(results) == len(full_results)
        assert result_summaries[0].get("cached") is not True

    async def test_reopened_calibration_invalidates_dependent_stage(self) -> None:
        task_ids = ["t0", "t1", "t2"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3", "3:2"]}).stages,
            seed=0,
        )
        baseline_resume = RecordingResume()
        successful_run = _fake_run_rollouts_factory()
        baseline_results, _ = await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            rows,
            successful_run,
            resume=baseline_resume,
        )
        rows_by_stage = {
            stage_index: [row for row in baseline_results if row["stage_index"] == stage_index]
            for stage_index in (0, 1)
        }
        missing = rows_by_stage[0].pop()
        missing_key = (missing[TASK_INDEX_KEY_NAME], missing[ROLLOUT_INDEX_KEY_NAME])
        resumed = RecordingResume(
            plans=dict(baseline_resume.planned),
            outcomes=dict(baseline_resume.completed),
            rows_by_stage=rows_by_stage,
        )
        dispatched: List[Tuple[int, Tuple[int, int]]] = []

        async def capture(rows_in: List[Dict[str, Any]]):
            dispatched.extend(
                (
                    row["stage_index"],
                    (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]),
                )
                for row in rows_in
            )
            return await successful_run(rows_in)

        await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            rows,
            capture,
            resume=resumed,
        )

        assert resumed.restarted == [0]
        assert [key for stage, key in dispatched if stage == 0] == [missing_key]
        assert len([key for stage, key in dispatched if stage == 1]) == 3

    async def test_resumed_empty_fit_blocks_adaptive_next_stage(self) -> None:
        task_ids = [f"t{i}" for i in range(6)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["2:4", "2:2", "2:1"]}).stages,
            seed=0,
            reuse_cached_deliverables=False,
        )
        baseline_resume = RecordingResume()
        successful_run = _fake_run_rollouts_factory()

        async def terminal_middle_stage(rows_in: List[Dict[str, Any]]):
            if rows_in and rows_in[0]["stage_index"] == 1:
                return [
                    (
                        row,
                        {
                            NG_FAILURE_CLASS_KEY: "skipped",
                            NG_TERMINAL_KEY: True,
                        },
                    )
                    for row in rows_in
                ]
            return await successful_run(rows_in)

        baseline_results, baseline_summaries = await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            rows,
            terminal_middle_stage,
            resume=baseline_resume,
        )
        assert baseline_summaries[0]["eval_elo"] is not None
        assert baseline_summaries[1]["eval_elo"] is None

        stage0_rows = [row for row in baseline_results if row["stage_index"] == 0]
        stage0_keys = {(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in stage0_rows}
        stage1_keys = {(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in baseline_resume.appended[1]}
        resumed = RecordingResume(
            plans=dict(baseline_resume.planned[:2]),
            outcomes=dict(baseline_resume.completed[:2]),
            rows_by_stage={0: stage0_rows},
            gated_keys={0: stage0_keys, 1: stage1_keys},
        )

        dispatched_stages: List[int] = []

        async def capture_stage(rows_in: List[Dict[str, Any]]):
            dispatched_stages.extend(row["stage_index"] for row in rows_in)
            return await successful_run(rows_in)

        _, resumed_summaries = await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            _materialized_rows(task_ids),
            capture_stage,
            resume=resumed,
        )

        assert dispatched_stages == []
        assert len(resumed_summaries) == 2
        assert resumed_summaries[0]["cached"] is True
        assert resumed_summaries[1]["eval_elo"] is None

    async def test_plan_replay_is_deterministic_without_seed(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3", "5:2"]}).stages,
            seed=None,
        )
        # A recorded plan pins task_ids/reference_ids regardless of the seedless RNG.
        pinned_tasks = ["t9", "t8", "t7"]
        resume = RecordingResume(plans={0: {"stage_index": 0, "reference_ids": ["a", "b"], "task_ids": pinned_tasks}})
        seen_tasks: List[str] = []
        base = _fake_run_rollouts_factory()

        async def recording_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                if r["stage_index"] == 0:
                    seen_tasks.append(r["task_id"])
            return await base(rows_in)

        _, summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, recording_run, resume=resume
        )
        assert set(seen_tasks) == set(pinned_tasks)
        assert summaries[0]["reference_ids"] == ["a", "b"]
        # No new plan was recorded for the replayed stage.
        assert all(i != 0 for i, _ in resume.planned)

    async def test_failure_rows_are_redispatched(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = _two_stage_cfg()
        full_run = _fake_run_rollouts_factory()

        all_results, base_summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, full_run
        )
        stage0_rows = [r for r in all_results if r["stage_index"] == 0]
        stage0_task_ids = list(dict.fromkeys(r["task_id"] for r in stage0_rows))
        # Mark one cached row as a failure: load_persisted_rows drops it, so it is
        # not in rows_by_stage and must be re-dispatched.
        good = stage0_rows[:-1]
        failed = dict(stage0_rows[-1])
        failed[NG_FAILURE_CLASS_KEY] = "some_error"
        failed_key = (failed[TASK_INDEX_KEY_NAME], failed[ROLLOUT_INDEX_KEY_NAME])

        # Simulate what build_file_resume does: successes only.
        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "reference_ids": base_summaries[0]["reference_ids"],
                    "task_ids": stage0_task_ids,
                }
            },
            rows_by_stage={0: good},
        )
        dispatched_keys: List[Tuple[int, int]] = []

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                if r["stage_index"] == 0:
                    dispatched_keys.append((r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
            return await full_run(rows_in)

        await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, capturing_run, resume=resume)
        assert dispatched_keys == [failed_key]

    async def test_resume_longest_first_uses_stage_aware_failure_timings(self) -> None:
        task_ids = [f"t{i}" for i in range(4)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["4"]}).stages,
            seed=0,
        )
        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "reference_ids": ["a"],
                    "task_ids": task_ids,
                    "task_reference_ids": {task_id: "a" for task_id in task_ids},
                }
            },
            elapsed_by_stage={0: {(0, 0): 60.0, (2, 0): 9000.0}},
        )
        dispatched: List[int] = []
        run = _fake_run_rollouts_factory()

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            dispatched.extend(row[TASK_INDEX_KEY_NAME] for row in rows_in)
            return await run(rows_in)

        await run_multistage_stages(
            cfg,
            REF_ELOS,
            _distribution(task_ids),
            rows,
            capturing_run,
            resume=resume,
            dispatch_longest_first=True,
        )

        assert dispatched == [2, 0, 1, 3]

    async def test_resume_reuses_failed_judge_deliverable_for_matching_stage_only(self) -> None:
        task_ids = ["t0"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["1", "1"]}).stages,
            seed=0,
            # Isolate sidecar propagation from normal cross-stage reuse.
            reuse_cached_deliverables=False,
        )
        resume = RecordingResume(reuse_cached_keys={1: {(0, 0)}})
        seen: List[Tuple[int, bool]] = []
        run = _fake_run_rollouts_factory()

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            seen.extend((row["stage_index"], bool(row.get("reuse_cached_deliverable"))) for row in rows_in)
            return await run(rows_in)

        await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, capturing_run, resume=resume)

        assert seen == [(0, False), (1, True)]

    async def test_gated_sidecar_artifact_is_produced_for_later_stages(self) -> None:
        task_ids = ["t0", "t1"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {
                            "num_tasks": 2,
                            "partial_completion": {
                                "min_success_fraction": 0.5,
                                "min_per_reference_success_fraction": 0.5,
                            },
                        },
                        {"num_tasks": 2},
                    ],
                }
            ).stages,
            seed=0,
        )
        cached_success = {
            **rows[1],
            "stage_index": 0,
            "reference_ids": ["a"],
            "per_reference": {"a": {"wins": 1, "losses": 0, "ties": 0, "reference_elo": REF_ELOS["a"]}},
        }
        # Stage 0's judge-invalid row exhausted its attempts, but its sidecar
        # flag proves that the reference-independent policy artifact exists.
        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "status": "planned",
                    "reference_ids": ["a"],
                    "task_ids": task_ids,
                    "task_reference_ids": {task_id: "a" for task_id in task_ids},
                }
            },
            rows_by_stage={0: [cached_success]},
            gated_keys={0: {(0, 0), (1, 0)}},
            reuse_cached_keys={0: {(0, 0)}},
            attempts_by_stage={0: {(0, 0): 3}},
        )
        seen: List[Tuple[int, int, bool]] = []
        run = _fake_run_rollouts_factory()

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            seen.extend(
                (row["stage_index"], row[TASK_INDEX_KEY_NAME], bool(row.get("reuse_cached_deliverable")))
                for row in rows_in
            )
            return await run(rows_in)

        await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, capturing_run, resume=resume)

        assert sorted(seen) == [(1, 0, True), (1, 1, True)]

    async def test_new_terminal_judge_failure_reuses_artifact_in_next_stage(self) -> None:
        task_ids = ["t0", "t1"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {
                            "num_tasks": 2,
                            "partial_completion": {
                                "min_success_fraction": 0.5,
                                "min_per_reference_success_fraction": 0.5,
                            },
                        },
                        {"num_tasks": 2},
                    ],
                }
            ).stages,
            seed=0,
        )
        seen: List[Tuple[int, int, bool]] = []
        success_run = _fake_run_rollouts_factory()

        async def terminal_then_success(rows_in: List[Dict[str, Any]]):
            seen.extend(
                (row["stage_index"], row[TASK_INDEX_KEY_NAME], bool(row.get("reuse_cached_deliverable")))
                for row in rows_in
            )
            if rows_in and rows_in[0]["stage_index"] == 0:
                pairs = await success_run(rows_in)
                row, _ = pairs[0]
                pairs[0] = (
                    row,
                    {
                        NG_FAILURE_CLASS_KEY: "permanent",
                        NG_TERMINAL_KEY: True,
                        "reuse_cached_deliverable": True,
                        "deliverables_dir": "/cached/task_t0/repeat_0",
                    },
                )
                return pairs
            return await success_run(rows_in)

        await run_multistage_stages(
            cfg,
            {"a": REF_ELOS["a"]},
            _distribution(task_ids),
            rows,
            terminal_then_success,
        )

        assert sorted(seen) == [
            (0, 0, False),
            (0, 1, False),
            (1, 0, True),
            (1, 1, True),
        ]

    async def test_third_failure_stops_adaptive_stage_without_fourth_invocation(self) -> None:
        task_ids = ["t0"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["1", "1"]}).stages,
            seed=0,
            reuse_cached_deliverables=False,
        )
        resume = RecordingResume(attempts_by_stage={0: {(0, 0): 2}})
        seen_attempts: List[Tuple[int, Any]] = []
        run = _fake_run_rollouts_factory()

        async def fail_third_then_succeed(rows_in: List[Dict[str, Any]]):
            stage_index = rows_in[0]["stage_index"]
            seen_attempts.append((stage_index, rows_in[0].get(ATTEMPT_INDEX_KEY_NAME)))
            if stage_index == 0:
                return [(rows_in[0], {NG_FAILURE_CLASS_KEY: "judge_invalid"})]
            return await run(rows_in)

        await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, fail_third_then_succeed, resume=resume
        )

        assert seen_attempts == [(0, 2)]
        assert resume.completed == []
        assert resume.appended[0][0][ATTEMPT_INDEX_KEY_NAME] == 2

    async def test_drained_stage_is_not_marked_complete(self) -> None:
        task_ids = ["t0"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["1", "1"]}).stages,
            seed=0,
        )
        resume = RecordingResume()
        dispatches = 0

        async def drain(rows_in: List[Dict[str, Any]]):
            nonlocal dispatches
            dispatches += 1
            return [
                (
                    row,
                    {NG_FAILURE_CLASS_KEY: "kill_shaped", NG_NO_PERSIST_KEY: True},
                )
                for row in rows_in
            ]

        results, _ = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, drain, resume=resume)

        assert results == []
        assert resume.completed == []
        assert dispatches == 1

    async def test_partial_final_stage_declares_full_expected_row_count(self) -> None:
        task_ids = ["t0", "t1"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["2", "2"]}).stages,
            seed=0,
        )
        resume = RecordingResume()
        base_run = _fake_run_rollouts_factory()
        calls = 0

        async def partial_final(rows_in: List[Dict[str, Any]]):
            nonlocal calls
            calls += 1
            pairs = await base_run(rows_in)
            if calls == 2:
                row, _ = pairs[-1]
                pairs[-1] = (row, {NG_FAILURE_CLASS_KEY: "kill_shaped", NG_NO_PERSIST_KEY: True})
            return pairs

        results, _ = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, partial_final, resume=resume
        )

        final_rows = [row for row in results if row["stage_index"] == 1]
        assert len(final_rows) == 1
        assert final_rows[0]["expected_final_stage_index"] == 1
        assert final_rows[0]["expected_stage_row_count"] == 2
        assert [index for index, _ in resume.completed] == [0]


class TestPartialStageCompletion:
    @staticmethod
    def _config(
        *,
        enabled: bool,
        min_success_fraction: float = 0.6,
        min_per_reference_success_fraction: float = 0.6,
        min_successful_rows_per_reference: int = 1,
        waivable_failure_classes: Optional[Sequence[str]] = None,
    ) -> MultiStageRunConfig:
        first_stage: Dict[str, Any] = {"num_tasks": 10}
        if enabled:
            first_stage["partial_completion"] = {
                "min_success_fraction": min_success_fraction,
                "min_per_reference_success_fraction": min_per_reference_success_fraction,
                "min_successful_rows_per_reference": min_successful_rows_per_reference,
            }
            if waivable_failure_classes is not None:
                first_stage["partial_completion"]["waivable_failure_classes"] = list(waivable_failure_classes)
        return MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [first_stage, {"num_tasks": 10, "num_models": 1}],
                }
            ).stages,
            seed=0,
        )

    @staticmethod
    def _balanced_resume(task_ids: List[str]) -> RecordingResume:
        return RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "status": "planned",
                    "reference_ids": ["a", "b"],
                    "task_ids": task_ids,
                    "task_reference_ids": {
                        task_id: ("a" if index % 2 == 0 else "b") for index, task_id in enumerate(task_ids)
                    },
                    "seed": None,
                    "prior_eval_elo": None,
                }
            }
        )

    @staticmethod
    def _runner(
        failed_indices: set[int],
        *,
        failure_class: str = "timeout_exceeded",
        no_persist: bool = False,
        terminal: bool = False,
        empty_successes: bool = False,
        dispatched_stages: Optional[List[int]] = None,
    ):
        successful_run = _fake_run_rollouts_factory(target_elo=1100.0)

        async def run(rows_in: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
            if dispatched_stages is not None:
                dispatched_stages.extend(row["stage_index"] for row in rows_in)
            pairs = await successful_run(rows_in)
            if not rows_in or rows_in[0]["stage_index"] != 0:
                return pairs

            rewritten: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for row, result in pairs:
                if row[TASK_INDEX_KEY_NAME] in failed_indices:
                    failure = {NG_FAILURE_CLASS_KEY: failure_class}
                    if no_persist:
                        failure[NG_NO_PERSIST_KEY] = True
                    if terminal:
                        failure[NG_TERMINAL_KEY] = True
                    rewritten.append((row, failure))
                elif empty_successes:
                    rewritten.append((row, {"task_id": row["task_id"], "per_reference": {}}))
                else:
                    rewritten.append((row, result))
            return rewritten

        return run

    async def test_four_retryable_timeouts_keep_default_retry_stage_open(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        dispatched_stages: List[int] = []

        results, summaries = await run_multistage_stages(
            self._config(enabled=False),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}, dispatched_stages=dispatched_stages),
            resume=resume,
        )

        assert set(dispatched_stages) == {0}
        assert len([row for row in results if row["stage_index"] == 0]) == 6
        assert len(summaries) == 1
        assert resume.completed == []

    async def test_existing_41_of_45_timeout_stage_advances_without_redispatch(self, tmp_path: Path) -> None:
        task_ids = [f"t{i}" for i in range(45)]
        rows = _materialized_rows(task_ids)
        reference_elos = {f"ref_{index}": 600.0 + 100.0 * index for index in range(9)}
        reference_ids = list(reference_elos)
        strict_cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [{"num_tasks": 45}, {"num_tasks": 45, "num_models": 4}],
                "seed": 0,
            }
        )
        partial_cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [
                    {
                        "num_tasks": 45,
                        "partial_completion": {
                            "min_success_fraction": 0.9,
                            "min_per_reference_success_fraction": 0.8,
                            "min_successful_rows_per_reference": 1,
                        },
                    },
                    {"num_tasks": 45, "num_models": 4},
                ],
                "seed": 0,
            }
        )
        distribution = _distribution(task_ids)
        strict_fingerprint = compute_fingerprint(strict_cfg, reference_elos, distribution)
        partial_fingerprint = compute_fingerprint(partial_cfg, reference_elos, distribution)
        assert partial_fingerprint == strict_fingerprint

        output = tmp_path / "rollouts.jsonl"
        output.write_bytes(b"")
        journal = journal_path_for(output)
        task_reference_ids = {task_id: reference_ids[index // 5] for index, task_id in enumerate(task_ids)}
        append_journal_record(
            journal,
            {
                "stage_index": 0,
                "status": "planned",
                "reference_ids": reference_ids,
                "task_ids": task_ids,
                "task_reference_ids": task_reference_ids,
                "seed": None,
                "prior_eval_elo": None,
            },
            strict_fingerprint,
        )

        async def successful(rows_in: List[Dict[str, Any]]):
            pairs = []
            for row in rows_in:
                reference_id = row["reference_ids"][0]
                reference_elo = reference_elos[reference_id]
                wins = 6 if reference_elo < 1000 else 4
                pairs.append(
                    (
                        row,
                        {
                            "task_id": row["task_id"],
                            "per_reference": {
                                reference_id: {
                                    "wins": wins,
                                    "losses": 10 - wins,
                                    "ties": 0,
                                    "reference_elo": reference_elo,
                                }
                            },
                        },
                    )
                )
            return pairs

        timeout_indices = {0, 5, 10, 15}

        async def first_allocation(rows_in: List[Dict[str, Any]]):
            assert {row["stage_index"] for row in rows_in} == {0}
            pairs = await successful(rows_in)
            return [
                (row, {NG_FAILURE_CLASS_KEY: "timeout_exceeded"})
                if row[TASK_INDEX_KEY_NAME] in timeout_indices
                else (row, result)
                for row, result in pairs
            ]

        first_results, first_summaries = await run_multistage_stages(
            strict_cfg,
            reference_elos,
            distribution,
            rows,
            first_allocation,
            resume=build_file_resume(output, journal, strict_fingerprint),
        )
        assert len(first_results) == 41
        assert len(first_summaries) == 1
        assert read_journal(journal)[1] == {}
        assert set(load_latest_failures(output)[0]) == {(index, 0) for index in timeout_indices}

        dispatched_stages: List[int] = []

        async def resumed_allocation(rows_in: List[Dict[str, Any]]):
            dispatched_stages.extend(row["stage_index"] for row in rows_in)
            if any(row["stage_index"] == 0 for row in rows_in):
                pytest.fail("persisted Stage-0 timeouts were unexpectedly redispatched")
            pairs = await successful(rows_in)
            row, _ = pairs[-1]
            pairs[-1] = (row, {NG_FAILURE_CLASS_KEY: "kill_shaped", NG_NO_PERSIST_KEY: True})
            return pairs

        resumed_results, resumed_summaries = await run_multistage_stages(
            partial_cfg,
            reference_elos,
            distribution,
            rows,
            resumed_allocation,
            resume=build_file_resume(output, journal, partial_fingerprint),
        )

        assert set(dispatched_stages) == {1}
        assert len([row for row in resumed_results if row["stage_index"] == 0]) == 41
        assert len([row for row in resumed_results if row["stage_index"] == 1]) == 44
        assert resumed_summaries[0]["partial"] is True
        assert resumed_summaries[0]["success_fraction"] == pytest.approx(41 / 45)
        plans, outcomes, fingerprint = read_journal(journal)
        assert fingerprint == partial_fingerprint
        assert 1 in plans
        assert outcomes[0]["status"] == "partial_complete"
        assert len(outcomes[0]["omitted_keys"]) == 4
        assert 1 not in outcomes
        assert all(record["success_fraction"] >= 0.8 for record in outcomes[0]["per_reference"].values())

        final_dispatches: List[Tuple[int, int]] = []

        async def finish_final_stage(rows_in: List[Dict[str, Any]]):
            final_dispatches.extend((row["stage_index"], row[TASK_INDEX_KEY_NAME]) for row in rows_in)
            return await successful(rows_in)

        final_results, final_summaries = await run_multistage_stages(
            partial_cfg,
            reference_elos,
            distribution,
            rows,
            finish_final_stage,
            resume=build_file_resume(output, journal, partial_fingerprint),
        )

        assert len(final_dispatches) == 1
        assert final_dispatches[0][0] == 1
        assert len([row for row in final_results if row["stage_index"] == 0]) == 41
        assert len([row for row in final_results if row["stage_index"] == 1]) == 45
        assert len(final_summaries) == 2
        assert read_journal(journal)[1][1]["status"] == "complete"

    async def test_no_persist_retry_overrides_older_timeout_on_resume(self, tmp_path: Path) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [
                    {
                        "num_tasks": 10,
                        "partial_completion": {
                            "min_success_fraction": 0.8,
                            "min_per_reference_success_fraction": 0.8,
                        },
                    },
                    {"num_tasks": 10, "num_models": 1},
                ],
                "seed": 0,
            }
        )
        reference_elos = {"a": 1000.0}
        distribution = _distribution(task_ids)
        output = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(output)
        fingerprint = compute_fingerprint(cfg, reference_elos, distribution)
        successful_run = _fake_run_rollouts_factory(target_elo=1100.0)

        async def first_attempt(rows_in: List[Dict[str, Any]]):
            pairs = await successful_run(rows_in)
            return [
                (row, {NG_FAILURE_CLASS_KEY: "timeout_exceeded"})
                if row["stage_index"] == 0 and row[TASK_INDEX_KEY_NAME] >= 6
                else (row, result)
                for row, result in pairs
            ]

        _, first_summaries = await run_multistage_stages(
            cfg,
            reference_elos,
            distribution,
            rows,
            first_attempt,
            resume=build_file_resume(output, journal, fingerprint),
        )
        assert len(first_summaries) == 1

        async def drained_retry(rows_in: List[Dict[str, Any]]):
            pairs = await successful_run(rows_in)
            return [
                (
                    row,
                    {NG_FAILURE_CLASS_KEY: "kill_shaped", NG_NO_PERSIST_KEY: True},
                )
                if row[TASK_INDEX_KEY_NAME] >= 8
                else (row, result)
                for row, result in pairs
            ]

        _, second_summaries = await run_multistage_stages(
            cfg,
            reference_elos,
            distribution,
            rows,
            drained_retry,
            resume=build_file_resume(output, journal, fingerprint),
        )
        assert len(second_summaries) == 1
        dispositions = load_latest_attempt_dispositions(journal)[0]
        assert dispositions[(8, 0)][NG_NO_PERSIST_KEY] is True
        assert dispositions[(9, 0)][NG_NO_PERSIST_KEY] is True

        dispatched: List[Tuple[int, int]] = []

        async def final_retry(rows_in: List[Dict[str, Any]]):
            dispatched.extend((row["stage_index"], row[TASK_INDEX_KEY_NAME]) for row in rows_in)
            return await successful_run(rows_in)

        _, final_summaries = await run_multistage_stages(
            cfg,
            reference_elos,
            distribution,
            rows,
            final_retry,
            resume=build_file_resume(output, journal, fingerprint),
        )

        assert sorted(index for stage, index in dispatched if stage == 0) == [8, 9]
        assert {stage for stage, _ in dispatched} == {0, 1}
        assert len(final_summaries) == 2

    async def test_newer_sidecar_failure_overrides_older_timeout_disposition(self, tmp_path: Path) -> None:
        task_ids = [f"t{i}" for i in range(5)]
        rows = _materialized_rows(task_ids)
        strict_cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [{"num_tasks": 5}, {"num_tasks": 5, "num_models": 1}],
                "seed": 0,
            }
        )
        partial_cfg = parse_multistage_config(
            {
                "enabled": True,
                "stages": [
                    {
                        "num_tasks": 5,
                        "partial_completion": {
                            "min_success_fraction": 0.8,
                            "min_per_reference_success_fraction": 0.8,
                        },
                    },
                    {"num_tasks": 5, "num_models": 1},
                ],
                "seed": 0,
            }
        )
        reference_elos = {"a": 1000.0}
        distribution = _distribution(task_ids)
        output = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(output)
        fingerprint = compute_fingerprint(strict_cfg, reference_elos, distribution)
        assert compute_fingerprint(partial_cfg, reference_elos, distribution) == fingerprint
        successful_run = _fake_run_rollouts_factory(target_elo=1100.0)

        async def initial_timeout(rows_in: List[Dict[str, Any]]):
            pairs = await successful_run(rows_in)
            return [
                (row, {NG_FAILURE_CLASS_KEY: "timeout_exceeded"}) if row[TASK_INDEX_KEY_NAME] == 4 else (row, result)
                for row, result in pairs
            ]

        _, first_summaries = await run_multistage_stages(
            strict_cfg,
            reference_elos,
            distribution,
            rows,
            initial_timeout,
            resume=build_file_resume(output, journal, fingerprint),
        )
        assert len(first_summaries) == 1
        timeout = load_latest_failures(output)[0][(4, 0)]
        assert load_latest_attempt_dispositions(journal)[0][(4, 0)][NG_FAILURE_CLASS_KEY] == "timeout_exceeded"

        # Simulate a crash after the newer sidecar row is fsynced but before its
        # disposition is appended to the journal.
        route_stage_rows(output, [{**timeout, NG_FAILURE_CLASS_KEY: "transient"}])
        assert load_latest_failures(output)[0][(4, 0)][NG_FAILURE_CLASS_KEY] == "transient"
        assert load_latest_attempt_dispositions(journal)[0][(4, 0)][NG_FAILURE_CLASS_KEY] == "timeout_exceeded"

        dispatched: List[Tuple[int, int]] = []

        async def capture(rows_in: List[Dict[str, Any]]):
            dispatched.extend((row["stage_index"], row[TASK_INDEX_KEY_NAME]) for row in rows_in)
            return await successful_run(rows_in)

        _, summaries = await run_multistage_stages(
            partial_cfg,
            reference_elos,
            distribution,
            rows,
            capture,
            resume=build_file_resume(output, journal, fingerprint),
        )

        assert [task_index for stage, task_index in dispatched if stage == 0] == [4]
        assert {stage for stage, _ in dispatched} == {0, 1}
        assert len(summaries) == 2

    async def test_explicit_timeout_only_policy_advances_with_balanced_coverage(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        dispatched_stages: List[int] = []

        results, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}, dispatched_stages=dispatched_stages),
            resume=resume,
        )

        assert set(dispatched_stages) == {0, 1}
        assert len(summaries) == 2
        assert summaries[0]["partial"] is True
        assert summaries[0]["success_fraction"] == pytest.approx(0.6)
        assert summaries[0]["num_omitted"] == 4
        assert len([row for row in results if row["stage_index"] == 1]) == 10
        assert [index for index, _ in resume.completed] == [0, 1]
        outcome = resume.completed[0][1]
        assert outcome["status"] == "partial_complete"
        assert len(outcome["included_keys"]) == 6
        assert len(outcome["omitted_keys"]) == 4
        assert outcome["per_reference"]["a"]["success_fraction"] == pytest.approx(0.6)
        assert outcome["per_reference"]["b"]["success_fraction"] == pytest.approx(0.6)

    @pytest.mark.parametrize(
        ("failure_class", "no_persist"),
        [("transient", False), ("timeout_exceeded", True)],
    )
    async def test_policy_rejects_non_timeout_or_unpersisted_omissions(
        self, failure_class: str, no_persist: bool
    ) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        dispatched_stages: List[int] = []

        _, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner(
                {0, 1, 2, 3},
                failure_class=failure_class,
                no_persist=no_persist,
                dispatched_stages=dispatched_stages,
            ),
            resume=resume,
        )

        assert set(dispatched_stages) == {0}
        assert len(summaries) == 1
        assert resume.completed == []

    async def test_transient_omission_advances_when_explicitly_waivable(self) -> None:
        """A judge-failure `transient` row is waivable only when the policy says so.

        This is the `dc6c776f3af506df` case: one unresolved `transient` rollout
        (an empty-PDF 400 from the judge panel) held the whole run at stage 1
        under the timeout-only default, costing 176 of 220 tasks.
        """
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        dispatched_stages: List[int] = []

        _, summaries = await run_multistage_stages(
            self._config(enabled=True, waivable_failure_classes=["timeout_exceeded", "transient"]),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner(
                {0, 1, 2, 3},
                failure_class="transient",
                dispatched_stages=dispatched_stages,
            ),
            resume=resume,
        )

        assert set(dispatched_stages) == {0, 1}
        assert len(summaries) == 2
        assert summaries[0]["partial"] is True
        assert summaries[0]["num_omitted"] == 4
        outcome = resume.completed[0][1]
        assert outcome["status"] == "partial_complete"
        assert outcome["policy"]["newly_waivable_failure_classes"] == ["timeout_exceeded", "transient"]

    async def test_unknown_waivable_failure_class_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="waivable_failure_classes"):
            parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {"num_tasks": 10, "partial_completion": {"waivable_failure_classes": ["skipped"]}},
                        {"num_tasks": 10},
                    ],
                }
            )

    async def test_empty_waivable_failure_classes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="waivable_failure_classes"):
            parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {"num_tasks": 10, "partial_completion": {"waivable_failure_classes": []}},
                        {"num_tasks": 10},
                    ],
                }
            )

    async def test_policy_rejects_missing_elo_fit(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)

        _, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}, empty_successes=True),
            resume=resume,
        )

        assert summaries[0]["eval_elo"] is None
        assert len(summaries) == 1
        assert resume.completed == []

    async def test_policy_rejects_one_success_row_without_battle_evidence(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        successful_run = _fake_run_rollouts_factory(target_elo=1100.0)

        async def one_empty_battle(rows_in: List[Dict[str, Any]]):
            pairs = await successful_run(rows_in)
            if rows_in and rows_in[0]["stage_index"] == 0:
                row, result = pairs[0]
                pairs[0] = (row, {**result, "per_reference": {}})
            return pairs

        _, summaries = await run_multistage_stages(
            self._config(enabled=True, min_success_fraction=0.8),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            one_empty_battle,
            resume=resume,
        )

        assert len(summaries) == 1
        assert resume.completed == []

    async def test_policy_requires_evidence_for_every_selected_reference(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = RecordingResume(
            plans={
                0: {
                    "stage_index": 0,
                    "status": "planned",
                    "reference_ids": ["a", "b"],
                    "task_ids": task_ids,
                    "task_reference_ids": {task_id: "a" for task_id in task_ids},
                    "seed": None,
                    "prior_eval_elo": None,
                }
            }
        )

        _, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            _fake_run_rollouts_factory(target_elo=1100.0),
            resume=resume,
        )

        assert len(summaries) == 1
        assert resume.completed == []

    async def test_policy_rejects_inadequate_per_reference_coverage(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)

        _, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 2, 4, 6}),
            resume=resume,
        )

        assert summaries[0]["eval_elo"] is not None
        assert len(summaries) == 1
        assert resume.completed == []

    @pytest.mark.parametrize(
        ("min_success_fraction", "min_per_reference_success_fraction", "min_successful_rows_per_reference"),
        [
            (0.8, 0.5, 1),
            (0.6, 0.5, 4),
        ],
        ids=["overall-fraction", "minimum-rows-per-reference"],
    )
    async def test_policy_enforces_independent_coverage_floors(
        self,
        min_success_fraction: float,
        min_per_reference_success_fraction: float,
        min_successful_rows_per_reference: int,
    ) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        cfg = self._config(
            enabled=True,
            min_success_fraction=min_success_fraction,
            min_per_reference_success_fraction=min_per_reference_success_fraction,
            min_successful_rows_per_reference=min_successful_rows_per_reference,
        )

        _, summaries = await run_multistage_stages(
            cfg,
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}),
            resume=resume,
        )

        assert len(summaries) == 1
        assert resume.completed == []

    async def test_policy_separates_existing_terminal_and_new_timeout_omissions(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        successful_run = _fake_run_rollouts_factory(target_elo=1100.0)

        async def mixed_failures(rows_in: List[Dict[str, Any]]):
            pairs = await successful_run(rows_in)
            if not rows_in or rows_in[0]["stage_index"] != 0:
                return pairs
            rewritten = []
            for row, result in pairs:
                task_index = row[TASK_INDEX_KEY_NAME]
                if task_index == 0:
                    rewritten.append((row, {NG_FAILURE_CLASS_KEY: "skipped", NG_TERMINAL_KEY: True}))
                elif task_index in {1, 2, 3}:
                    rewritten.append((row, {NG_FAILURE_CLASS_KEY: "timeout_exceeded"}))
                else:
                    rewritten.append((row, result))
            return rewritten

        _, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            mixed_failures,
            resume=resume,
        )

        assert len(summaries) == 2
        outcome = resume.completed[0][1]
        assert outcome["status"] == "partial_complete"
        assert outcome["accepted_unresolved_keys"] == [[1, 0], [2, 0], [3, 0]]
        assert outcome["already_resolved_omitted_keys"] == [[0, 0]]

    async def test_enabled_policy_is_a_coverage_floor_for_terminal_omissions(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)

        _, summaries = await run_multistage_stages(
            self._config(
                enabled=True,
                min_success_fraction=0.9,
                min_per_reference_success_fraction=0.5,
            ),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1}, failure_class="skipped", terminal=True),
            resume=resume,
        )

        assert len(summaries) == 1
        assert resume.completed == []

    async def test_policy_rejects_undispatched_row(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        resume = self._balanced_resume(task_ids)
        successful_run = _fake_run_rollouts_factory(target_elo=1100.0)

        async def omit_one_result(rows_in: List[Dict[str, Any]]):
            pairs = await successful_run(rows_in)
            if rows_in and rows_in[0]["stage_index"] == 0:
                return pairs[:-1]
            return pairs

        _, summaries = await run_multistage_stages(
            self._config(enabled=True),
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            omit_one_result,
            resume=resume,
        )

        assert len(summaries) == 1
        assert resume.completed == []

    async def test_direct_final_stage_policy_is_rejected(self) -> None:
        task_ids = ["t0"]
        cfg = self._config(enabled=True)
        cfg.stages[-1].partial_completion = cfg.stages[0].partial_completion

        with pytest.raises(ValueError, match="non-final calibration stages"):
            await run_multistage_stages(
                cfg,
                {"a": 1000.0},
                _distribution(task_ids),
                _materialized_rows(task_ids),
                _fake_run_rollouts_factory(),
            )

    async def test_invalid_directly_constructed_policy_is_rejected(self) -> None:
        task_ids = ["t0"]
        cfg = self._config(enabled=True)
        assert cfg.stages[0].partial_completion is not None
        cfg.stages[0].partial_completion.min_success_fraction = 1.1

        with pytest.raises(ValueError, match="min_success_fraction"):
            await run_multistage_stages(
                cfg,
                {"a": 1000.0},
                _distribution(task_ids),
                _materialized_rows(task_ids),
                _fake_run_rollouts_factory(),
            )

    async def test_cached_partial_snapshot_with_invalid_evidence_is_not_reused(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        cfg = self._config(enabled=True)
        initial_resume = self._balanced_resume(task_ids)
        results, _ = await run_multistage_stages(
            cfg,
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}),
            resume=initial_resume,
        )
        stage0_rows = [row for row in results if row["stage_index"] == 0]
        corrupted_rows = [dict(row, per_reference={}) for row in stage0_rows]
        outcome = initial_resume.completed[0][1]
        all_stage0_keys = {(index, 0) for index in range(10)}
        resumed = RecordingResume(
            plans={0: initial_resume.plans[0]},
            outcomes={0: outcome},
            rows_by_stage={0: corrupted_rows},
            gated_keys={0: all_stage0_keys},
        )
        dispatched_stages: List[int] = []

        with pytest.raises(RuntimeError, match="cached partial stage 0 snapshot is invalid"):
            await run_multistage_stages(
                cfg,
                {"a": 1000.0, "b": 1200.0},
                _distribution(task_ids),
                _materialized_rows(task_ids),
                self._runner(set(), dispatched_stages=dispatched_stages),
                resume=resumed,
            )

        assert dispatched_stages == []
        assert resumed.completed == []

    async def test_cached_partial_snapshot_rejects_changed_valid_evidence(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        cfg = self._config(enabled=True)
        initial_resume = self._balanced_resume(task_ids)
        results, _ = await run_multistage_stages(
            cfg,
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}),
            resume=initial_resume,
        )
        stage0_rows = [deepcopy(row) for row in results if row["stage_index"] == 0]
        reference_id = next(iter(stage0_rows[0]["per_reference"]))
        counts = stage0_rows[0]["per_reference"][reference_id]
        counts["wins"] += 1
        resumed = RecordingResume(
            plans={0: initial_resume.plans[0]},
            outcomes={0: initial_resume.completed[0][1]},
            rows_by_stage={0: stage0_rows},
            gated_keys={0: {(index, 0) for index in range(10)}},
        )
        dispatched_stages: List[int] = []

        with pytest.raises(RuntimeError, match="cached partial stage 0 snapshot is invalid"):
            await run_multistage_stages(
                cfg,
                {"a": 1000.0, "b": 1200.0},
                _distribution(task_ids),
                _materialized_rows(task_ids),
                self._runner(set(), dispatched_stages=dispatched_stages),
                resume=resumed,
            )

        assert dispatched_stages == []

    async def test_changed_policy_does_not_reuse_frozen_partial_outcome(self) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        initial_cfg = self._config(enabled=True)
        initial_resume = self._balanced_resume(task_ids)
        results, _ = await run_multistage_stages(
            initial_cfg,
            {"a": 1000.0, "b": 1200.0},
            _distribution(task_ids),
            _materialized_rows(task_ids),
            self._runner({0, 1, 2, 3}),
            resume=initial_resume,
        )
        outcome = initial_resume.completed[0][1]
        stage0_rows = [row for row in results if row["stage_index"] == 0]
        resumed = RecordingResume(
            plans={0: initial_resume.plans[0]},
            outcomes={0: outcome},
            rows_by_stage={0: stage0_rows},
            gated_keys={0: {(index, 0) for index in range(10)}},
        )

        with pytest.raises(RuntimeError, match="cached partial stage 0 snapshot is invalid"):
            await run_multistage_stages(
                self._config(enabled=True, min_success_fraction=0.5),
                {"a": 1000.0, "b": 1200.0},
                _distribution(task_ids),
                _materialized_rows(task_ids),
                self._runner(set()),
                resume=resumed,
            )

    async def test_file_resume_freezes_partial_rows_and_ignores_late_success(self, tmp_path: Path) -> None:
        task_ids = [f"t{i}" for i in range(5)]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config(
                {
                    "enabled": True,
                    "stages": [
                        {
                            "num_tasks": 5,
                            "partial_completion": {
                                "min_success_fraction": 0.8,
                                "min_per_reference_success_fraction": 0.8,
                            },
                        },
                        {"num_tasks": 5, "num_models": 1},
                    ],
                }
            ).stages,
            seed=0,
        )
        reference_elos = {"a": 1000.0}
        distribution = _distribution(task_ids)
        output = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(output)
        fingerprint = compute_fingerprint(cfg, reference_elos, distribution)

        first_resume = build_file_resume(output, journal, fingerprint)
        _, first_summaries = await run_multistage_stages(
            cfg,
            reference_elos,
            distribution,
            rows,
            self._runner({4}),
            resume=first_resume,
        )
        plans, outcomes, _ = read_journal(journal)
        assert outcomes[0]["status"] == "partial_complete"
        assert first_summaries[0]["num_omitted"] == 1
        original_stage0_elo = first_summaries[0]["eval_elo"]
        original_stage1_plan = deepcopy(plans[1])

        omitted_key = tuple(outcomes[0]["omitted_keys"][0])
        late_base = next(row for row in rows if (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) == omitted_key)
        route_stage_rows(
            output,
            [
                {
                    TASK_INDEX_KEY_NAME: omitted_key[0],
                    ROLLOUT_INDEX_KEY_NAME: omitted_key[1],
                    AGENT_REF_KEY_NAME: late_base[AGENT_REF_KEY_NAME],
                    "stage_index": 0,
                    "task_id": late_base["task_id"],
                    "per_reference": {"a": {"wins": 1, "losses": 0, "ties": 0, "reference_elo": 1000.0}},
                }
            ],
        )

        resumed = build_file_resume(output, journal, fingerprint)

        async def no_dispatch(rows_in: List[Dict[str, Any]]):
            pytest.fail(f"completed partial run unexpectedly dispatched {len(rows_in)} row(s)")

        resumed_results, resumed_summaries = await run_multistage_stages(
            cfg,
            reference_elos,
            distribution,
            rows,
            no_dispatch,
            resume=resumed,
        )

        resumed_stage0 = [row for row in resumed_results if row["stage_index"] == 0]
        assert len(resumed_stage0) == 4
        assert omitted_key not in {(row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) for row in resumed_stage0}
        assert resumed_summaries[0]["partial"] is True
        assert resumed_summaries[0]["num_rollouts"] == 5
        assert resumed_summaries[0]["num_successful"] == 4
        assert resumed_summaries[0]["eval_elo"] == pytest.approx(original_stage0_elo)
        assert resumed.plans[1] == original_stage1_plan


class TestFingerprint:
    def test_stable_and_config_sensitive(self) -> None:
        dist = _distribution(["t0", "t1", "t2"])
        cfg = _two_stage_cfg()
        fp1 = compute_fingerprint(cfg, REF_ELOS, dist)
        fp2 = compute_fingerprint(cfg, REF_ELOS, dist)
        assert fp1 == fp2

        other_dist = _distribution(["t0", "t1", "t9"])
        assert compute_fingerprint(cfg, REF_ELOS, other_dist) != fp1

        other_elos = dict(REF_ELOS, a=999.0)
        assert compute_fingerprint(cfg, other_elos, dist) != fp1

        cfg2 = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["4", "5:2"]}).stages,
            seed=0,
        )
        assert compute_fingerprint(cfg2, REF_ELOS, dist) != fp1

    def test_percentage_change_invalidates(self) -> None:
        # Same task-id sets but different group weights ⇒ seeded sampling draws
        # different tasks, so the fingerprint must differ.
        cfg = _two_stage_cfg()
        dist_a = {
            "g0": {"percentage": 0.5, "task_ids": ["t0", "t1"]},
            "g1": {"percentage": 0.5, "task_ids": ["t2", "t3"]},
        }
        dist_b = {
            "g0": {"percentage": 0.9, "task_ids": ["t0", "t1"]},
            "g1": {"percentage": 0.1, "task_ids": ["t2", "t3"]},
        }
        assert compute_fingerprint(cfg, REF_ELOS, dist_a) != compute_fingerprint(cfg, REF_ELOS, dist_b)

    def test_partial_completion_policy_preserves_rollout_cache_fingerprint(self) -> None:
        dist = _distribution(["t0", "t1"])
        strict = parse_multistage_config(
            {"enabled": True, "stages": [{"num_tasks": 1}, {"num_tasks": 2, "num_models": 1}]}
        )
        partial = parse_multistage_config(
            {
                "enabled": True,
                "stages": [
                    {
                        "num_tasks": 1,
                        "partial_completion": {
                            "min_success_fraction": 0.9,
                            "min_per_reference_success_fraction": 0.5,
                        },
                    },
                    {"num_tasks": 2, "num_models": 1},
                ],
            }
        )
        changed_threshold = deepcopy(partial)
        assert changed_threshold.stages[0].partial_completion is not None
        changed_threshold.stages[0].partial_completion.min_success_fraction = 0.8

        strict_fingerprint = compute_fingerprint(strict, REF_ELOS, dist)
        partial_fingerprint = compute_fingerprint(partial, REF_ELOS, dist)
        assert partial_fingerprint == strict_fingerprint
        assert compute_fingerprint(changed_threshold, REF_ELOS, dist) == partial_fingerprint

    def test_materialized_rows_and_result_affecting_run_config_invalidate(self) -> None:
        cfg = _two_stage_cfg()
        dist = _distribution(["t0"])
        rows = _materialized_rows(["t0"])
        run_config = SimpleNamespace(
            agent_name="gdpval_stirrup_agent",
            input_jsonl_fpath="tasks.jsonl",
            limit=None,
            num_repeats=1,
            num_repeats_add_seed=False,
            responses_create_params={"temperature": 0.6},
            prompt_config="prompt.yaml",
            skills=None,
        )
        baseline = compute_fingerprint(
            cfg,
            REF_ELOS,
            dist,
            materialized_rows=rows,
            rollout_collection_config=run_config,
        )

        changed_prompt_row = [dict(rows[0], responses_create_params={"input": [{"role": "user", "content": "new"}]})]
        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=changed_prompt_row,
                rollout_collection_config=run_config,
            )
            != baseline
        )

        changed_repeats = SimpleNamespace(**vars(run_config))
        changed_repeats.num_repeats = 2
        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=rows,
                rollout_collection_config=changed_repeats,
            )
            != baseline
        )

        remapped = [dict(rows[0], **{TASK_INDEX_KEY_NAME: 99})]
        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=remapped,
                rollout_collection_config=run_config,
            )
            != baseline
        )

    def test_fixed_policy_and_judge_config_invalidate_fingerprint(self) -> None:
        cfg = _two_stage_cfg()
        dist = _distribution(["t0"])
        rows = _materialized_rows(["t0"])
        runtime = {
            "policy_model": {"responses_api_models": {"vllm": {"model": "/checkpoints/policy-a"}}},
            "gdpval_resources_server": {
                "resources_servers": {
                    "gdpval": {
                        "num_comparison_trials": 2,
                        "judge_responses_create_params_overrides": {"model": "judge-a"},
                    }
                }
            },
            "operational_logging": {"level": "INFO"},
        }
        baseline = compute_fingerprint(
            cfg,
            REF_ELOS,
            dist,
            materialized_rows=rows,
            resolved_global_config=runtime,
        )

        changed_policy = deepcopy(runtime)
        changed_policy["policy_model"]["responses_api_models"]["vllm"]["model"] = "/checkpoints/policy-b"
        changed_judge = deepcopy(runtime)
        changed_judge["gdpval_resources_server"]["resources_servers"]["gdpval"]["num_comparison_trials"] = 4
        changed_strict_trials = deepcopy(runtime)
        changed_strict_trials["gdpval_resources_server"]["resources_servers"]["gdpval"]["strict_comparison_trials"] = (
            True
        )
        changed_logging = deepcopy(runtime)
        changed_logging["operational_logging"]["level"] = "DEBUG"

        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=rows,
                resolved_global_config=changed_policy,
            )
            != baseline
        )
        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=rows,
                resolved_global_config=changed_strict_trials,
            )
            != baseline
        )
        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=rows,
                resolved_global_config=changed_judge,
            )
            != baseline
        )
        assert (
            compute_fingerprint(
                cfg,
                REF_ELOS,
                dist,
                materialized_rows=rows,
                resolved_global_config=changed_logging,
            )
            == baseline
        )

    def test_runtime_bind_addresses_do_not_invalidate_fingerprint(self) -> None:
        cfg = _two_stage_cfg()
        dist = _distribution(["t0"])
        runtime = _runtime_components_with_bind_addresses("node-a", 0)
        rebound_runtime = _runtime_components_with_bind_addresses("node-b", 100)

        baseline = compute_fingerprint(cfg, REF_ELOS, dist, resolved_global_config=runtime)
        assert compute_fingerprint(cfg, REF_ELOS, dist, resolved_global_config=rebound_runtime) == baseline

        changed_nested_host = deepcopy(rebound_runtime)
        changed_nested_host["agent"]["responses_api_agents"]["stirrup"]["worker_target"]["host"] = (
            "other-sandbox.internal"
        )
        assert compute_fingerprint(cfg, REF_ELOS, dist, resolved_global_config=changed_nested_host) != baseline

        changed_nested_port = deepcopy(rebound_runtime)
        changed_nested_port["agent"]["responses_api_agents"]["stirrup"]["worker_target"]["port"] = 9001
        assert compute_fingerprint(cfg, REF_ELOS, dist, resolved_global_config=changed_nested_port) != baseline

        changed_model = deepcopy(rebound_runtime)
        changed_model["policy"]["responses_api_models"]["vllm"]["model"] = "/checkpoints/policy-b"
        assert compute_fingerprint(cfg, REF_ELOS, dist, resolved_global_config=changed_model) != baseline


class TestJournalIO:
    def test_journal_round_trip_latest_wins(self, tmp_path: Path) -> None:
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        journal = journal_path_for(tmp_path / "rollouts.jsonl")
        assert journal.name == "rollouts_multistage_state.jsonl"
        # Plan carries references/tasks; completion is just a marker (eval_elo is
        # re-fit from rows on resume, so it is not stored).
        append_journal_record(journal, {"stage_index": 0, "status": "planned", "reference_ids": ["a"]}, "FP")
        append_journal_record(journal, {"stage_index": 0, "status": "complete"}, "FP")
        append_journal_record(journal, {"stage_index": 1, "status": "planned", "reference_ids": ["b", "c"]}, "FP")

        plans, outcomes, fingerprint = read_journal(journal)
        assert fingerprint == "FP"
        assert plans[0]["reference_ids"] == ["a"]
        assert plans[1]["reference_ids"] == ["b", "c"]
        assert outcomes[0] == {"stage_index": 0, "status": "complete", "fingerprint": "FP"}
        assert "eval_elo" not in outcomes[0]
        assert 1 not in outcomes

    def test_read_journal_missing_file(self, tmp_path: Path) -> None:
        plans, outcomes, fp = read_journal(tmp_path / "nope.jsonl")
        assert plans == {} and outcomes == {} and fp is None

    def test_load_persisted_rows_groups_by_stage(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        results = [
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"},
            # Defensive: a legacy failure row in the main jsonl is still dropped.
            {
                "stage_index": 0,
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "task_id": "t1",
                NG_FAILURE_CLASS_KEY: "boom",
            },
            {"stage_index": 1, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"},
        ]
        with out.open("wb") as handle:
            for r in results:
                handle.write(json.dumps(r).encode() + b"\n")
        by_stage = load_persisted_rows(out)
        assert len(by_stage[0]) == 1  # legacy failure dropped
        assert by_stage[0][0]["task_id"] == "t0"
        assert len(by_stage[1]) == 1

    def test_build_file_resume_persists_via_callbacks(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        # Seed one persisted success row + one journal plan.
        with out.open("wb") as handle:
            handle.write(
                json.dumps(
                    {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"}
                ).encode()
                + b"\n"
            )
        with journal.open("wb") as handle:
            handle.write(
                json.dumps(
                    {"stage_index": 0, "status": "planned", "reference_ids": ["a"], "fingerprint": "FP"}
                ).encode()
                + b"\n"
            )

        resume = build_file_resume(out, journal, "FP")
        assert 0 in resume.plans
        assert len(resume.rows_by_stage[0]) == 1

        resume.on_plan(1, {"stage_index": 1, "status": "planned", "reference_ids": ["b"]})
        resume.on_rows(1, [{"stage_index": 1, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"}])
        resume.on_outcome(1, {"stage_index": 1, "status": "complete"})

        plans, outcomes, fp = read_journal(journal)
        assert fp == "FP"
        assert 1 in plans and 1 in outcomes
        assert len(load_persisted_rows(out)[1]) == 1


class TestFailureRouting:
    def test_route_stage_rows_splits_by_outcome(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        rows = [
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"},
            {
                "stage_index": 0,
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "task_id": "t1",
                NG_FAILURE_CLASS_KEY: "boom",
            },
            {
                "stage_index": 0,
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "task_id": "t2",
                NG_NO_PERSIST_KEY: True,
            },
        ]
        route_stage_rows(out, rows)

        main = [json.loads(line) for line in out.read_text().splitlines()]
        sidecar = [json.loads(line) for line in failures_path_for(out).read_text().splitlines()]
        # Success -> main; failure -> sidecar (with stage_index); kill_shaped -> nowhere.
        assert [r["task_id"] for r in main] == ["t0"]
        assert [r["task_id"] for r in sidecar] == ["t1"]
        assert sidecar[0]["stage_index"] == 0

    def test_load_gated_keys_terminal_and_max_attempts(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        # One success in main jsonl (stage 0, task 0).
        with out.open("wb") as handle:
            handle.write(
                json.dumps(
                    {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "t0"}
                ).encode()
                + b"\n"
            )
        # Sidecar: task 1 terminal (never retried), task 2 hit 3 attempts (gated),
        # task 3 has 1 attempt (still re-dispatchable).
        sidecar = failures_path_for(out)
        entries = [
            {
                "stage_index": 0,
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                NG_FAILURE_CLASS_KEY: "x",
                NG_TERMINAL_KEY: True,
            },
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "x"},
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "x"},
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "x"},
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "x"},
        ]
        with sidecar.open("wb") as handle:
            for e in entries:
                handle.write(json.dumps(e).encode() + b"\n")

        rows_by_stage = load_persisted_rows(out)
        gated = load_gated_keys(out, rows_by_stage)
        # Success + terminal + max-attempts are gated; the single-attempt one is not.
        assert (0, 0) in gated[0]
        assert (1, 0) in gated[0]
        assert (2, 0) in gated[0]
        assert (3, 0) not in gated[0]

    def test_reopening_stage_prunes_all_downstream_persisted_state(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        main_rows = [
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 9, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "keep"},
            {"stage_index": 1, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "stale-1"},
            {"stage_index": 2, TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "stale-2"},
        ]
        out.write_text("".join(json.dumps(row) + "\n" for row in main_rows))
        failures = [
            {
                "stage_index": 0,
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                NG_FAILURE_CLASS_KEY: "judge_invalid",
                "elapsed_seconds": 10,
                "reuse_cached_deliverable": True,
            },
            {
                "stage_index": 1,
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                NG_FAILURE_CLASS_KEY: "judge_invalid",
                "elapsed_seconds": 20,
                "reuse_cached_deliverable": True,
            },
        ]
        failures_path_for(out).write_text("".join(json.dumps(row) + "\n" for row in failures))
        for stage_index in range(3):
            append_journal_record(
                journal,
                {
                    "stage_index": stage_index,
                    "status": "planned",
                    "reference_ids": ["a"],
                    "task_ids": [f"t{stage_index}"],
                },
                "FP",
            )
            append_journal_record(journal, {"stage_index": stage_index, "status": "complete"}, "FP")

        resume = build_file_resume(out, journal, "FP")

        assert set(resume.plans) == {0}
        assert resume.outcomes == {}
        assert set(resume.rows_by_stage) == {0}
        assert set(resume.gated_keys) <= {0}
        assert set(resume.elapsed_by_stage) == {0}
        assert set(resume.reuse_cached_keys) == {0}
        assert set(resume.attempts_by_stage) == {0}
        assert {json.loads(line)["stage_index"] for line in out.read_text().splitlines()} == {0}
        assert {json.loads(line)["stage_index"] for line in failures_path_for(out).read_text().splitlines()} == {0}
        journal_records = [json.loads(line) for line in journal.read_text().splitlines()]
        assert any(row.get("status") == "restart_from_stage" and row["stage_index"] == 0 for row in journal_records)
        assert journal_records[-1]["status"] == "restart_cleanup_complete"

    def test_historical_failure_does_not_reopen_stage_after_later_success(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        main_rows = [
            {"stage_index": 0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "resolved"},
            {"stage_index": 1, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "task_id": "downstream"},
        ]
        out.write_text("".join(json.dumps(row) + "\n" for row in main_rows))
        failures_path_for(out).write_text(
            json.dumps(
                {
                    "stage_index": 0,
                    TASK_INDEX_KEY_NAME: 0,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    NG_FAILURE_CLASS_KEY: "judge_invalid",
                }
            )
            + "\n"
        )
        for stage_index in range(2):
            append_journal_record(
                journal,
                {"stage_index": stage_index, "status": "planned", "reference_ids": ["a"], "task_ids": ["t0"]},
                "FP",
            )
            append_journal_record(journal, {"stage_index": stage_index, "status": "complete"}, "FP")

        resume = build_file_resume(out, journal, "FP")

        assert set(resume.outcomes) == {0, 1}
        assert set(resume.rows_by_stage) == {0, 1}
        assert not any(
            json.loads(line).get("status") == "restart_from_stage" for line in journal.read_text().splitlines()
        )

    def test_legacy_terminal_timeout_reopens_stage_and_preserves_timing(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        out.write_bytes(b"")
        journal = journal_path_for(out)
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        append_journal_record(
            journal,
            {
                "stage_index": 1,
                "status": "planned",
                "reference_ids": ["a"],
                "task_ids": ["t0"],
            },
            "FP",
        )
        append_journal_record(journal, {"stage_index": 1, "status": "complete"}, "FP")
        failures_path_for(out).write_text(
            json.dumps(
                {
                    "stage_index": 1,
                    TASK_INDEX_KEY_NAME: 7,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    NG_FAILURE_CLASS_KEY: "timeout_exceeded",
                    NG_TERMINAL_KEY: True,
                    "elapsed_seconds": 123.0,
                    "reuse_cached_deliverable": True,
                }
            )
            + "\n"
        )

        resume = build_file_resume(out, journal, "FP")

        assert 1 not in resume.outcomes
        assert (7, 0) not in resume.gated_keys.get(1, set())
        assert load_failure_timings(out)[1][(7, 0)] == 123.0
        assert load_failure_attempts(out)[1][(7, 0)] == 1
        assert load_reuse_cached_keys(out)[1] == {(7, 0)}
        assert resume.reuse_cached_keys[1] == {(7, 0)}
        assert resume.attempts_by_stage[1][(7, 0)] == 1

    def test_legacy_invalid_judge_main_row_migrates_before_resume(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        deliverables_dir = tmp_path / "deliverables" / "task-4"
        deliverables_dir.mkdir(parents=True)
        (deliverables_dir / "answer.txt").write_text("answer")
        out.write_text(
            json.dumps(
                {
                    "stage_index": 0,
                    TASK_INDEX_KEY_NAME: 4,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    "task_id": "t4",
                    "invalid_judge_response": True,
                    "reward": 0.0,
                    "deliverables_dir": str(deliverables_dir),
                }
            )
            + "\n"
        )
        journal = journal_path_for(out)
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        append_journal_record(
            journal,
            {"stage_index": 0, "status": "planned", "reference_ids": ["a"], "task_ids": ["t4"]},
            "FP",
        )
        append_journal_record(journal, {"stage_index": 0, "status": "complete"}, "FP")

        resume = build_file_resume(out, journal, "FP")

        assert out.read_bytes() == b""
        assert resume.rows_by_stage == {}
        assert resume.outcomes == {}
        assert resume.reuse_cached_keys[0] == {(4, 0)}
        assert resume.attempts_by_stage[0][(4, 0)] == 1
        migrated = [json.loads(line) for line in failures_path_for(out).read_text().splitlines()]
        assert len(migrated) == 1
        assert migrated[0][NG_FAILURE_CLASS_KEY] == "judge_invalid"
        assert migrated[0]["reuse_cached_deliverable"] is True

    def test_skipped_failure_keeps_stage_complete(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        out.write_bytes(b"")
        journal = journal_path_for(out)
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        append_journal_record(journal, {"stage_index": 0, "status": "complete"}, "FP")
        failures_path_for(out).write_text(
            json.dumps(
                {
                    "stage_index": 0,
                    TASK_INDEX_KEY_NAME: 7,
                    ROLLOUT_INDEX_KEY_NAME: 0,
                    NG_FAILURE_CLASS_KEY: "skipped",
                    NG_TERMINAL_KEY: True,
                }
            )
            + "\n"
        )

        resume = build_file_resume(out, journal, "FP")

        assert 0 in resume.outcomes
        assert (7, 0) in resume.gated_keys[0]

    async def test_failure_row_redispatched_but_terminal_not(self) -> None:
        # A failed (non-terminal, below max) row is re-dispatched; a terminal one is not.
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = _two_stage_cfg()
        full_run = _fake_run_rollouts_factory()

        all_results, base_summaries = await run_multistage_stages(
            cfg, REF_ELOS, _distribution(task_ids), rows, full_run
        )
        stage0_rows = [r for r in all_results if r["stage_index"] == 0]
        stage0_task_ids = list(dict.fromkeys(r["task_id"] for r in stage0_rows))
        # Cache all but the last two stage-0 successes.
        good = stage0_rows[:-2]
        failing_key = (stage0_rows[-1][TASK_INDEX_KEY_NAME], stage0_rows[-1][ROLLOUT_INDEX_KEY_NAME])
        terminal_key = (stage0_rows[-2][TASK_INDEX_KEY_NAME], stage0_rows[-2][ROLLOUT_INDEX_KEY_NAME])

        # gated_keys marks the terminal one only; the failing one is NOT gated.
        good_keys = {(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in good}
        resume = RecordingResume(
            plans={
                0: {"stage_index": 0, "reference_ids": base_summaries[0]["reference_ids"], "task_ids": stage0_task_ids}
            },
            rows_by_stage={0: good},
            gated_keys={0: good_keys | {terminal_key}},
        )

        dispatched_keys: List[Tuple[int, int]] = []

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                if r["stage_index"] == 0:
                    dispatched_keys.append((r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
            return await full_run(rows_in)

        await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, capturing_run, resume=resume)
        assert dispatched_keys == [failing_key]
        assert terminal_key not in dispatched_keys

    async def test_only_successes_reach_all_results(self) -> None:
        # A run whose fake runner emits one failure + one kill-shaped row per stage:
        # neither reaches all_results / pooling; only successes do.
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        cfg = _two_stage_cfg()

        async def mixed_run(rows_in: List[Dict[str, Any]]):
            pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for i, row in enumerate(rows_in):
                result = {"task_id": row["task_id"], "per_reference": {}, "reward": 1.0}
                if i % 3 == 1:
                    result[NG_FAILURE_CLASS_KEY] = "boom"
                elif i % 3 == 2:
                    result[NG_NO_PERSIST_KEY] = True
                pairs.append((row, result))
            return pairs

        all_results, _ = await run_multistage_stages(cfg, REF_ELOS, _distribution(task_ids), rows, mixed_run)
        assert all(NG_FAILURE_CLASS_KEY not in r for r in all_results)
        assert all(not r.get(NG_NO_PERSIST_KEY) for r in all_results)

    async def test_runtime_stale_stage_prunes_file_backed_downstream(self, tmp_path: Path) -> None:
        task_ids = ["t0", "t1", "t2"]
        rows = _materialized_rows(task_ids)
        cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3", "3:2"]}).stages,
            seed=0,
        )
        distribution = _distribution(task_ids)
        output = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(output)
        fingerprint = compute_fingerprint(cfg, REF_ELOS, distribution)
        successful_run = _fake_run_rollouts_factory()

        await run_multistage_stages(
            cfg,
            REF_ELOS,
            distribution,
            rows,
            successful_run,
            resume=build_file_resume(output, journal, fingerprint),
        )
        persisted = [json.loads(line) for line in output.read_text().splitlines()]
        missing = next(row for row in persisted if row["stage_index"] == 0)
        missing_key = (missing[TASK_INDEX_KEY_NAME], missing[ROLLOUT_INDEX_KEY_NAME])
        output.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in persisted
                if (row["stage_index"], row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]) != (0, *missing_key)
            )
        )
        aggregate_path = aggregate_metrics_path_for(output)
        aggregate_path.write_text("stale")

        resume = build_file_resume(output, journal, fingerprint)
        assert set(resume.outcomes) == {0, 1}
        dispatched: List[Tuple[int, Tuple[int, int]]] = []

        async def fresh_retry(rows_in: List[Dict[str, Any]]):
            dispatched.extend(
                (
                    row["stage_index"],
                    (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]),
                )
                for row in rows_in
            )
            pairs = await successful_run(rows_in)
            return [(row, {**result, "fresh_retry": True}) for row, result in pairs]

        await run_multistage_stages(
            cfg,
            REF_ELOS,
            distribution,
            rows,
            fresh_retry,
            resume=resume,
        )

        assert [key for stage, key in dispatched if stage == 0] == [missing_key]
        assert len([key for stage, key in dispatched if stage == 1]) == 3
        assert not aggregate_path.exists()
        final_rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len([row for row in final_rows if row["stage_index"] == 1]) == 3
        assert all(row["fresh_retry"] is True for row in final_rows if row["stage_index"] == 1)
        assert any(
            json.loads(line).get("status") == "restart_from_stage" and json.loads(line)["stage_index"] == 0
            for line in journal.read_text().splitlines()
        )
        assert set(read_journal(journal)[1]) == {0, 1}

    async def test_file_backed_resume_end_to_end(self, tmp_path: Path) -> None:
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        dist = _distribution(task_ids)
        run = _fake_run_rollouts_factory()

        # Reference: uninterrupted 2-stage run.
        _, ref_summaries = await run_multistage_stages(_two_stage_cfg(), REF_ELOS, dist, rows, run)

        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        fp = compute_fingerprint(_two_stage_cfg(), REF_ELOS, dist)

        # Simulate a crash after stage 0: run only stage 0 through a file-backed
        # resume so its plan + rows + outcome are persisted to real files.
        stage0_cfg = MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": ["3"]}).stages,
            seed=0,
        )
        await run_multistage_stages(stage0_cfg, REF_ELOS, dist, rows, run, resume=build_file_resume(out, journal, fp))

        # Resume the full 2-stage run from the persisted files.
        dispatched_stages: List[int] = []

        async def capturing_run(rows_in: List[Dict[str, Any]]):
            for r in rows_in:
                dispatched_stages.append(r["stage_index"])
            return await run(rows_in)

        _, summaries = await run_multistage_stages(
            _two_stage_cfg(), REF_ELOS, dist, rows, capturing_run, resume=build_file_resume(out, journal, fp)
        )

        # Stage 0 reused from cache (never dispatched); only stage 1 ran.
        assert 0 not in dispatched_stages
        assert set(dispatched_stages) == {1}
        assert summaries[0]["cached"] is True
        # Threaded ELO + downstream selection match the uninterrupted run.
        assert summaries[0]["eval_elo"] == ref_summaries[0]["eval_elo"]
        assert summaries[1]["reference_ids"] == ref_summaries[1]["reference_ids"]
        assert summaries[1]["eval_elo"] == ref_summaries[1]["eval_elo"]


class TestPrepareResume:
    """The integration wiring: a fresh run must persist so a later resume can read."""

    def _cfg(self, resume_from_cache: bool) -> SimpleNamespace:
        return SimpleNamespace(resume_from_cache=resume_from_cache)

    def test_fresh_returns_writing_resume_with_empty_state(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        fp = compute_fingerprint(_two_stage_cfg(), REF_ELOS, _distribution(["t0"]))
        resume = _prepare_resume(self._cfg(True), out, journal, fp)
        assert isinstance(resume, StageResume)
        assert resume.plans == {} and resume.outcomes == {} and resume.rows_by_stage == {}
        resume.on_plan(0, {"stage_index": 0, "status": "planned", "reference_ids": ["a"], "task_ids": ["t0"]})
        assert journal.exists()
        plans, _, got_fp = read_journal(journal)
        assert 0 in plans and got_fp == fp

    def test_resume_disabled_clears_existing_and_empties_state(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        metrics = tmp_path / "rollouts_aggregate_metrics.json"
        out.write_text('{"x": 1}\n')
        journal.write_text('{"stage_index": 0, "status": "complete"}\n')
        metrics.write_text('{"stale": true}\n')
        fp = compute_fingerprint(_two_stage_cfg(), REF_ELOS, _distribution(["t0"]))
        resume = _prepare_resume(self._cfg(False), out, journal, fp)
        assert isinstance(resume, StageResume)
        assert not out.exists() and not journal.exists() and not metrics.exists()
        assert resume.outcomes == {}

    def test_stale_fingerprint_clears_and_starts_fresh(self, tmp_path: Path) -> None:
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        dist = _distribution(["t0"])
        out.write_text('{"stage_index": 0}\n')
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        append_journal_record(journal, {"stage_index": 0, "status": "complete"}, "STALEFP")
        fp = compute_fingerprint(_two_stage_cfg(), REF_ELOS, dist)
        resume = _prepare_resume(self._cfg(True), out, journal, fp)
        assert not out.exists() and not journal.exists()
        assert resume.outcomes == {}

    def test_runtime_bind_address_change_preserves_resume_state(self, tmp_path: Path) -> None:
        from resources_servers.gdpval.multistage_orchestrator import append_journal_record

        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        out.write_text("")
        dist = _distribution(["t0"])
        old_fingerprint = compute_fingerprint(
            _two_stage_cfg(),
            REF_ELOS,
            dist,
            resolved_global_config=_runtime_components_with_bind_addresses("node-a", 0),
        )
        append_journal_record(journal, {"stage_index": 0, "status": "complete"}, old_fingerprint)
        rebound_fingerprint = compute_fingerprint(
            _two_stage_cfg(),
            REF_ELOS,
            dist,
            resolved_global_config=_runtime_components_with_bind_addresses("node-b", 100),
        )

        resume = _prepare_resume(self._cfg(True), out, journal, rebound_fingerprint)

        assert out.exists() and journal.exists()
        assert set(resume.outcomes) == {0}

    async def test_fresh_run_persists_journal_then_resume_reuses_all(self, tmp_path: Path) -> None:
        # Regression: a fresh run through _prepare_resume must write the journal +
        # rows, so a second _prepare_resume resumes without re-dispatching anything.
        task_ids = [f"t{i}" for i in range(10)]
        rows = _materialized_rows(task_ids)
        dist = _distribution(task_ids)
        run = _fake_run_rollouts_factory()
        out = tmp_path / "rollouts.jsonl"
        journal = journal_path_for(out)
        fp = compute_fingerprint(_two_stage_cfg(), REF_ELOS, dist)
        cfg = self._cfg(True)

        r1 = _prepare_resume(cfg, out, journal, fp)
        _, base = await run_multistage_stages(_two_stage_cfg(), REF_ELOS, dist, rows, run, resume=r1)
        assert journal.exists() and out.exists()
        plans, outcomes, _ = read_journal(journal)
        assert set(plans) == {0, 1} and set(outcomes) == {0, 1}

        r2 = _prepare_resume(cfg, out, journal, fp)
        assert set(r2.outcomes) == {0, 1}

        async def no_dispatch(rows_in: List[Dict[str, Any]]):
            raise AssertionError(f"resume re-dispatched {len(rows_in)} rows; expected full cache reuse")

        _, again = await run_multistage_stages(_two_stage_cfg(), REF_ELOS, dist, rows, no_dispatch, resume=r2)
        assert all(s["cached"] for s in again)
        assert [s["reference_ids"] for s in again] == [s["reference_ids"] for s in base]
        assert [s["eval_elo"] for s in again] == [s["eval_elo"] for s in base]


class TestIntegrationWiring:
    async def test_multistage_forwards_one_budget_and_shared_tracker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: List[Dict[str, Any]] = []
        cache_namespaces: List[str] = []

        class FakeHelper:
            def _preprocess_rows_from_config(self, config):
                return _materialized_rows(["t0"])

            def run_examples(self, rows, **kwargs):
                calls.append(kwargs)
                cache_namespaces.extend(row["verify_cache_namespace"] for row in rows)

                async def done(row):
                    reference_id = row["reference_ids"][0]
                    return row, {
                        "task_id": row["task_id"],
                        "per_reference": {
                            reference_id: {
                                "wins": 1,
                                "losses": 0,
                                "ties": 0,
                                "reference_elo": REF_ELOS[reference_id],
                            }
                        },
                    }

                return [done(row) for row in rows]

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return tmp_path / "aggregate.json"

        monkeypatch.setattr(rollout_collection_module, "RolloutCollectionHelper", FakeHelper)
        monkeypatch.setattr(
            "resources_servers.gdpval.multistage_orchestrator.ensure_distribution",
            lambda *args, **kwargs: (_distribution(["t0"]), None),
        )
        config = SimpleNamespace(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            num_samples_in_parallel=2,
            dispatch_budget_s=120.0,
            drain_margin_s=15.0,
            dispatch_longest_first=True,
            resume_from_cache=False,
        )
        global_config = {
            "multistage": {"enabled": True, "stages": ["1", "1"], "seed": 0},
            "gdpval": {
                "resources_servers": {
                    "gdpval": {"reference_models": {key: {"elo": elo} for key, elo in REF_ELOS.items()}}
                }
            },
        }

        await run_e2e_multistage(config, global_config)

        assert len(calls) == 2
        assert calls[0]["latency_tracker"] is calls[1]["latency_tracker"]
        assert 0 <= calls[1]["dispatch_budget_s"] <= calls[0]["dispatch_budget_s"] <= 120.0
        assert [call["drain_margin_s"] for call in calls] == [15.0, 15.0]
        persisted = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
        assert {row["expected_final_stage_index"] for row in persisted} == {1}
        assert {row["expected_stage_row_count"] for row in persisted} == {1}
        assert len(set(cache_namespaces)) == 1
        assert len(cache_namespaces[0]) == 64


class TestReferenceMissingCoverage:
    """A terminal ``reference_missing`` row is an omission, not a stage-killer.

    Pins the behaviour the failure-class change relies on: because the row is
    terminal it is "already resolved", so it never reaches the unresolved-key
    loop that consults ``waivable_failure_classes``. It is simply an omission
    governed by the coverage floors -- which is why no leaf policy needs to list
    ``reference_missing`` as waivable.
    """

    @staticmethod
    def _rows(n: int, ref: str):
        return [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "reference_ids": [ref]} for i in range(n)]

    def _outcome(self, *, judged: int, planned: int, ref: str = "r1"):
        from resources_servers.gdpval.multistage_elo import PartialStagePolicy
        from resources_servers.gdpval.multistage_orchestrator import _partial_stage_outcome

        stage_rows = self._rows(planned, ref)
        successful = []
        for i in range(judged):
            row = dict(stage_rows[i])
            row["per_reference"] = {ref: {"wins": 4, "losses": 0, "ties": 0}}
            successful.append(row)
        policy = PartialStagePolicy(
            min_success_fraction=0.9,
            min_per_reference_success_fraction=0.5,
            min_successful_rows_per_reference=1,
            waivable_failure_classes=("timeout_exceeded", "transient"),
        )
        return _partial_stage_outcome(policy, stage_rows, successful, [], set(), [ref], 1200.0, 1)

    def test_terminal_omission_is_accepted_without_being_waivable(self) -> None:
        # 19 of 20 judged: the missing one is the reference_missing row. It is
        # NOT in successful_rows and NOT in unresolved_keys (terminal).
        outcome = self._outcome(judged=19, planned=20)
        assert outcome is not None
        assert outcome["success_fraction"] == pytest.approx(0.95)
        assert len(outcome["omitted_keys"]) == 1

    def test_still_rejected_when_coverage_floor_is_breached(self) -> None:
        # Forgiveness is bounded: 17/20 = 0.85 is below min_success_fraction 0.9.
        assert self._outcome(judged=17, planned=20) is None

    def test_evidence_less_success_row_still_hard_rejects(self) -> None:
        """The gate this change routes around is intentionally left intact: a
        row that IS in the success set while carrying no battle stays fatal."""
        from resources_servers.gdpval.multistage_elo import PartialStagePolicy
        from resources_servers.gdpval.multistage_orchestrator import _partial_stage_outcome

        ref = "r1"
        stage_rows = self._rows(20, ref)
        successful = []
        for i, row in enumerate(stage_rows):
            r = dict(row)
            r["per_reference"] = {} if i == 0 else {ref: {"wins": 4, "losses": 0, "ties": 0}}
            successful.append(r)
        policy = PartialStagePolicy(
            min_success_fraction=0.9,
            min_per_reference_success_fraction=0.5,
            min_successful_rows_per_reference=1,
        )
        assert _partial_stage_outcome(policy, stage_rows, successful, [], set(), [ref], 1200.0, 1) is None


class TestReferenceMissingIsTerminal:
    """A missing reference deliverable can never be fixed by retrying.

    The verify response stamps the generic ``_ng_failure_terminal`` flag rather
    than teaching ``_is_terminal_failure`` a GDPVal-specific class name -- the
    harness already honours that flag, so no change to nemo_gym is needed.
    """

    def test_generic_terminal_flag_is_honoured(self) -> None:
        from nemo_gym.rollout_collection import NG_TERMINAL_KEY, _is_terminal_failure

        assert _is_terminal_failure({NG_TERMINAL_KEY: True}) is True

    def test_environment_faults_are_revalidated_on_resume(self) -> None:
        # reference_missing/eval_missing are terminal within a run (retrying
        # cannot make the file appear) but the tree can be repaired between
        # runs, so resume re-dispatches them for a cheap /verify recheck.
        from nemo_gym.rollout_collection import (
            NG_FAILURE_CLASS_KEY,
            NG_TERMINAL_KEY,
            _is_terminal_failure,
        )
        from resources_servers.gdpval.app import (
            EVAL_MISSING_FAILURE_CLASS,
            REFERENCE_MISSING_FAILURE_CLASS,
        )

        for failure_class in (REFERENCE_MISSING_FAILURE_CLASS, EVAL_MISSING_FAILURE_CLASS):
            row = {NG_FAILURE_CLASS_KEY: failure_class, NG_TERMINAL_KEY: True}
            assert _is_terminal_failure(row) is False

    def test_timeout_stays_retryable(self) -> None:
        from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, _is_terminal_failure

        assert _is_terminal_failure({NG_FAILURE_CLASS_KEY: "timeout_exceeded"}) is False

    def test_reference_missing_row_is_not_a_success(self) -> None:
        """The point of the change: it must not reach the success set, where the
        non-final-stage coverage gate would reject it before any threshold."""
        from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
        from resources_servers.gdpval.app import REFERENCE_MISSING_FAILURE_CLASS
        from resources_servers.gdpval.multistage_orchestrator import _is_success_row

        row = {
            "task_id": "t1",
            "reward": 0.0,
            "judge_response": {"error": "reference_missing"},
            NG_FAILURE_CLASS_KEY: REFERENCE_MISSING_FAILURE_CLASS,
        }
        assert _is_success_row(row) is False
