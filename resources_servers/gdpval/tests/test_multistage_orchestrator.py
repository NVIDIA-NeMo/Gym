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
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    _failures_path_for,
)
from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    StageResume,
    _prepare_resume,
    build_file_resume,
    build_stage_rows,
    compute_fingerprint,
    find_gdpval_reference_elos,
    index_rows_by_task,
    journal_path_for,
    load_gated_keys,
    load_persisted_rows,
    parse_multistage_config,
    read_journal,
    route_stage_rows,
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


# ---------------------------------------------------------------------------
# Resume seam (pure, in-memory)
# ---------------------------------------------------------------------------


class RecordingResume(StageResume):
    """In-memory StageResume that records callback invocations.

    ``gated_keys`` defaults to the successes in ``rows_by_stage``; pass it
    explicitly to model terminal / max-attempt gating from the sidecar.
    """

    def __init__(self, plans=None, outcomes=None, rows_by_stage=None, gated_keys=None) -> None:
        self.planned: List[Tuple[int, dict]] = []
        self.completed: List[Tuple[int, dict]] = []
        self.appended: Dict[int, List[Dict[str, Any]]] = {}
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
            "task_ids": [f"t{i}" for i in range(3)],
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
        sidecar = [json.loads(line) for line in _failures_path_for(out).read_text().splitlines()]
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
        sidecar = _failures_path_for(out)
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
        out.write_text('{"x": 1}\n')
        journal.write_text('{"stage_index": 0, "status": "complete"}\n')
        fp = compute_fingerprint(_two_stage_cfg(), REF_ELOS, _distribution(["t0"]))
        resume = _prepare_resume(self._cfg(False), out, journal, fp)
        assert isinstance(resume, StageResume)
        assert not out.exists() and not journal.exists()
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
