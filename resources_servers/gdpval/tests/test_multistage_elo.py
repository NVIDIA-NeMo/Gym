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
import json
import random
from pathlib import Path

import pytest

from resources_servers.gdpval.multistage_elo import (
    StageSpec,
    all_task_ids,
    assign_task_references,
    ensure_distribution,
    fit_stage_elo,
    load_distribution,
    plan_stage_task_ids,
    pool_per_reference,
    select_references,
    stage_assignment_rng,
)


def _dist(groups):
    """groups: {key: [task_ids]} -> distribution dict with proportional pct."""
    total = sum(len(v) for v in groups.values()) or 1
    return {k: {"percentage": len(v) / total, "task_ids": list(v)} for k, v in groups.items()}


class TestSelectReferences:
    ELOS = {"a": 1000.0, "b": 1200.0, "c": 1300.0, "d": 1500.0}

    def test_all_when_num_models_none(self) -> None:
        assert select_references(self.ELOS, 1234.0, None) == ["a", "b", "c", "d"]

    def test_all_when_eval_elo_none(self) -> None:
        assert select_references(self.ELOS, None, 2) == ["a", "b", "c", "d"]

    def test_all_when_num_models_exceeds_available(self) -> None:
        assert select_references(self.ELOS, 1234.0, 10) == ["a", "b", "c", "d"]

    def test_closest_subset(self) -> None:
        # eval 1250 -> closest are c(1300,50) and b(1200,50); tie broken by id.
        assert select_references(self.ELOS, 1250.0, 2) == ["b", "c"]

    def test_closest_single(self) -> None:
        assert select_references(self.ELOS, 1490.0, 1) == ["d"]

    def test_zero_models_returns_empty(self) -> None:
        assert select_references(self.ELOS, 1250.0, 0) == []

    def test_result_sorted_by_id(self) -> None:
        chosen = select_references(self.ELOS, 1100.0, 3)
        assert chosen == sorted(chosen)


class TestAllTaskIds:
    def test_returns_every_task_deduped_in_order(self) -> None:
        dist = {
            "g0": {"percentage": 0.5, "task_ids": ["t0", "t1"]},
            "g1": {"percentage": 0.5, "task_ids": ["t2", "t1"]},  # t1 repeated across groups
        }
        assert all_task_ids(dist) == ["t0", "t1", "t2"]

    def test_empty_distribution(self) -> None:
        assert all_task_ids({}) == []

    def test_stringifies_ids(self) -> None:
        assert all_task_ids({"g": {"task_ids": [0, 1]}}) == ["0", "1"]


class TestAssignTaskReferences:
    def test_one_reference_per_task_from_included_set(self) -> None:
        task_ids = [f"t{i}" for i in range(50)]
        refs = ["a", "b", "c"]
        assignment = assign_task_references(task_ids, refs, rng=random.Random(0))
        # Every task gets exactly one reference, always from the included set.
        assert set(assignment) == set(task_ids)
        assert all(ref in refs for ref in assignment.values())

    def test_equal_probability_across_references(self) -> None:
        # Over many tasks, a uniform draw spreads roughly evenly across refs.
        task_ids = [f"t{i}" for i in range(3000)]
        refs = ["a", "b", "c"]
        assignment = assign_task_references(task_ids, refs, rng=random.Random(7))
        counts = {r: 0 for r in refs}
        for ref in assignment.values():
            counts[ref] += 1
        # Each ref should get ~1000; allow generous slack for randomness.
        for ref in refs:
            assert 800 < counts[ref] < 1200

    def test_deterministic_for_same_rng_seed(self) -> None:
        task_ids = [f"t{i}" for i in range(20)]
        refs = ["a", "b"]
        a = assign_task_references(task_ids, refs, rng=stage_assignment_rng(0, None, 1))
        b = assign_task_references(task_ids, refs, rng=stage_assignment_rng(0, None, 1))
        assert a == b

    def test_empty_reference_set_yields_no_assignment(self) -> None:
        assert assign_task_references(["t0", "t1"], [], rng=random.Random(0)) == {}


class TestStageAssignmentRng:
    def test_distinct_streams_per_stage_and_seed(self) -> None:
        base = stage_assignment_rng(0, None, 0).random()
        assert stage_assignment_rng(0, None, 1).random() != base
        assert stage_assignment_rng(1, None, 0).random() != base
        assert stage_assignment_rng(0, 5, 0).random() != base

    def test_reproducible(self) -> None:
        assert stage_assignment_rng(3, 9, 2).random() == stage_assignment_rng(3, 9, 2).random()


class TestPlanStageTaskIds:
    def test_nested_is_superset(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(10)], "y": [f"y{i}" for i in range(10)]})
        stages = [StageSpec(num_tasks=3), StageSpec(num_tasks=8)]
        planned = plan_stage_task_ids(dist, stages, rng=random.Random(0), nested=True)
        assert len(planned[0]) == 3
        assert len(planned[1]) == 8
        assert set(planned[0]).issubset(set(planned[1]))

    def test_nested_no_duplicates(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(20)]})
        stages = [StageSpec(num_tasks=5), StageSpec(num_tasks=12)]
        planned = plan_stage_task_ids(dist, stages, rng=random.Random(1), nested=True)
        assert len(planned[1]) == len(set(planned[1]))

    def test_nested_capped_at_available(self) -> None:
        dist = _dist({"x": ["a", "b", "c"]})
        stages = [StageSpec(num_tasks=2), StageSpec(num_tasks=100)]
        planned = plan_stage_task_ids(dist, stages, rng=random.Random(2), nested=True)
        assert sorted(planned[1]) == ["a", "b", "c"]

    def test_non_increasing_stage_reuses_prefix(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(10)]})
        stages = [StageSpec(num_tasks=5), StageSpec(num_tasks=3)]
        planned = plan_stage_task_ids(dist, stages, rng=random.Random(3), nested=True)
        assert planned[1] == planned[0][:3]

    def test_independent_sampling(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(50)]})
        stages = [StageSpec(num_tasks=5, seed=1), StageSpec(num_tasks=5, seed=2)]
        planned = plan_stage_task_ids(dist, stages, nested=False)
        assert len(planned[0]) == 5 and len(planned[1]) == 5

    def test_seed_reproducible(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(50)]})
        stages = [StageSpec(num_tasks=7, seed=42)]
        a = plan_stage_task_ids(dist, stages, nested=False)
        b = plan_stage_task_ids(dist, stages, nested=False)
        assert a == b

    def test_num_tasks_none_defaults_to_full_dataset(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(10)], "y": [f"y{i}" for i in range(5)]})
        # A stage with num_tasks unset judges every task in the distribution.
        planned = plan_stage_task_ids(dist, [StageSpec()], rng=random.Random(0), nested=False)
        assert len(planned[0]) == 15
        assert set(planned[0]) == set(all_task_ids(dist))

    def test_num_tasks_none_nested_full_then_prefix(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(10)]})
        # None (full) as the largest stage; a smaller explicit stage is a prefix.
        planned = plan_stage_task_ids(dist, [StageSpec(num_tasks=4), StageSpec()], rng=random.Random(1), nested=True)
        assert len(planned[0]) == 4
        assert len(planned[1]) == 10
        assert set(planned[0]).issubset(set(planned[1]))


class TestFitStageElo:
    ELOS = {"a": 1000.0, "b": 1400.0}

    def test_no_battles_returns_none(self) -> None:
        assert fit_stage_elo({}, self.ELOS) == (None, None, 0)

    def test_zero_games_skipped(self) -> None:
        per_ref = {"a": {"wins": 0, "losses": 0, "ties": 0}}
        assert fit_stage_elo(per_ref, self.ELOS) == (None, None, 0)

    def test_fits_elo_uses_config_anchor(self) -> None:
        per_ref = {"a": {"wins": 5, "losses": 5, "ties": 0}}
        elo, norm, n = fit_stage_elo(per_ref, self.ELOS)
        # 50% win rate vs a single anchor -> eval elo ~= anchor elo.
        assert n == 1
        assert elo == pytest.approx(1000.0, abs=1.0)
        assert norm == pytest.approx((elo - 500.0) / 2000.0)

    def test_falls_back_to_recorded_reference_elo(self) -> None:
        per_ref = {"z": {"wins": 5, "losses": 5, "ties": 0, "reference_elo": 1100.0}}
        elo, _norm, n = fit_stage_elo(per_ref, {})
        assert n == 1
        assert elo == pytest.approx(1100.0, abs=1.0)

    def test_multi_reference_battles(self) -> None:
        per_ref = {
            "a": {"wins": 8, "losses": 2, "ties": 0},
            "b": {"wins": 2, "losses": 8, "ties": 0},
        }
        elo, _norm, n = fit_stage_elo(per_ref, self.ELOS)
        assert n == 2
        assert 1000.0 < elo < 1400.0


class TestPoolPerReference:
    def test_sums_counts_across_responses(self) -> None:
        responses = [
            {"per_reference": {"a": {"wins": 1, "losses": 2, "ties": 0, "reference_elo": 1000.0}}},
            {"per_reference": {"a": {"wins": 3, "losses": 0, "ties": 1}, "b": {"wins": 2, "losses": 2, "ties": 0}}},
        ]
        pooled = pool_per_reference(responses)
        assert pooled["a"] == {"wins": 4, "losses": 2, "ties": 1, "reference_elo": 1000.0}
        assert pooled["b"]["wins"] == 2 and pooled["b"]["losses"] == 2

    def test_handles_missing_per_reference(self) -> None:
        assert pool_per_reference([{}, {"per_reference": None}]) == {}


class TestEnsureDistribution:
    def test_loads_existing_distribution_file(self, tmp_path: Path) -> None:
        dist = _dist({"x": ["t0", "t1"]})
        path = tmp_path / "d.json"
        path.write_text(json.dumps(dist))
        loaded, returned_path = ensure_distribution(str(path))
        assert loaded == dist
        assert returned_path == path

    def test_load_distribution_rejects_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError):
            load_distribution(path)

    def test_builds_and_caches_when_missing(self, tmp_path: Path, monkeypatch) -> None:
        dataset = tmp_path / "data.jsonl"
        dataset.write_text("")  # contents irrelevant; build is monkeypatched
        built = _dist({"occupation=x": ["t0", "t1"]})

        import responses_api_agents.stirrup_agent.task_distribution as td

        monkeypatch.setattr(td, "build_distribution_from_dataset", lambda path, cols: built)

        out = tmp_path / "cache" / "occupation_distribution.json"
        loaded, path = ensure_distribution(
            None, dataset_path=str(dataset), columns=["occupation"], cache_dir=str(tmp_path / "cache")
        )
        assert loaded == built
        assert path == out
        assert json.loads(out.read_text()) == built

    def test_raises_when_no_distribution_and_no_dataset(self, monkeypatch) -> None:
        import responses_api_agents.stirrup_agent.task_distribution as td

        monkeypatch.setattr(td, "resolve_default_dataset", lambda: None)
        with pytest.raises(FileNotFoundError):
            ensure_distribution(None)
