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
import random

import pytest

from resources_servers.gdpval.multistage_elo import (
    MultiStageEloConfig,
    MultiStageEloRunner,
    StageSpec,
    fit_stage_elo,
    plan_stage_task_ids,
    select_references,
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


class TestMultiStageEloRunner:
    def _config(self, **overrides):
        base = dict(
            distribution_path="unused.json",
            stages=[StageSpec(num_tasks=3, num_models=None), StageSpec(num_tasks=6, num_models=2)],
            reference_elos={"a": 1000.0, "b": 1200.0, "c": 1300.0, "d": 1500.0},
        )
        base.update(overrides)
        return MultiStageEloConfig(**base)

    def test_requires_stages(self) -> None:
        with pytest.raises(ValueError):
            MultiStageEloConfig(distribution_path="x", stages=[], reference_elos={})

    def test_unknown_selection_rejected(self) -> None:
        with pytest.raises(ValueError):
            MultiStageEloConfig(distribution_path="x", stages=[StageSpec(1)], reference_elos={}, selection="zzz")

    def test_two_stage_flow_threads_elo_and_shrinks_refs(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(20)]})
        seen_stage_refs = []

        def judge_stage(task_ids, reference_ids):
            seen_stage_refs.append(list(reference_ids))
            # Eval beats everyone 7-3 -> high elo estimate.
            return {rid: {"wins": 7, "losses": 3, "ties": 0} for rid in reference_ids}

        runner = MultiStageEloRunner(self._config(nested_tasks=True), dist, judge_stage, rng=random.Random(0))
        results = runner.run()

        assert len(results) == 2
        # Stage 1 uses all references.
        assert seen_stage_refs[0] == ["a", "b", "c", "d"]
        # Stage 2 narrows to 2 references (closest to the stage-1 estimate).
        assert len(seen_stage_refs[1]) == 2
        assert set(seen_stage_refs[1]).issubset({"a", "b", "c", "d"})
        # Nested task sets (nested_tasks=True): stage 2 superset of stage 1.
        assert set(results[0].task_ids).issubset(set(results[1].task_ids))
        assert results[1].eval_elo is not None

    def test_stage_with_no_games_leaves_elo_unset(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(10)]})

        def judge_stage(task_ids, reference_ids):
            return {}

        cfg = self._config(stages=[StageSpec(num_tasks=2, num_models=None)])
        results = MultiStageEloRunner(cfg, dist, judge_stage, rng=random.Random(0)).run()
        assert results[0].eval_elo is None
        assert results[0].num_references == 0

    def test_on_event_emits_lifecycle_events(self) -> None:
        dist = _dist({"x": [f"x{i}" for i in range(10)]})

        def judge_stage(task_ids, reference_ids):
            return {rid: {"wins": 6, "losses": 4, "ties": 0} for rid in reference_ids}

        events = []
        cfg = self._config(stages=[StageSpec(num_tasks=2, num_models=None), StageSpec(num_tasks=3, num_models=2)])
        MultiStageEloRunner(
            cfg, dist, judge_stage, rng=random.Random(0), on_event=lambda name, data: events.append((name, data))
        ).run()

        names = [n for n, _ in events]
        assert names[0] == "planned"
        assert names.count("stage_start") == 2
        assert names.count("stage_end") == 2
        # stage_start carries the selected references and task count.
        first_start = next(d for n, d in events if n == "stage_start")
        assert first_start["num_tasks"] == 2
        assert first_start["reference_ids"] == ["a", "b", "c", "d"]
