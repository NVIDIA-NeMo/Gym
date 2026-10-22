# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
from collections import Counter
from pathlib import Path

from pytest import MonkeyPatch

import responses_api_models.vllm_model.least_loaded as least_loaded
from responses_api_models.vllm_model.least_loaded import LeastLoadedPlacement


def _place_many(path: str, worker: int, n_rollouts: int, n_endpoints: int) -> None:
    placement = LeastLoadedPlacement(path, live_s=600.0, busy_timeout_s=10.0)
    for i in range(n_rollouts):
        placement.place(f"worker-{worker}-rollout-{i}", n_endpoints)


class TestLeastLoadedPlacement:
    def test_new_rollouts_fill_the_least_loaded_endpoints(self, tmp_path: Path) -> None:
        placement = LeastLoadedPlacement(str(tmp_path / "state.sqlite"), live_s=600.0)

        endpoints = [placement.place(f"rollout-{i}", 4) for i in range(12)]

        assert Counter(endpoints) == {0: 3, 1: 3, 2: 3, 3: 3}

    def test_a_rollout_keeps_its_endpoint(self, tmp_path: Path) -> None:
        placement = LeastLoadedPlacement(str(tmp_path / "state.sqlite"), live_s=600.0)
        first = placement.place("rollout-a", 4)
        for i in range(10):
            placement.place(f"other-{i}", 4)

        assert placement.place("rollout-a", 4) == first

    def test_workers_sharing_the_file_agree_and_count_each_others_rollouts(self, tmp_path: Path) -> None:
        path = str(tmp_path / "state.sqlite")
        worker_a = LeastLoadedPlacement(path, live_s=600.0)
        worker_b = LeastLoadedPlacement(path, live_s=600.0)

        placed_by_a = worker_a.place("rollout-a", 2)
        # The other endpoint is now the least loaded, whichever worker places the next rollout.
        placed_by_b = worker_b.place("rollout-b", 2)

        assert placed_by_b != placed_by_a
        assert worker_b.place("rollout-a", 2) == placed_by_a

    def test_concurrent_worker_processes_balance_exactly(self, tmp_path: Path) -> None:
        path = str(tmp_path / "state.sqlite")
        LeastLoadedPlacement(path, live_s=600.0).place("warm-up", 4)
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_place_many, args=(path, w, 25, 4)) for w in range(4)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0

        reader = LeastLoadedPlacement(path, live_s=600.0)
        connection = reader._connect()
        counts = dict(
            connection.execute(
                "select endpoint, count(*) from placements where rollout_id != 'warm-up' group by endpoint"
            ).fetchall()
        )
        warm_up = reader.place("warm-up", 4)
        counts[warm_up] = counts.get(warm_up, 0) + 1
        assert sorted(counts.values()) == [25, 25, 25, 26]

    def test_rollouts_idle_past_the_live_window_stop_counting(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        clock = [1000.0]
        monkeypatch.setattr(least_loaded, "time", lambda: clock[0])
        placement = LeastLoadedPlacement(str(tmp_path / "state.sqlite"), live_s=60.0)
        connection = placement._connect()
        for i in range(3):
            connection.execute("insert into placements values (?, 0, 2, ?)", (f"idle-{i}", clock[0]))

        clock[0] += 61.0
        # Counted, the three idle rollouts on endpoint 0 would send both new rollouts to endpoint 1.
        endpoints = [placement.place(f"rollout-{i}", 2) for i in range(2)]

        assert sorted(endpoints) == [0, 1]

    def test_a_changed_endpoint_count_places_the_rollout_again(self, tmp_path: Path) -> None:
        placement = LeastLoadedPlacement(str(tmp_path / "state.sqlite"), live_s=600.0)
        for i in range(3):
            placement.place(f"rollout-{i}", 3)

        assert placement.place("rollout-0", 1) == 0

    async def test_later_calls_are_served_from_the_worker_cache(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        placement = LeastLoadedPlacement(str(tmp_path / "state.sqlite"), live_s=600.0)
        calls = []
        place = placement.place
        monkeypatch.setattr(placement, "place", lambda rid, n: calls.append(rid) or place(rid, n))

        first = await placement.endpoint_for("rollout-a", 4)
        for _ in range(5):
            assert await placement.endpoint_for("rollout-a", 4) == first

        assert calls == ["rollout-a"]

    async def test_an_unavailable_state_file_yields_no_placement(self, tmp_path: Path) -> None:
        placement = LeastLoadedPlacement(str(tmp_path / "missing-dir" / "state.sqlite"), live_s=600.0)

        assert await placement.endpoint_for("rollout-a", 4) is None
