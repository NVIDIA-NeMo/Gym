# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes
from benchmarks.vs_bench_gameplay_breakthrough_shard.prepare import prepare
from responses_api_agents.vs_bench_gameplay.app import VSBenchGameplayAgent


def test_shards_partition_and_merge_all_paper_games(tmp_path):
    shards, tasks = [], []
    for seat in (0, 1):
        path = prepare(tmp_path / f"{seat}.jsonl", seat=seat)
        metas = [json.loads(line)["verifier_metadata"] for line in path.read_text().splitlines()]
        assert len(metas) == 10
        assert {meta["paper_episode"]["model_seat"] for meta in metas} == {seat}
        shards.append({(meta["paper_episode"]["block"], meta["paper_episode"]["index"]) for meta in metas})
        for meta in metas:
            tasks.append(
                [
                    {
                        "verifier_metadata": meta,
                        "model_seats": [seat],
                        "reward": 0.0,
                        "terminal_returns": [[0.0, 0.0]],
                        "episode_steps": [1],
                    }
                ]
            )
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == {
        (episode.block, episode.index) for episode in paper_competitive_episodes("breakthrough")
    }
    metrics = VSBenchGameplayAgent.compute_metrics(None, tasks)
    assert metrics["vs_bench/gameplay/breakthrough/episodes"] == 20
    assert metrics["vs_bench/gameplay/breakthrough/normalized_approx"] == 50
