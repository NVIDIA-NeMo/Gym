# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes
from benchmarks.vs_bench_gameplay_poker_shard.prepare import prepare


def test_shards_partition_paper_poker_without_duplicate_or_missing_tasks(tmp_path):
    shards = []
    for shard in (0, 1):
        path = prepare(tmp_path / f"{shard}.jsonl", shard=shard)
        rows = [json.loads(line)["verifier_metadata"]["paper_episode"] for line in path.read_text().splitlines()]
        assert len(rows) == 600
        shards.append({(row["block"], row["index"], row["environment_seed"], row["model_seat"]) for row in rows})
    expected = {
        (row.block, row.index, row.environment_seed, row.model_seat)
        for row in paper_competitive_episodes("kuhn_poker")
    }
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == expected


def test_shard_outcomes_merge_by_episode_identity_and_cover_all_blocks(tmp_path):
    from responses_api_agents.vs_bench_gameplay.app import VSBenchGameplayAgent

    tasks = []
    for shard in (0, 1):
        path = prepare(tmp_path / f"merge-{shard}.jsonl", shard=shard)
        for line in path.read_text().splitlines():
            meta = json.loads(line)["verifier_metadata"]
            tasks.append(
                [
                    {
                        "verifier_metadata": meta,
                        "model_seats": [meta["paper_episode"]["model_seat"]],
                        "reward": 0.0,
                        "terminal_returns": [[0.0, 0.0]],
                        "episode_steps": [1],
                    }
                ]
            )
    metrics = VSBenchGameplayAgent.compute_metrics(None, tasks)
    assert metrics["vs_bench/gameplay/kuhn_poker/episodes"] == 1200
    assert metrics["vs_bench/gameplay/kuhn_poker/normalized_approx"] == 100
    assert all(metrics[f"vs_bench/gameplay/kuhn_poker/block_{i}/raw_mean"] == 0 for i in range(10))
