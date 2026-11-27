# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage and duplicate checks for the author-style per-game aggregate."""

import pytest

from benchmarks.vs_bench.prepare import GAMES
from responses_api_agents.vs_bench_reference.app import VSBenchReferenceAgent


def sample(game, index, reward, track="reasoning"):
    return [
        {
            "verifier_metadata": {"track": track, "game": game, "row_index": index},
            "reward": reward,
            "author_source_variant": "kaz_numpy_labels_v1",
        }
    ]


def test_macro_is_available_only_for_all_unique_questions():
    tasks = [sample(game, index, index % 2) for game in GAMES for index in range(400)]
    metrics = VSBenchReferenceAgent.compute_metrics(None, tasks)
    assert metrics["vs_bench/reasoning/macro_accuracy"] == 0.5
    assert metrics["vs_bench/reasoning/full_coverage"] is True
    partial = VSBenchReferenceAgent.compute_metrics(None, tasks[:-1])
    assert partial["vs_bench/reasoning/full_coverage"] is False
    assert "vs_bench/reasoning/macro_accuracy" not in partial
    with pytest.raises(ValueError, match="Duplicate"):
        VSBenchReferenceAgent.compute_metrics(None, tasks + tasks[:1])


def test_perception_macro_weights_games_equally_despite_smaller_kaz_split():
    tasks = [
        sample(game, index, float(game == "knights_archers_zombies"), "perception")
        for game in GAMES
        for index in range(300 if game == "knights_archers_zombies" else 400)
    ]
    metrics = VSBenchReferenceAgent.compute_metrics(None, tasks)
    assert metrics["vs_bench/perception/macro_score"] == 0.1
    assert metrics["vs_bench/perception/knights_archers_zombies/count"] == 300


def test_aggregate_rejects_mixed_source_variants():
    first = sample("hanabi", 0, 1, "perception")
    second = sample("hanabi", 1, 1, "perception")
    second[0]["author_source_variant"] = "upstream"
    with pytest.raises(ValueError, match="source variant"):
        VSBenchReferenceAgent.compute_metrics(None, [first, second])
