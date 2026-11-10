# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare disjoint noncompetitive gameplay suites for parallel execution."""

import os
from pathlib import Path

from benchmarks.vs_bench_gameplay.prepare import prepare as prepare_gameplay


OUTPUT = Path("benchmarks/vs_bench_gameplay_suites_shard/data/vs_bench_suites_shard.jsonl")
SHARDS = (
    ("hanabi", "overcooked"),
    ("knights_archers_zombies", "atari_pong"),
    ("simple_push", "coin_dilemma"),
    ("monster_hunt", "battle_of_the_colors"),
)


def prepare(output_fpath=OUTPUT, shard=None):
    shard = str(shard if shard is not None else os.environ["VS_BENCH_SUITES_SHARD"])
    if shard not in ("0", "1", "2", "3"):
        raise ValueError("Gameplay suites shard must be 0, 1, 2 or 3")
    return prepare_gameplay(output_fpath, games=SHARDS[int(shard)])
