# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned VS-Bench reasoning split for Gym."""

from pathlib import Path

from benchmarks.vs_bench.prepare import GAMES
from benchmarks.vs_bench.prepare import prepare as prepare_offline


OUTPUT = Path("benchmarks/vs_bench_reasoning/data/vs_bench_reasoning.jsonl")


def prepare(dataset_root=None, output_fpath=OUTPUT, limit=400, games=GAMES, reasoning_model=True):
    output_fpath = Path(output_fpath)
    if output_fpath == OUTPUT and (limit != 400 or tuple(games) != GAMES or not reasoning_model):
        raise ValueError("Partial or alternative preparation requires a separate output path")
    return prepare_offline(
        dataset_root=Path(dataset_root) if dataset_root else None,
        output_fpath=output_fpath,
        tracks=("reasoning",),
        games=tuple(games),
        limit=limit,
        reasoning_model=reasoning_model,
    )
