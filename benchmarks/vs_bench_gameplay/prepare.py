# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create one task per original VS-Bench game configuration."""

import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes
from benchmarks.vs_bench.prepare import AUTHOR_REVISION, GAMES


OUTPUT = Path("benchmarks/vs_bench_gameplay/data/vs_bench_gameplay.jsonl")


def prepare(output_fpath=OUTPUT, games=GAMES):
    output_fpath = Path(output_fpath)
    games = tuple(games)
    if not games or len(set(games)) != len(games) or any(game not in GAMES for game in games):
        raise ValueError("Invalid gameplay game selection")
    if output_fpath == OUTPUT and games != GAMES:
        raise ValueError("A partial selection needs a distinct output path")
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_fpath.with_suffix(output_fpath.suffix + ".partial")
    if output_fpath.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output_fpath}")
    with temporary.open("x", encoding="utf-8") as stream:
        for game in games:
            episodes = paper_competitive_episodes(game) if game in ("kuhn_poker", "breakthrough") else (None,)
            for episode in episodes:
                metadata = {
                    "track": "gameplay",
                    "game": game,
                    "author_revision": AUTHOR_REVISION,
                    "protocol": "paper_competitive_v1",
                }
                if episode is not None:
                    metadata["paper_episode"] = asdict(episode)
                record = {"responses_create_params": {"input": "VS-Bench gameplay"}, "verifier_metadata": metadata}
                stream.write(json.dumps(record) + "\n")
    temporary.rename(output_fpath)
    return output_fpath
