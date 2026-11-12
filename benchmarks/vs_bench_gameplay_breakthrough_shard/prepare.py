# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create one seat's disjoint ten-game Breakthrough paper schedule."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes
from benchmarks.vs_bench.prepare import AUTHOR_REVISION


OUTPUT = Path("benchmarks/vs_bench_gameplay_breakthrough_shard/data/vs_bench_breakthrough_shard.jsonl")


def prepare(output_fpath=OUTPUT, seat=None):
    seat = str(seat if seat is not None else os.environ["VS_BENCH_BREAKTHROUGH_SEAT"])
    if seat not in ("0", "1"):
        raise ValueError("Breakthrough seat must be 0 or 1")
    target = Path(output_fpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for episode in paper_competitive_episodes("breakthrough"):
            if episode.model_seat != int(seat):
                continue
            metadata = {
                "track": "gameplay",
                "game": episode.game,
                "author_revision": AUTHOR_REVISION,
                "protocol": "paper_competitive_v1",
                "paper_episode": asdict(episode),
            }
            stream.write(
                json.dumps({"responses_create_params": {"input": "VS-Bench gameplay"}, "verifier_metadata": metadata})
                + "\n"
            )
    return target
