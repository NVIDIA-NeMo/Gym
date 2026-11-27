# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.vs_bench.prepare import GAMES
from benchmarks.vs_bench_gameplay_suites_shard.prepare import prepare


def test_suites_partition_noncompetitive_games_without_duplicates(tmp_path):
    games = []
    for shard in range(4):
        path = prepare(tmp_path / f"{shard}.jsonl", shard=shard)
        rows = [json.loads(line)["verifier_metadata"] for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert all(row["protocol"] == "paper_competitive_v1" for row in rows)
        games.extend(row["game"] for row in rows)
    assert len(set(games)) == 8
    assert set(games) == set(GAMES) - {"kuhn_poker", "breakthrough"}


def test_invalid_suite_partition_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="shard"):
        prepare(tmp_path / "invalid.jsonl", shard=4)
