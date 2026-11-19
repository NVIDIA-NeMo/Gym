# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gym rows (JSONL) from an ARC-AGI-3 pivot pool split (``train.parquet`` / ``eval.parquet``).

One row per pivot. ``verifier_metadata.pivot`` carries the pool row's columns the environment needs (``pivot_json``, the
row's harness revision and its recorded harness env file); the paths inside them stay relative to the pool directory,
which the server config names (``pool_dir``). Only rows with a recording that the pool builder restored are kept:
``restore_mode`` snapshot with ``restore_ok``.

    python resources_servers/arc_agi_3_pivot/create_dataset.py <pool_dir>/eval.parquet \
        --output resources_servers/arc_agi_3_pivot/data/eval.jsonl [--limit N] [--per-game N] [--pivot-ids a,b] \
        [--reasoning-effort low]

``--reasoning-effort`` sets each row's ``verifier_metadata.reasoning_effort``: the thinking effort of the window's live
requests (the server's ``reasoning_effort`` otherwise, then the pivot's recorded one).
"""

import argparse
import json
from collections import Counter
from typing import Any


# thinking_effort (optional pool column): the effort the pivot's run was sampled with when the collector set it past the
# harness's request; the window's live requests default to it.
PIVOT_KEYS = ("pivot_json", "harness", "harness_env", "harness_env_sha256", "thinking_effort")


def gym_row(row: dict[str, Any], reasoning_effort: str | None = None) -> dict[str, Any]:
    out = {
        "responses_create_params": {
            "input": [
                {"role": "user", "content": f"Continue ARC-AGI-3 game {row['game_id']} from pivot {row['pivot_id']}."}
            ]
        },
        "verifier_metadata": {
            "pivot_id": row["pivot_id"],
            "game_id": row["game_id"],
            "level": row.get("level"),
            "po_d0": row.get("po_d0"),
            "pivot": {k: row[k] for k in PIVOT_KEYS if row.get(k) is not None},
        },
    }
    if reasoning_effort:
        out["verifier_metadata"]["reasoning_effort"] = reasoning_effort
    return out


def select(rows, limit: int | None, per_game: int | None, ids: set | None) -> list[dict[str, Any]]:
    """Restorable rows, in order, at most ``per_game`` per game and ``limit`` in all."""
    out, per = [], Counter()
    for row in rows:
        if row.get("restore_mode") != "snapshot" or not row.get("restore_ok"):
            continue
        if ids is not None and row["pivot_id"] not in ids:
            continue
        if per_game is not None and per[row["game_id"]] >= per_game:
            continue
        per[row["game_id"]] += 1
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def iter_rows(path: str):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):  # pools are large: never the whole split in memory
        yield from batch.to_pylist()


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", help="a pivot pool split parquet (train.parquet / eval.parquet)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-game", type=int)
    parser.add_argument("--pivot-ids", help="comma-separated pivot ids to keep")
    parser.add_argument("--reasoning-effort", help="the live requests' thinking effort, e.g. low / high / max")
    args = parser.parse_args(argv)

    ids = set(args.pivot_ids.split(",")) if args.pivot_ids else None
    rows = select(iter_rows(args.split), args.limit, args.per_game, ids)
    with open(args.output, "w") as fh:
        for row in rows:
            fh.write(json.dumps(gym_row(row, args.reasoning_effort)) + "\n")
    print(f"{len(rows)} rows from {len({r['game_id'] for r in rows})} games -> {args.output}")


if __name__ == "__main__":
    main()
