# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the AgentArena benchmark split.

The benchmark is the 433-turn Arena validation split (arena_rl_val.v2.jsonl), held out from the
Arena RL training rows. It is NVIDIA internal use only and not in git: stage the file where the
run can read it and pass its path as `source_jsonl` (prepare_script_args) or
AGENT_ARENA_TURNS_JSONL. Preparation copies it into place and fails when no source is given, so a
run never scores a partial or stale set.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from nemo_gym.server_runtime import BaseRunRequest


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "agent_arena_benchmark.jsonl"
SOURCE_ENV = "AGENT_ARENA_TURNS_JSONL"


def prepare(source_jsonl: str | None = None, limit: int | None = None) -> Path:
    source = source_jsonl or os.environ.get(SOURCE_ENV)
    if not source:
        raise ValueError(
            f"AgentArena data is not in git: pass prepare_script_args.source_jsonl or set {SOURCE_ENV} "
            "to the staged Arena turns file (see resources_servers/agentic_turn_simulation/docs/data.md)."
        )
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"AgentArena turns file not found: {source_path}")

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    # Publish only what validated: a prepare that left a broken file behind would be taken for a
    # prepared benchmark, since the cache check only looks for the path.
    with TemporaryDirectory(dir=OUTPUT_FPATH.parent) as staging_dir:
        staged = Path(staging_dir) / OUTPUT_FPATH.name
        if limit is None:
            shutil.copyfile(source_path, staged)
        else:
            with source_path.open(encoding="utf-8") as src, staged.open("w", encoding="utf-8") as out:
                for i, line in enumerate(src):
                    if i >= limit:
                        break
                    out.write(line)
        rows = _validate(staged, source_path)
        staged.replace(OUTPUT_FPATH)
    print(f"Wrote {len(rows)} AgentArena turns to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


def _validate(staged: Path, source_path: Path) -> list[dict]:
    """Every physical line of the copy is a request the environment can run."""
    # Not `if line.strip()`: the copy keeps the source's bytes, and collation reads each of its
    # lines, so a blank line it skipped here would stop the run instead of the preparation.
    with staged.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise ValueError(f"{source_path} has no rows")
    for number, row in enumerate(rows, 1):
        try:
            BaseRunRequest.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"{source_path}:{number}: {exc}") from exc
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the AgentArena benchmark split")
    parser.add_argument("--source-jsonl", default=None, help=f"Staged Arena turns file (default: ${SOURCE_ENV})")
    parser.add_argument("--limit", type=int, default=None, help="Max rows, for a smoke subset")
    args = parser.parse_args()
    prepare(source_jsonl=args.source_jsonl, limit=args.limit)
