# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Split a materialized sweep across several jobs, and merge their rollouts back.

One job cannot use 256 nodes: ``--segment`` needs a topology-contiguous allocation and an NVL72
rack is 18 nodes, and a single driver at ``512 x decode_nodes`` concurrency would exceed the
aiohttp per-host connector limit long before the GPUs saturated. Several identical jobs over
disjoint slices of the same input is the shape that scales.

This is safe because ``materialize`` stamps ``_ng_task_index`` and ``_ng_sweep_label`` **before**
any sharding, and Gym never rewrites either -- only ``global_config`` names the index, and nothing
in rollout collection assigns it. Indices therefore stay globally unique across shards, so shard
rollouts concatenate without renumbering and ``split`` gives the same answer on the merged file as
on an unsharded run.

Rows are dealt round-robin rather than sliced into contiguous blocks. A contiguous slice would hand
one shard an entire slow environment -- ``lean`` and ``math_cot`` dominate the tail -- while other
shards finish early and idle.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

INPUTS_NAME = "rollouts_materialized_inputs.jsonl"
ROLLOUTS_NAME = "rollouts.jsonl"
CONFIG_NAME = "sweep_config.yaml"
REPORT_NAME = "sweep_report.json"
SHARD_REPORT_NAME = "shard_report.json"
TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"


class SweepShardError(RuntimeError):
    """Raised when a sweep cannot be sharded or merged."""


@dataclass
class ShardResult:
    shard_dirs: List[Path] = field(default_factory=list)
    rows_per_shard: List[int] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_per_shard)


@dataclass
class MergeResult:
    output_fpath: Path
    merged: int = 0
    duplicates: int = 0
    shards_seen: int = 0
    shards_empty: List[str] = field(default_factory=list)


def shard_sweep(sweep_dir: str | Path, num_shards: int, out_dir: Optional[str | Path] = None) -> ShardResult:
    """Deal a materialized input into ``num_shards`` sibling sweep directories.

    Each shard directory is itself a valid SWEEP_DIR -- it carries the same ``sweep_config.yaml``
    and ``sweep_report.json`` and an empty ``rollouts.jsonl`` -- so the launcher runs against one
    with no special handling. The report is copied unchanged because ``task_index_range`` refers to
    global indices, which sharding does not disturb.
    """
    sweep_dir = Path(sweep_dir)
    if num_shards < 1:
        raise SweepShardError(f"num_shards must be at least 1, got {num_shards}")

    inputs = sweep_dir / INPUTS_NAME
    if not inputs.is_file():
        raise SweepShardError(f"No materialized inputs at {inputs}; run `nemo_gym.sweep materialize` first.")

    out_dir = Path(out_dir) if out_dir is not None else sweep_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [out_dir / f"shard_{i:03d}" for i in range(num_shards)]
    for d in shard_dirs:
        d.mkdir(parents=True, exist_ok=True)

    counts = [0] * num_shards
    handles = [open(d / INPUTS_NAME, "w") for d in shard_dirs]
    try:
        with open(inputs) as reader:
            for position, line in enumerate(reader):
                if not line.strip():
                    continue
                index = position % num_shards
                handles[index].write(line if line.endswith("\n") else line + "\n")
                counts[index] += 1
    finally:
        for handle in handles:
            handle.close()

    for d in shard_dirs:
        for name in (CONFIG_NAME, REPORT_NAME):
            source = sweep_dir / name
            if source.is_file():
                shutil.copy2(source, d / name)
        # Completes the --resume gate, exactly as materialize does for an unsharded sweep.
        (d / ROLLOUTS_NAME).touch()

    (out_dir / SHARD_REPORT_NAME).write_text(
        json.dumps(
            {
                "source_sweep_dir": str(sweep_dir),
                "num_shards": num_shards,
                "strategy": "round_robin",
                "rows_per_shard": {d.name: c for d, c in zip(shard_dirs, counts)},
                "total_rows": sum(counts),
            },
            indent=2,
        )
        + "\n"
    )
    return ShardResult(shard_dirs=shard_dirs, rows_per_shard=counts)


def merge_shards(shards_dir: str | Path, output_fpath: Optional[str | Path] = None) -> MergeResult:
    """Concatenate every shard's rollouts into one file, dropping duplicates.

    Deduplicates on ``(_ng_task_index, _ng_rollout_index)``, which is the same key Gym resumes on.
    Re-running a shard after a walltime kill therefore cannot double-count, and neither can merging
    a directory that was resharded and partially rerun.
    """
    shards_dir = Path(shards_dir)
    shard_dirs = sorted(d for d in shards_dir.glob("shard_*") if d.is_dir())
    if not shard_dirs:
        raise SweepShardError(f"No shard_* directories under {shards_dir}")

    output_fpath = Path(output_fpath) if output_fpath is not None else shards_dir / ROLLOUTS_NAME
    seen: Set[Tuple[object, object]] = set()
    result = MergeResult(output_fpath=output_fpath, shards_seen=len(shard_dirs))

    with open(output_fpath, "w") as sink:
        for d in shard_dirs:
            src = d / ROLLOUTS_NAME
            rows_here = 0
            if not src.is_file():
                result.shards_empty.append(d.name)
                continue
            with open(src) as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    key = (row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY))
                    if key in seen:
                        result.duplicates += 1
                        continue
                    seen.add(key)
                    sink.write(line + "\n")
                    result.merged += 1
                    rows_here += 1
            if rows_here == 0:
                result.shards_empty.append(d.name)

    return result


def reshard(sweep_dir: str | Path, num_shards: int, out_dir: Optional[str | Path] = None) -> ShardResult:
    """Re-deal the original materialized input into a different shard count.

    Always works from the parent sweep's input rather than from the existing shards, so changing
    the count is not lossy and does not compound rounding. Any rollouts already collected stay
    valid: they are keyed by global task index, so merging after a reshard still deduplicates
    correctly against work done under the previous layout.
    """
    return shard_sweep(sweep_dir, num_shards, out_dir=out_dir)
