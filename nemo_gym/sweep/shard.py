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

This is safe because ``materialize`` stamps ``_ng_task_index`` **before** any sharding and Gym
preserves it through collection -- it is a reserved key, present on 100% of rollouts measured.
Indices therefore stay globally unique across shards, so shard rollouts concatenate without
renumbering and ``split`` gives the same answer on the merged file as on an unsharded run.
(``_ng_sweep_label`` is stamped too but most agents drop it; see materialize.)

Rows are dealt round-robin rather than sliced into contiguous blocks. A contiguous slice would hand
one shard an entire slow environment -- ``lean`` and ``math_cot`` dominate the tail -- while other
shards finish early and idle.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


INPUTS_NAME = "rollouts_materialized_inputs.jsonl"
ROLLOUTS_NAME = "rollouts.jsonl"
CONFIG_NAME = "sweep_config.yaml"
REPORT_NAME = "sweep_report.json"
SHARD_REPORT_NAME = "shard_report.json"
SNAPSHOTS_DIR = "snapshots"
TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"


class SweepShardError(RuntimeError):
    """Raised when a sweep cannot be sharded or merged."""


@dataclass
class ShardResult:
    shard_dirs: List[Path] = field(default_factory=list)
    rows_per_shard: List[int] = field(default_factory=list)
    carried_rollouts: int = 0
    removed_stale: List[str] = field(default_factory=list)
    absorbed_rollouts: int = 0
    snapshot_dir: Optional[Path] = None

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

    Each shard directory is itself a valid SWEEP_DIR -- same ``sweep_config.yaml`` and
    ``sweep_report.json``, plus a ``rollouts.jsonl`` -- so the launcher runs against one with no
    special handling. The report is copied unchanged because ``task_index_range`` refers to global
    indices, which sharding does not disturb.

    If the parent sweep has already collected rollouts, they are routed into whichever shard now
    owns each row, so resharding a half-finished run resumes rather than recollecting.
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

    # Anything the previous layout collected is folded back into the parent first, and the parent
    # is snapshotted, before a single shard directory is touched. Resharding is destructive --
    # shard directories are rewritten and extras removed -- so the only safe order is to make the
    # parent authoritative first. Without this, resharding a run whose shards were never merged
    # would delete the only copy of that work.
    absorbed = 0
    snapshot_dir: Optional[Path] = None
    if any(out_dir.glob("shard_*/" + ROLLOUTS_NAME)):
        merge_shards(out_dir, output_fpath=out_dir / "_premerge.jsonl")
        parent_rollouts = sweep_dir / ROLLOUTS_NAME
        seen_parent: Set[Tuple[object, object]] = set()
        if parent_rollouts.is_file():
            with open(parent_rollouts) as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # a job killed mid-write leaves a torn final line
                    seen_parent.add((row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY)))
        with open(parent_rollouts, "a") as sink, open(out_dir / "_premerge.jsonl") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY)) in seen_parent:
                    continue
                sink.write(line + "\n")
                absorbed += 1
        (out_dir / "_premerge.jsonl").unlink()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_dir = sweep_dir / SNAPSHOTS_DIR / stamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if parent_rollouts.is_file():
            shutil.copy2(parent_rollouts, snapshot_dir / ROLLOUTS_NAME)
        for name in (REPORT_NAME, SHARD_REPORT_NAME):
            src = sweep_dir / name if name == REPORT_NAME else out_dir / name
            if src.is_file():
                shutil.copy2(src, snapshot_dir / name)

    # Resharding to fewer shards would otherwise strand the extras. They still hold rows and
    # rollouts from the old layout, so merge would union them back in (dedup saves correctness,
    # but only by accident) and a job launched against one would redo work that now belongs
    # elsewhere. Their rollouts are already carried into the new layout below, so dropping them
    # loses nothing.
    stale = sorted(d for d in out_dir.glob("shard_*") if d.is_dir() and d not in set(shard_dirs))
    for d in stale:
        shutil.rmtree(d)

    counts = [0] * num_shards
    owner_of_key: Dict[Tuple[object, object], int] = {}
    handles = [open(d / INPUTS_NAME, "w") for d in shard_dirs]
    try:
        with open(inputs) as reader:
            for position, line in enumerate(reader):
                if not line.strip():
                    continue
                index = position % num_shards
                handles[index].write(line if line.endswith("\n") else line + "\n")
                counts[index] += 1
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                # Repeats of one task are dealt to different shards, so a rollout is routed by
                # the exact (task, rollout) pair -- the task index alone does not identify a shard.
                owner_of_key[(row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY))] = index
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

    # Carry any work the parent sweep already collected into the shard that now owns those rows.
    # Without this, resharding a half-finished run silently discards every rollout: each shard
    # starts with an empty rollouts.jsonl, so --resume finds nothing and recollects from scratch.
    carried = _carry_existing_rollouts(sweep_dir, shard_dirs, owner_of_key)

    result = ShardResult(
        shard_dirs=shard_dirs,
        rows_per_shard=counts,
        carried_rollouts=carried,
        removed_stale=[d.name for d in stale],
        absorbed_rollouts=absorbed,
        snapshot_dir=snapshot_dir,
    )
    (out_dir / SHARD_REPORT_NAME).write_text(
        json.dumps(
            {
                "source_sweep_dir": str(sweep_dir),
                "num_shards": num_shards,
                "strategy": "round_robin",
                "rows_per_shard": {d.name: c for d, c in zip(shard_dirs, counts)},
                "total_rows": sum(counts),
                "carried_rollouts": carried,
                "removed_stale": [d.name for d in stale],
                "absorbed_rollouts": absorbed,
                "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
            },
            indent=2,
        )
        + "\n"
    )
    return result


def _carry_existing_rollouts(
    sweep_dir: Path, shard_dirs: List[Path], owner_of_key: Dict[Tuple[object, object], int]
) -> int:
    """Route rollouts the parent already has into whichever shard now owns each row.

    Keyed on (task, rollout), the same pair Gym resumes on, so a shard sees exactly the work
    already done for its own rows and recollects only the remainder.
    """
    existing = sweep_dir / ROLLOUTS_NAME
    if not existing.is_file() or existing.stat().st_size == 0:
        return 0

    carried = 0
    handles = [open(d / ROLLOUTS_NAME, "w") for d in shard_dirs]
    try:
        with open(existing) as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                index = owner_of_key.get((row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY)))
                if index is None:
                    index = owner_of_key.get((row.get(TASK_INDEX_KEY), None))
                if index is None:
                    # A rollout whose input is not in this sweep; dropping it is correct, and
                    # merge would have deduplicated it against nothing anyway.
                    continue
                handles[index].write(line + "\n")
                carried += 1
    finally:
        for handle in handles:
            handle.close()
    return carried


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
            saw_any = False
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
                    saw_any = True
                    key = (row.get(TASK_INDEX_KEY), row.get(ROLLOUT_INDEX_KEY))
                    if key in seen:
                        result.duplicates += 1
                        continue
                    seen.add(key)
                    sink.write(line + "\n")
                    result.merged += 1
                    rows_here += 1
            if rows_here == 0 and not saw_any:
                result.shards_empty.append(d.name)

    return result
