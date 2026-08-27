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
"""Expand a sweep manifest into Gym's materialized-inputs file, in parallel.

Rollout collection expands `num_repeats` itself, single-threaded, at roughly 300 source rows/s --
about 100 minutes for a full sweep on a 144-core node, paid on every launch. Doing it here instead
parallelizes across entries and lands the result exactly where Gym's resume path looks for it, so
subsequent runs skip preprocessing entirely.

Identity is assigned deterministically from manifest order and within-file line order, so
regenerating on a different node reproduces the same `(_ng_task_index, _ng_rollout_index)` keys and
`--resume` still matches rollouts completed by an earlier job.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import orjson
import yaml

from nemo_gym.sweep.manifest import AGENT_REF_KEY, SweepManifest, SweepValidationError
from nemo_gym.sweep.shuffle import DEFAULT_BUFFER_ROWS, streaming_shuffle


TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"
# Which manifest entry a row came from. Rows already carry agent_ref, but entries routinely share
# an agent -- math_tir, stem_mcqa_tools_ultra_0 and stem_openqa_tools_ultra_0 all dispatch to
# ns_tools_simple_agent -- so agent_ref cannot identify the entry.
#
# This is reliable on the materialized inputs and NOT on the rollouts. Gym preserves the reserved
# _ng_task_index and _ng_rollout_index through collection, but an arbitrary key survives only if
# that agent copies the input row rather than rebuilding it. Measured on job 6564684: 376 of 2,468
# rollouts kept it, all from ns_tools_simple_agent and inverse_if_simple_agent; the other 34 agents
# dropped it. So treat it as a convenience on the inputs, and use task_index_range as the ground
# truth for attributing rollouts -- which is what split does.
SWEEP_LABEL_KEY = "_ng_sweep_label"


REPORT_NAME = "sweep_report.json"
CONFIG_NAME = "sweep_config.yaml"


@dataclass
class MaterializeReport:
    """Observed counts for a materialized sweep.

    The input manifest declares intent; this records what the data actually contained. Written
    beside the artifacts so downstream cost estimates and blend accounting read a measured file
    rather than hand-maintained numbers.
    """

    materialized_fpath: Path
    output_fpath: Path
    report_fpath: Path
    config_fpath: Path
    rows_per_entry: Dict[str, int]
    materialized_per_entry: Dict[str, int]
    num_repeats_per_entry: Dict[str, int]
    # label -> [first task index, last task index inclusive]. Rollout results carry only
    # (_ng_task_index, _ng_rollout_index, agent_ref), so this table is how a rollout is traced
    # back to the entry and dataset it came from -- especially when several entries share an agent.
    task_index_ranges: Dict[str, Tuple[int, int]]
    nickname: str = ""
    shuffle_seed: int = 0

    @property
    def total_source_rows(self) -> int:
        return sum(self.rows_per_entry.values())

    @property
    def total_materialized_rows(self) -> int:
        return sum(self.materialized_per_entry.values())

    def to_dict(self) -> Dict:
        return {
            "materialized_fpath": str(self.materialized_fpath),
            "output_fpath": str(self.output_fpath),
            "config_fpath": str(self.config_fpath),
            "materialized_bytes": self.materialized_fpath.stat().st_size if self.materialized_fpath.exists() else None,
            "nickname": self.nickname,
            "shuffle_seed": self.shuffle_seed,
            "total_source_rows": self.total_source_rows,
            "total_materialized_rows": self.total_materialized_rows,
            "entries": {
                label: {
                    "source_rows": self.rows_per_entry[label],
                    "materialized_rows": self.materialized_per_entry[label],
                    "num_repeats": self.num_repeats_per_entry[label],
                    "task_index_range": list(self.task_index_ranges[label]),
                }
                for label in self.rows_per_entry
            },
        }

    def write(self) -> Path:
        with open(self.report_fpath, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return self.report_fpath


def _count_rows(path: str, limit: Optional[int]) -> int:
    n = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                n += 1
                if limit is not None and n >= limit:
                    break
    return n


def _expand_entry(args: Tuple[str, str, str, Optional[str], int, int, Optional[int]]) -> Tuple[str, int, int]:
    """Expand one entry into its own part file. Runs in a worker process."""
    label, data, part_path, override, repeats, task_offset, limit = args
    src_rows = 0
    written = 0
    with open(data, "rb") as source, open(part_path, "wb") as sink:
        for line in source:
            if not line.strip():
                continue
            if limit is not None and src_rows >= limit:
                break
            row = orjson.loads(line)
            ref = row.get(AGENT_REF_KEY)
            if not isinstance(ref, dict):
                raise SweepValidationError(f"[{label}] row {src_rows} has no agent_ref; cannot route it")
            if override:
                ref["name"] = override
            row[SWEEP_LABEL_KEY] = label
            row[TASK_INDEX_KEY] = task_offset + src_rows
            for rollout_index in range(repeats):
                row[ROLLOUT_INDEX_KEY] = rollout_index
                sink.write(orjson.dumps(row) + b"\n")
                written += 1
            src_rows += 1
    return label, src_rows, written


def materialize(
    manifest: SweepManifest,
    out_dir: str | Path,
    *,
    jobs: Optional[int] = None,
    limit_per_entry: Optional[int] = None,
    overwrite: bool = False,
    shuffle_seed: int = 0,
    shuffle_buffer_rows: int = DEFAULT_BUFFER_ROWS,
) -> MaterializeReport:
    out_dir = Path(out_dir) / manifest.nickname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gym derives the materialized path from the output path, so write to exactly that name and
    # leave an empty rollouts file beside it: both must exist for --resume to take the fast path.
    output_fpath = out_dir / "rollouts.jsonl"
    materialized_fpath = output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")
    if materialized_fpath.exists() and not overwrite:
        raise SweepValidationError(f"{materialized_fpath} already exists. Pass overwrite=True to replace it.")

    repeats_by_entry = {
        entry.label: (entry.num_repeats if entry.num_repeats is not None else manifest.defaults.num_repeats)
        for entry in manifest.entries
    }

    # Task indices are laid out as contiguous per-entry ranges in manifest order, so the assignment
    # depends only on the manifest and the data -- never on worker scheduling.
    print(f"counting rows across {len(manifest.entries)} entries...", flush=True)
    counts = [_count_rows(entry.data, limit_per_entry) for entry in manifest.entries]
    offsets: List[int] = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count

    parts_dir = out_dir / "_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()

    tasks = [
        (
            entry.label,
            entry.data,
            str(parts_dir / f"{index:04d}_{entry.label}.jsonl"),
            entry.agent_ref_override,
            repeats_by_entry[entry.label],
            offsets[index],
            limit_per_entry,
        )
        for index, entry in enumerate(manifest.entries)
    ]

    workers = jobs or min(len(tasks), os.cpu_count() or 1)
    print(f"expanding {running:,} source rows with {workers} workers...", flush=True)
    rows_per_entry: Dict[str, int] = {}
    materialized_per_entry: Dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for label, src_rows, written in pool.map(_expand_entry, tasks):
            rows_per_entry[label] = src_rows
            materialized_per_entry[label] = written
            print(f"  {label:28} {src_rows:8,} -> {written:9,}", flush=True)

    def _parts_in_order():
        for _, _, part_path, *_ in tasks:
            with open(part_path, "rb") as part:
                yield from part

    if shuffle_seed > 0:
        # Task identity was already fixed above, so reordering here changes only dispatch order.
        print(f"concatenating with seeded shuffle (seed={shuffle_seed})...", flush=True)
        stream = streaming_shuffle(_parts_in_order(), shuffle_seed, shuffle_buffer_rows)
    else:
        print("concatenating parts in manifest order (shuffle disabled)...", flush=True)
        stream = _parts_in_order()
    with open(materialized_fpath, "wb") as sink:
        sink.writelines(stream)
    shutil.rmtree(parts_dir)

    # An empty rollouts file is the second half of the resume gate.
    output_fpath.touch(exist_ok=True)

    # The launchers serve this deployment from SWEEP_DIR, so the composed config has to land
    # beside the inputs; otherwise SWEEP_DIR is not self-contained and `gym env start` has nothing
    # to read.
    config_fpath = out_dir / CONFIG_NAME
    with open(config_fpath, "w") as handle:
        yaml.safe_dump(
            {"config_paths": manifest.config_paths(), **manifest.config_overlay},
            handle,
            default_flow_style=False,
            sort_keys=False,
        )

    report = MaterializeReport(
        materialized_fpath=materialized_fpath,
        output_fpath=output_fpath,
        report_fpath=out_dir / REPORT_NAME,
        config_fpath=config_fpath,
        task_index_ranges={
            entry.label: (offsets[i], offsets[i] + counts[i] - 1) for i, entry in enumerate(manifest.entries)
        },
        nickname=manifest.nickname,
        shuffle_seed=shuffle_seed,
        rows_per_entry=rows_per_entry,
        materialized_per_entry=materialized_per_entry,
        num_repeats_per_entry=repeats_by_entry,
    )
    report.write()
    return report
