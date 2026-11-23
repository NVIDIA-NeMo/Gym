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
"""Split a sweep's inputs and rollouts back into one directory per manifest entry.

A sweep collapses many environments into one input file so they share a deployment. Reporting
wants the opposite: per-entry files, named for the entry, that `gym eval profile` can run over
individually.

``agent_ref`` cannot do this. Entries routinely share an agent -- the three ``ns_tools`` entries in
the reward-profiling manifest all dispatch to ``ns_tools_simple_agent`` -- so splitting on it would
merge them.

``_ng_task_index`` against the ``task_index_range`` recorded per entry is the ground truth, and is
what attributes almost every rollout. ``_ng_sweep_label`` is checked first only as a shortcut: it
is stamped on every materialized input, but survives collection for just the handful of agents that
copy the input row rather than rebuilding it (376 of 2,468 rollouts on job 6564684). Both routes
resolve to the same entry, so the fallback is not a degradation.
"""

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INPUTS_NAME = "rollouts_materialized_inputs.jsonl"
ROLLOUTS_NAME = "rollouts.jsonl"
REPORT_NAME = "sweep_report.json"
SPLIT_REPORT_NAME = "split_report.json"
TASK_INDEX_KEY = "_ng_task_index"
SWEEP_LABEL_KEY = "_ng_sweep_label"


class SweepSplitError(RuntimeError):
    """Raised when a sweep directory cannot be split."""


@dataclass
class SplitCounts:
    """Per-label tallies, so a label that collected nothing is visible rather than absent."""

    inputs: int = 0
    rollouts: int = 0


@dataclass
class SplitResult:
    out_dir: Path
    counts: Dict[str, SplitCounts] = field(default_factory=dict)
    unmapped_inputs: int = 0
    unmapped_rollouts: int = 0

    @property
    def labels_without_rollouts(self) -> List[str]:
        return sorted(label for label, c in self.counts.items() if c.rollouts == 0)


def _load_ranges(report_fpath: Path) -> List[Tuple[int, int, str]]:
    """Return (low, high, label) sorted by low, from a materialize report."""
    try:
        report = json.loads(report_fpath.read_text())
    except (OSError, ValueError) as exc:
        raise SweepSplitError(f"Could not read {report_fpath}: {exc}") from exc

    entries = report.get("entries") or {}
    if not isinstance(entries, dict):
        raise SweepSplitError(f"{report_fpath} has no per-entry mapping; it may predate task_index_range.")

    ranges: List[Tuple[int, int, str]] = []
    for label, entry in entries.items():
        span = (entry or {}).get("task_index_range")
        if not span or len(span) != 2:
            raise SweepSplitError(f"Entry '{label}' in {report_fpath} has no task_index_range.")
        ranges.append((int(span[0]), int(span[1]), label))

    if not ranges:
        raise SweepSplitError(
            f"{report_fpath} lists no entries, so there is nothing to split. This is usually the "
            f"wrong directory: split takes the <out-dir>/<nickname> directory materialize wrote, "
            f"or a single shards/shard_NNN, not <out-dir> or shards/ itself."
        )

    ranges.sort()
    for (lo, hi, label), (next_lo, _, next_label) in zip(ranges, ranges[1:]):
        if hi >= next_lo:
            raise SweepSplitError(
                f"task_index_range overlap between '{label}' [{lo},{hi}] and '{next_label}' "
                f"starting at {next_lo}; the report is inconsistent."
            )
    return ranges


def _lookup(ranges: List[Tuple[int, int, str]], lows: List[int], task_index: int) -> Optional[str]:
    position = bisect_right(lows, task_index) - 1
    if position < 0:
        return None
    lo, hi, label = ranges[position]
    return label if lo <= task_index <= hi else None


def _split_file(
    src: Path,
    ranges: List[Tuple[int, int, str]],
    lows: List[int],
    out_dir: Path,
    filename: str,
    counts: Dict[str, SplitCounts],
    attribute: str,
) -> int:
    """Stream ``src`` into ``out_dir/<label>/<filename>``. Returns the unmapped row count.

    Rows are streamed and handles kept open per label rather than buffering: a full sweep's
    rollouts run to tens of gigabytes.
    """
    if not src.is_file():
        return 0

    handles: Dict[str, object] = {}
    unmapped = 0
    try:
        with open(src) as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    unmapped += 1
                    continue
                # Label first as a shortcut, but most agents drop it during collection, so the
                # index range does the real work here.
                label = row.get(SWEEP_LABEL_KEY)
                if label not in counts:
                    task_index = row.get(TASK_INDEX_KEY)
                    label = _lookup(ranges, lows, task_index) if isinstance(task_index, int) else None
                if label is None:
                    unmapped += 1
                    continue
                handle = handles.get(label)
                if handle is None:
                    target = out_dir / label
                    target.mkdir(parents=True, exist_ok=True)
                    handle = handles[label] = open(target / filename, "w")
                handle.write(line + "\n")
                setattr(counts[label], attribute, getattr(counts[label], attribute) + 1)
    finally:
        for handle in handles.values():
            handle.close()
    return unmapped


def split_sweep(sweep_dir: str | Path, out_dir: Optional[str | Path] = None) -> SplitResult:
    """Write one directory per manifest entry, each holding that entry's inputs and rollouts."""
    sweep_dir = Path(sweep_dir)
    out_dir = Path(out_dir) if out_dir is not None else sweep_dir / "by_label"

    ranges = _load_ranges(sweep_dir / REPORT_NAME)
    lows = [lo for lo, _, _ in ranges]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every label gets an entry up front, so one that collected nothing shows as 0 rather than
    # vanishing from the report -- that silence is usually the finding.
    counts = {label: SplitCounts() for _, _, label in ranges}

    result = SplitResult(out_dir=out_dir, counts=counts)
    result.unmapped_inputs = _split_file(sweep_dir / INPUTS_NAME, ranges, lows, out_dir, INPUTS_NAME, counts, "inputs")
    result.unmapped_rollouts = _split_file(
        sweep_dir / ROLLOUTS_NAME, ranges, lows, out_dir, ROLLOUTS_NAME, counts, "rollouts"
    )

    report = {
        "sweep_dir": str(sweep_dir),
        "labels": {
            label: {"inputs": c.inputs, "rollouts": c.rollouts, "dir": str(out_dir / label)}
            for label, c in sorted(counts.items())
        },
        "unmapped_inputs": result.unmapped_inputs,
        "unmapped_rollouts": result.unmapped_rollouts,
        "labels_without_rollouts": result.labels_without_rollouts,
    }
    (out_dir / SPLIT_REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return result
