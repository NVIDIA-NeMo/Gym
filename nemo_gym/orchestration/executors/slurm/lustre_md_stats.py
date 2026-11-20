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
"""Lustre metadata request counts of a submitted Slurm job, per phase (the compute's `lustre_md_stats`).

The job script appends one TSV row per node, MDT and operation at each phase to
`<benchmark dir>/logs/lustre-md-stats-<slurm job id>.tsv`:
`phase<TAB>epoch<TAB>node<TAB>mdt<TAB>op<TAB>count`. A count is the node's cumulative MDC request counter
(`lctl get_param mdc.*.md_stats`), so only the difference between two phases means anything.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.orchestration.jobs import LOGS_DIRNAME
from nemo_gym.utils.decorators import experimental


START = "start"
SERVICES_READY = "services_ready"
DRIVER_END = "driver_end"
END = "end"
PHASES = (START, SERVICES_READY, DRIVER_END, END)
INTERVALS = ((START, SERVICES_READY), (SERVICES_READY, DRIVER_END), (DRIVER_END, END))

STATS_FILE_PREFIX = "lustre-md-stats-"
STATS_FILE_SUFFIX = ".tsv"

_TOP = 3


@dataclass
class NodeSnapshot:
    """One node's counters at one phase, by (MDT, operation). An operation lctl omitted counts 0."""

    epoch: int
    counts: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class IntervalStats:
    """Requests made between two phases, summed over the nodes that recorded both."""

    name: str
    first: str
    last: str
    seconds: int
    nodes: int
    requests: int
    by_mdt: dict[str, int]
    by_op: dict[str, int]
    peak_node: str | None
    peak_node_requests_per_second: float | None

    @property
    def requests_per_second(self) -> float | None:
        return self.requests / self.seconds if self.seconds > 0 else None

    def to_json(self) -> dict[str, Any]:
        return {
            "interval": self.name,
            "from": self.first,
            "to": self.last,
            "seconds": self.seconds,
            "nodes": self.nodes,
            "requests": self.requests,
            "requests_per_second": self.requests_per_second,
            "peak_node": self.peak_node,
            "peak_node_requests_per_second": self.peak_node_requests_per_second,
            "by_mdt": self.by_mdt,
            "by_op": self.by_op,
        }


def load_snapshots(path: Path) -> dict[str, dict[str, NodeSnapshot]]:
    """Read a stats file into phase -> node -> snapshot.

    A line that is not a whole row (one cut short when the job was killed) is skipped. A `start` row after rows of
    a later phase begins the file again: Slurm requeued the job under the same ID, and only its last run counts.
    """
    snapshots: dict[str, dict[str, NodeSnapshot]] = {}
    previous = None
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            continue
        phase, epoch, node, mdt, op, count = fields
        try:
            epoch_s, requests = int(epoch), int(count)
        except ValueError:
            continue
        if phase == START and previous not in (None, START):
            snapshots.clear()
        previous = phase
        snapshot = snapshots.setdefault(phase, {}).setdefault(node, NodeSnapshot(epoch_s))
        snapshot.counts[(mdt, op)] = snapshot.counts.get((mdt, op), 0) + requests
    return snapshots


def _ranked(counter: Counter) -> dict[str, int]:
    return {key: value for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])) if value}


def interval_stats(
    snapshots: dict[str, dict[str, NodeSnapshot]], first: str, last: str, *, name: str | None = None
) -> IntervalStats | None:
    """The requests between phases `first` and `last`, or None when no node recorded both.

    `seconds` is the longest per-node interval; the peak is the node with the highest requests per second.
    """
    before, after = snapshots.get(first, {}), snapshots.get(last, {})
    nodes = sorted(before.keys() & after.keys())
    if not nodes:
        return None
    by_mdt: Counter = Counter()
    by_op: Counter = Counter()
    seconds = requests = 0
    peak_node, peak_rate = None, None
    for node in nodes:
        a, b = before[node], after[node]
        node_requests = 0
        for mdt, op in a.counts.keys() | b.counts.keys():
            delta = b.counts.get((mdt, op), 0) - a.counts.get((mdt, op), 0)
            by_mdt[mdt] += delta
            by_op[op] += delta
            node_requests += delta
        node_seconds = b.epoch - a.epoch
        if node_seconds > 0 and (peak_rate is None or node_requests / node_seconds > peak_rate):
            peak_node, peak_rate = node, node_requests / node_seconds
        seconds = max(seconds, node_seconds)
        requests += node_requests
    return IntervalStats(
        name=name or f"{first}->{last}",
        first=first,
        last=last,
        seconds=seconds,
        nodes=len(nodes),
        requests=requests,
        by_mdt=_ranked(by_mdt),
        by_op=_ranked(by_op),
        peak_node=peak_node,
        peak_node_requests_per_second=peak_rate,
    )


def analyze(path: Path) -> dict[str, Any]:
    """Every phase interval of a stats file, then `total` from its first recorded phase to its last."""
    snapshots = load_snapshots(path)
    phases = [phase for phase in PHASES if phase in snapshots]
    intervals = [stats for first, last in INTERVALS if (stats := interval_stats(snapshots, first, last))]
    if len(phases) > 1 and (total := interval_stats(snapshots, phases[0], phases[-1], name="total")):
        intervals.append(total)
    return {
        "nodes": sorted({node for by_node in snapshots.values() for node in by_node}),
        "phases": phases,
        "intervals": [stats.to_json() for stats in intervals],
    }


def stats_files(target: Path) -> list[Path]:
    """The stats files of a benchmark directory (one per Slurm job of the run), or `target` itself when a file."""
    if target.is_file():
        return [target]
    pattern = f"{STATS_FILE_PREFIX}*{STATS_FILE_SUFFIX}"
    for directory in (target / LOGS_DIRNAME, target):
        # Numeric job IDs in submission order: a shorter ID is an older job.
        files = sorted(directory.glob(pattern), key=lambda path: (len(path.name), path.name))
        if files:
            return files
    raise FileNotFoundError(
        f"No {STATS_FILE_PREFIX}<slurm job id>{STATS_FILE_SUFFIX} in {target / LOGS_DIRNAME} or {target}. "
        "Lustre metadata stats are recorded only for a job whose Slurm compute sets lustre_md_stats: true."
    )


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _top(counts: dict[str, int], seconds: int) -> str:
    shown = list(counts.items())[:_TOP]
    if seconds > 0:
        return ", ".join(f"{name} {count / seconds:.1f}/s" for name, count in shown)
    return ", ".join(f"{name} {count}" for name, count in shown)


def format_report(path: Path, report: dict[str, Any]) -> str:
    """A text table of one stats file's report: one row per interval."""
    phases = report["phases"]
    lines = [f"{path}: {len(report['nodes'])} node(s), phases {', '.join(phases) or 'none'}"]
    if END not in phases:
        lines.append("  no `end` snapshot: the job did not exit normally (time limit, cancel or node failure).")
    if not report["intervals"]:
        lines.append("  fewer than two phases recorded; nothing to compare.")
        return "\n".join(lines)
    header = ("interval", "seconds", "requests", "req/s", "peak node req/s", "top MDTs", "top ops")
    rows = [header] + [
        (
            item["interval"],
            str(item["seconds"]),
            str(item["requests"]),
            _rate(item["requests_per_second"]),
            f"{_rate(item['peak_node_requests_per_second'])} ({item['peak_node']})" if item["peak_node"] else "-",
            _top(item["by_mdt"], item["seconds"]),
            _top(item["by_op"], item["seconds"]),
        )
        for item in report["intervals"]
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    for row in rows:
        cells = [
            cell.rjust(width) if 0 < column < 4 else cell.ljust(width)
            for column, (cell, width) in enumerate(zip(row, widths))
        ]
        lines.append("  " + "  ".join(cells).rstrip())
    return "\n".join(lines)


@experimental
def print_lustre_md_stats(target: Path, *, json_output: bool = False) -> None:
    """Print the report of every stats file under `target`; with `json_output`, a mapping from file to report."""
    reports = {path: analyze(path) for path in stats_files(target)}
    if json_output:
        print(json.dumps({str(path): report for path, report in reports.items()}, indent=2))
        return
    print("\n\n".join(format_report(path, report) for path, report in reports.items()))
