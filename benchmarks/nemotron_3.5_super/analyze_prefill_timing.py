# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize VLLM_PREFILL_TIMING JSON embedded in combined serving logs.

    python benchmarks/nemotron_3.5_super/analyze_prefill_timing.py temp.log
    python benchmarks/nemotron_3.5_super/analyze_prefill_timing.py temp.log \
        --skip-first-seconds 60 --json results/prefill_timing.json --csv results/prefill_timing.csv

Uses only the standard library. Reads lines incrementally, retaining parsed windows
for sorting/filtering and optional CSV export. Damaged/interleaved fragments are
counted and skipped; exact repeated records across input files are deduplicated.
Means are weighted by observation count. Global percentiles cannot be recovered
from window percentiles; CSV preserves the original per-window quantiles instead.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


MARKER = re.compile(r"VLLM_PREFILL_TIMING\s+(?=\{)")
SECTIONS = ("cpu_ms", "cuda_elapsed_ms", "batch")
NOTES = [
    "Means are sum(total)/sum(count); totals cover observed samples only and are not scaled by sample_every.",
    "Worker steps are sampled; scheduler CPU phases and execute_start_gap are unsampled.",
    "execute_start_gap is time between execute starts, including work and idle periods; it is not GPU idle time.",
    "Phases can cover different steps (execute_model also includes no-forward calls); compare observation counts.",
    "Nested phases overlap. Do not sum phase totals, CPU+CUDA times, or interpret rank sums as wall time.",
    "CUDA intervals include stream waits and host launch gaps; they are not pure kernel compute time.",
    "CUDA completions may land in later windows, so CPU and CUDA counts/windows need not match.",
    "Global p50/p90/p99 cannot be reconstructed. CSV retains logged per-window quantiles (up to 4096 samples).",
    "Time filtering uses reporting-window starts; it cannot remove individual samples or delayed CUDA completions.",
]


def number(value: object, *, integer: bool = False) -> float | int:
    """Validate nonnegative, finite numeric fields before aggregating them."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    if not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError("expected a nonnegative finite number" + (" (integer)" if integer else ""))
    return value


def mapping(value: object) -> dict[str, object]:
    """Require a JSON object."""
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


@dataclass
class Samples:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0

    def add(self, stats: dict[str, float | int]) -> None:
        self.count += int(stats["count"])
        self.total += stats["total"]
        self.maximum = max(self.maximum, stats["max"])

    def report(self) -> dict[str, float | int]:
        return {"count": self.count, "total": self.total, "mean": self.total / self.count, "max": self.maximum}


@dataclass
class Window:
    labels: dict[str, str | int]
    timestamp: float
    seconds: float
    sample_every: int
    sections: dict[str, dict[str, dict[str, float | int]]]
    counts: dict[str, int]
    pending: int
    source: str
    line: int

    @property
    def process(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.labels.items())

    @property
    def start(self) -> float:
        return self.timestamp - self.seconds

    @classmethod
    def parse(cls, record: dict[str, object], *, source: str, line: int) -> "Window":
        labels: dict[str, str | int] = {}
        for name in ("component", "scope", "host", "pid", "dp_rank", "rank"):
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(f"invalid {name}")
            labels[name] = value
        sections = {}
        for section in SECTIONS:
            phases = {}
            for name, raw in mapping(record[section]).items():
                stats = mapping(raw)
                parsed = {k: number(stats[k], integer=k == "count") for k in ("count", "total", "max")}
                if parsed["count"] < 1:
                    raise ValueError("empty phase summary")
                for key in ("p50", "p90", "p99", "percentile_samples"):
                    if key in stats:
                        parsed[key] = number(stats[key], integer=key == "percentile_samples")
                phases[name] = parsed
            sections[section] = phases
        sample_every = int(number(record["sample_every"], integer=True))
        if sample_every < 1:
            raise ValueError("sample_every must be positive")
        return cls(
            labels=labels,
            timestamp=float(number(record["timestamp"])),
            seconds=float(number(record["window_seconds"])),
            sample_every=sample_every,
            sections=sections,
            counts={k: int(number(v, integer=True)) for k, v in mapping(record["counts"]).items()},
            pending=int(number(record["pending_cuda_ranges"], integer=True)),
            source=source,
            line=line,
        )


def read_windows(paths: list[Path]) -> tuple[list[Window], dict[str, object]]:
    """Recover complete records after each marker, including concatenated lines."""
    windows = []
    seen: set[bytes] = set()
    malformed = duplicates = 0
    errors = []
    decoder = json.JSONDecoder()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                markers = list(MARKER.finditer(line))
                for index, marker in enumerate(markers):
                    end = markers[index + 1].start() if index + 1 < len(markers) else len(line)
                    try:
                        raw, _ = decoder.raw_decode(line[marker.end() : end])
                        window = Window.parse(mapping(raw), source=str(path), line=line_number)
                    except (ValueError, KeyError, OverflowError) as exc:
                        malformed += 1
                        if len(errors) < 5:
                            errors.append(f"{path}:{line_number}: {exc}")
                        continue
                    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).digest()
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    windows.append(window)
    windows.sort(key=lambda w: w.timestamp)
    return windows, {
        "valid_records": len(windows),
        "malformed_fragments": malformed,
        "duplicates": duplicates,
        "error_examples": errors,
    }


def summarize(windows: list[Window]) -> dict[str, object]:
    """Pool counts/totals while preserving process identity and pending gauges."""
    stats: dict[str, dict[str, Samples]] = {section: {} for section in SECTIONS}
    counts: Counter[str] = Counter()
    latest: dict[str, Window] = {}
    for window in sorted(windows, key=lambda w: w.timestamp):
        latest[window.process] = window
        counts.update(window.counts)
        for section, phases in window.sections.items():
            for phase, values in phases.items():
                stats[section].setdefault(phase, Samples()).add(values)
    return {
        "records": len(windows),
        "processes": len(latest),
        "first_window_start_utc": utc(min(w.start for w in windows)),
        "last_window_end_utc": utc(max(w.timestamp for w in windows)),
        "sample_every": sorted({w.sample_every for w in windows}),
        "counts": dict(counts),
        "max_pending_in_one_record": max(w.pending for w in windows),
        "pending_at_last_record_per_process": {k: w.pending for k, w in latest.items()},
        **{s: {p: v.report() for p, v in phases.items()} for s, phases in stats.items()},
    }


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def analyze(paths: list[Path], *, skip_first_seconds: float = 0) -> tuple[dict[str, object], list[Window]]:
    """Discard windows starting before the global observed start plus an offset."""
    number(skip_first_seconds)
    windows, parsing = read_windows(paths)
    if not windows:
        raise ValueError(
            f"No valid VLLM_PREFILL_TIMING records ({parsing['malformed_fragments']} malformed fragments)"
        )
    origin = min(w.start for w in windows)
    selected = [w for w in windows if w.start >= origin + skip_first_seconds]
    groups: dict[str, list[Window]] = {}
    processes: dict[str, list[Window]] = {}
    for window in selected:
        key = f"{window.labels['component']} / {window.labels['scope']} / sample_every={window.sample_every}"
        groups.setdefault(key, []).append(window)
        processes.setdefault(window.process, []).append(window)
    report = {
        "inputs": [str(p) for p in paths],
        "parsing": parsing,
        "observed_start_utc": utc(origin),
        "cutoff_utc": utc(origin + skip_first_seconds),
        "excluded_records": len(windows) - len(selected),
        "included_records": len(selected),
        "groups": {k: summarize(v) for k, v in sorted(groups.items())},
        "processes": {k: summarize(v) for k, v in sorted(processes.items())},
        "notes": NOTES,
    }
    return report, selected


def write_csv(path: Path, windows: list[Window]) -> None:
    """Export original window quantiles for time-series and rank comparisons."""
    fields = [
        "source",
        "line",
        "component",
        "scope",
        "host",
        "pid",
        "dp_rank",
        "rank",
        "window_start_utc",
        "window_end_utc",
        "window_seconds",
        "sample_every",
        "section",
        "phase",
        "count",
        "total",
        "mean",
        "max",
        "p50",
        "p90",
        "p99",
        "percentile_samples",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for window in windows:
            for section, phases in window.sections.items():
                for phase, stats in phases.items():
                    writer.writerow(
                        {
                            **window.labels,
                            "source": window.source,
                            "line": window.line,
                            "window_start_utc": utc(window.start),
                            "window_end_utc": utc(window.timestamp),
                            "window_seconds": window.seconds,
                            "sample_every": window.sample_every,
                            "section": section,
                            "phase": phase,
                            **stats,
                            "mean": stats["total"] / stats["count"],
                        }
                    )


def print_group(label: str, group: dict[str, object]) -> None:
    print(f"\n{label}: {group['records']} records; {group['processes']} processes")
    print(f"  Window coverage: {group['first_window_start_utc']} to {group['last_window_end_utc']}")
    print(f"  Counters: {group['counts']}; max pending in one record: {group['max_pending_in_one_record']}")
    for section in SECTIONS:
        print(f"  {section} (batch is values, timing sections are milliseconds):")
        print(f"    {'phase':34} {'count':>10} {'total':>14} {'mean':>12} {'max':>12}")
        for phase, raw in mapping(group[section]).items():
            stats = mapping(raw)
            print(
                f"    {phase:34} {stats['count']:>10,} {stats['total']:>14,.3f} "
                f"{stats['mean']:>12,.3f} {stats['max']:>12,.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="Write pooled and per-process summaries")
    parser.add_argument("--csv", type=Path, help="Write original per-window phase statistics, including quantiles")
    parser.add_argument("--per-worker", action="store_true", help="Also print each worker/scheduler process")
    parser.add_argument(
        "--skip-first-seconds",
        type=float,
        default=0,
        help="Exclude windows starting within N seconds of the first observed window start",
    )
    args = parser.parse_args()
    try:
        report, windows = analyze(args.logs, skip_first_seconds=args.skip_first_seconds)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Parsing: {report['parsing']}")
    print(f"Included: {report['included_records']}; excluded by time filter: {report['excluded_records']}")
    for label, group in mapping(report["groups"]).items():
        print_group(label, mapping(group))
    if args.per_worker:
        for label, group in mapping(report["processes"]).items():
            print_group(label, mapping(group))
    print("\n" + "\n".join(NOTES))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON: {args.json}")
    if args.csv:
        write_csv(args.csv, windows)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
