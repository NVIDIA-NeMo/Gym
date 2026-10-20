# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize removal-only cache traces in combined vLLM/Mooncake logs.

    python results/analyze_mambacachetrace.py run.log
    python results/analyze_mambacachetrace.py run.log --json summary.json
    python results/analyze_mambacachetrace.py run.log --hash FULL_HASH --json hash.json

Only observed removals contribute statistics; live entries are absent. GPU
records describe individual group/block residencies, not unique requests.
Reads are streamed, but distinct IDs/hashes and scalar samples are retained
for deduplication and exact quantiles. Without instance IDs, events in separate
files cannot be reliably deduplicated; avoid overlapping log exports.
No third-party dependencies are needed.
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean


MARKER = re.compile(r"MAMBA_CACHE_TRACE\s+(?=\{)")
MOONCAKE_GROUP = re.compile(r"@group:(\d+)@")
COUNTERS = (
    "lookup_hit_count",
    "successful_load_count",
    "checkpoint_resume_count",
    "read_request_count",
    "computed_hash_count_direct",
    "computed_hash_count_descendants",
)
LEGACY_TIME_FIELDS = {"created_ns", "retired_ns", "removed_ns", "lifetime_ns", "time_to_retirement_ns"}
THRESHOLDS = {
    "lookup_hit_count": (4, 8, 16),
    "checkpoint_resume_count": (2, 4, 6),
    "computed_hash_count_direct": (2, 3, 4, 5, 6),
}


def normalize_record(record: dict[str, object]) -> dict[str, object]:
    """Extract Mooncake groups and normalize legacy trace fields."""
    record = record.copy()
    record.pop("instance_id", None)
    object_key = record.get("object_key")
    if "group_id" not in record and isinstance(object_key, str):
        if match := MOONCAKE_GROUP.search(object_key):
            record["group_id"] = int(match[1])
    descendants = record.pop("computed_hashes", None)
    if isinstance(descendants, list):
        record.setdefault("computed_hash_count_descendants", len(set(descendants)))
    return record


def source_of(record: dict[str, object]) -> str:
    """Distinguish native storage records from GPU and connector observations."""
    if "object_key" in record and "read_request_count" in record:
        return "mooncake"
    if "block_id" in record and "group_id" in record:
        return "gpu"
    return "other"


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """Return exact, linearly interpolated sample quantiles (null when absent)."""
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "samples": len(values),
        "mean": mean(values) if values else None,
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "max": ordered[-1] if ordered else None,
    }


@dataclass
class RemovalStats:
    records: int = 0
    hashes: set[str] = field(default_factory=set)
    groups: Counter[str] = field(default_factory=Counter)
    samples: dict[str, list[float]] = field(default_factory=dict)
    retired: int = 0
    retirement_observed: int = 0
    first_ns: int | None = None
    last_ns: int | None = None
    top_records: list[dict[str, object]] = field(default_factory=list)

    def add(self, record: dict[str, object], *, top: int) -> None:
        self.records += 1
        prefix_hash = record.get("prefix_hash")
        if isinstance(prefix_hash, str) and prefix_hash:
            self.hashes.add(prefix_hash)
        if "group_id" in record:
            self.groups[str(record["group_id"])] += 1
        for name in (*COUNTERS, "lifetime_ns", "time_to_retirement_ns"):
            value = record.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                key = name.replace("_ns", "_seconds")
                self.samples.setdefault(key, []).append(value / 1e9 if name.endswith("_ns") else value)
        if "retired_ns" in record:
            self.retirement_observed += 1
            self.retired += record["retired_ns"] is not None
        removed = record.get("removed_ns")
        if isinstance(removed, int):
            self.first_ns = removed if self.first_ns is None else min(self.first_ns, removed)
            self.last_ns = removed if self.last_ns is None else max(self.last_ns, removed)
        if top:
            self.top_records.append({key: value for key, value in record.items() if key not in LEGACY_TIME_FIELDS})
            self.top_records.sort(key=self._score, reverse=True)
            del self.top_records[top:]

    @staticmethod
    def _score(record: dict[str, object]) -> tuple[int, int]:
        def count(key: str) -> int:
            value = record.get(key)
            return value if isinstance(value, int) else 0

        return count("checkpoint_resume_count") + count("read_request_count"), count("lookup_hit_count")

    def report(self) -> dict[str, object]:
        def utc(value: int | None) -> str | None:
            return datetime.fromtimestamp(value / 1e9, UTC).isoformat() if value is not None else None

        return {
            "records": self.records,
            "distinct_prefix_hashes": len(self.hashes),
            "groups": dict(self.groups),
            "first_removal_utc": utc(self.first_ns),
            "last_removal_utc": utc(self.last_ns),
            "retirement_field_present": self.retirement_observed,
            "retired_before_removal": self.retired,
            "statistics": {
                key: {
                    **distribution(values),
                    "sum": sum(values),
                    "zero_records": values.count(0),
                    "threshold_counts": {
                        str(threshold): sum(value >= threshold for value in values)
                        for threshold in THRESHOLDS.get(key, ())
                    },
                }
                for key, values in self.samples.items()
            },
            "top_records": self.top_records,
        }


def analyze(paths: list[Path], *, prefix_hash: str | None = None, top: int = 3) -> dict[str, object]:
    """Parse each marker independently; skip damaged JSON and duplicate event IDs."""
    decoder = json.JSONDecoder()
    seen: set[tuple[object, ...]] = set()
    stats = {source: RemovalStats() for source in ("gpu", "mooncake", "other")}
    events: Counter[str] = Counter()
    malformed = duplicates = lines = 0
    errors: list[str] = []
    for path in paths:
        resolved = str(path.resolve())
        with path.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                lines += 1
                markers = list(MARKER.finditer(line))
                for index, marker in enumerate(markers):
                    end = markers[index + 1].start() if index + 1 < len(markers) else len(line)
                    try:
                        record, _ = decoder.raw_decode(line[marker.end() : end])
                        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                            raise ValueError("Expected an object with an event name")
                    except ValueError as error:
                        malformed += 1
                        if len(errors) < 5:
                            errors.append(f"{path}:{number}: {error}")
                        continue
                    instance, sequence = record.get("instance_id"), record.get("sequence")
                    if isinstance(instance, str) and isinstance(sequence, int):
                        identity = ("legacy", instance, sequence)
                    else:
                        # Sequence alone collides across processes. Only skip
                        # the same file location when a path is supplied twice.
                        identity = ("location", resolved, number, index)
                    if identity in seen:
                        duplicates += 1
                        continue
                    seen.add(identity)
                    if prefix_hash is not None and record.get("prefix_hash") != prefix_hash:
                        continue
                    source = source_of(record)
                    events[f"{source}:{record['event']}"] += 1
                    if record["event"] == "hash_removed":
                        stats[source].add(normalize_record(record), top=top)
    return {
        "files": [str(path) for path in paths],
        "hash_filter": prefix_hash,
        "lines": lines,
        "malformed_fragments": malformed,
        "malformed_examples": errors,
        "duplicate_events": duplicates,
        "events": dict(events),
        "removals": {source: summary.report() for source, summary in stats.items()},
    }


def print_report(report: dict) -> None:
    """Print a compact CLI summary; JSON output retains full top-record hashes."""
    print(f"Events: {report['events']}")
    print(f"Skipped: {report['malformed_fragments']} malformed fragments, {report['duplicate_events']} duplicates")
    for source, stats in report["removals"].items():
        if not stats["records"]:
            continue
        print(
            f"\n{source.upper()}: {stats['records']:,} removals; {stats['distinct_prefix_hashes']:,} distinct hashes"
        )
        if stats["groups"]:
            print(f"  Groups: {stats['groups']}")
        if stats["first_removal_utc"] is not None:
            print(f"  Removal window: {stats['first_removal_utc']} to {stats['last_removal_utc']}")
        for key, values in stats["statistics"].items():
            if key.endswith("_seconds"):
                print(
                    f"  {key}: median={values['p50']:.3f}, p90={values['p90']:.3f}, "
                    f"p99={values['p99']:.3f}, max={values['max']:.3f} (n={values['samples']:,})"
                )
            else:
                thresholds = "".join(
                    f"; >={threshold}={count:,}" for threshold, count in values.get("threshold_counts", {}).items()
                )
                print(
                    f"  {key}: total={values['sum']:,}; zero={values['zero_records']:,}/{values['samples']:,}; "
                    f"p50={values['p50']:,.3f}; p90={values['p90']:,.3f}; "
                    f"max={values['max']:,}{thresholds}"
                )
        if source == "gpu":
            direct = stats["statistics"].get("computed_hash_count_direct", {}).get("samples", 0)
            if direct < stats["records"]:
                print(f"  computed_hash_count_direct: unavailable for {stats['records'] - direct:,} legacy records")
        if stats["retirement_field_present"]:
            print(
                f"  Retired before removal: {stats['retired_before_removal']:,}/{stats['retirement_field_present']:,}"
            )
        for record in stats["top_records"]:
            usage = (
                f"group={record.get('group_id')} resumes={record.get('checkpoint_resume_count')} "
                f"direct={record.get('computed_hash_count_direct', 'unavailable')} "
                f"descendants={record.get('computed_hash_count_descendants', 'unavailable')}"
                if source == "gpu"
                else f"reads={record.get('read_request_count', 'unavailable')}"
            )
            print(f"  Top: {record.get('prefix_hash')} {usage}")
    if not report["removals"]["mooncake"]["records"]:
        print("\nNo native Mooncake hash_removed records found in this selection.")
        print("This does not distinguish disabled tracing from no object removals during the captured interval.")
    print("\nStatistics cover observed removals only, not live entries or a global hit rate.")
    print("GPU counts are per residency/group, not unique requests; computed hashes need not be stored checkpoints.")
    print(
        "Direct counts distinct first-child hashes; descendants counts all distinct subsequent hashes, including direct."
    )
    print("Without instance IDs, overlapping log exports cannot be deduplicated across files.")
    print("Mooncake read_request_count counts metadata responses, not confirmed loads or model resumes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--hash", dest="prefix_hash", help="Only summarize this full prefix hash")
    parser.add_argument("--top", type=int, default=3, help="Number of most reused records per source (default: 3)")
    parser.add_argument(
        "--json", type=Path, help="Write full summary and top records, including computed-hash counters"
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    report = analyze(args.logs, prefix_hash=args.prefix_hash, top=args.top)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
