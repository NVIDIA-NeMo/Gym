# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create reproducible, source-balanced samples from a materialized JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


SOURCE_INDEX_PATTERN = re.compile(rb'"source_index"\s*:\s*(\d+)')
OUTSIDE_POLICY_SCOPE_PATTERN = re.compile(rb'"outside_policy_scope"\s*:\s*(true|false)')
DEFAULT_ROWS_PER_SOURCE = 50_000
DEFAULT_SEED = 260715
DEFAULT_BUCKET_COUNT = 64


@dataclass(frozen=True)
class SampleVariant:
    output_path: Path
    dataset_name: str
    seed: int
    exclude_gt_transfer: bool = False
    report_path: Path | None = None

    @property
    def resolved_report_path(self) -> Path:
        return self.report_path or self.output_path.with_suffix(".report.json")


@dataclass(frozen=True)
class InputCounts:
    rows_by_source: dict[int, int]
    gt_transfer_rows_by_source: dict[int, int]

    @property
    def no_transfer_rows_by_source(self) -> dict[int, int]:
        return {
            source_index: count - self.gt_transfer_rows_by_source.get(source_index, 0)
            for source_index, count in self.rows_by_source.items()
        }


def extract_sampling_fields(raw_line: bytes, *, line_number: int) -> tuple[int, bool]:
    source_values = {int(match) for match in SOURCE_INDEX_PATTERN.findall(raw_line)}
    if not source_values:
        raise ValueError(f"line {line_number} has no source_index")
    if len(source_values) != 1:
        raise ValueError(f"line {line_number} has conflicting source_index values: {sorted(source_values)}")
    outside_scope_values = {match == b"true" for match in OUTSIDE_POLICY_SCOPE_PATTERN.findall(raw_line)}
    if not outside_scope_values:
        raise ValueError(f"line {line_number} has no outside_policy_scope flag")
    if len(outside_scope_values) != 1:
        raise ValueError(f"line {line_number} has conflicting outside_policy_scope values")
    return source_values.pop(), outside_scope_values.pop()


def _log_progress(phase: str, rows_seen: int, bytes_seen: int, progress_every: int) -> None:
    if progress_every > 0 and rows_seen % progress_every == 0:
        gib_seen = bytes_seen / (1024**3)
        print(f"[{phase}] scanned {rows_seen:,} rows ({gib_seen:.1f} GiB)", file=sys.stderr, flush=True)


def count_input_rows(input_path: Path, *, progress_every: int = 100_000) -> InputCounts:
    rows_by_source: Counter[int] = Counter()
    gt_transfer_rows_by_source: Counter[int] = Counter()
    bytes_seen = 0
    rows_seen = 0
    with input_path.open("rb") as input_file:
        for line_number, raw_line in enumerate(input_file, 1):
            bytes_seen += len(raw_line)
            if not raw_line.strip():
                continue
            rows_seen += 1
            source_index, gt_transfer = extract_sampling_fields(raw_line, line_number=line_number)
            rows_by_source[source_index] += 1
            gt_transfer_rows_by_source[source_index] += int(gt_transfer)
            _log_progress("count", rows_seen, bytes_seen, progress_every)
    if not rows_by_source:
        raise ValueError(f"input dataset is empty: {input_path}")
    return InputCounts(dict(rows_by_source), dict(gt_transfer_rows_by_source))


def _candidate_counts(variant: SampleVariant, input_counts: InputCounts) -> dict[int, int]:
    if variant.exclude_gt_transfer:
        return input_counts.no_transfer_rows_by_source
    return input_counts.rows_by_source


def choose_rows(
    variant: SampleVariant,
    input_counts: InputCounts,
    *,
    rows_per_source: int,
) -> dict[tuple[int, int], int]:
    candidate_counts = _candidate_counts(variant, input_counts)
    selected_keys: list[tuple[int, int]] = []
    rng = random.Random(variant.seed)
    for source_index in sorted(candidate_counts):
        candidate_count = candidate_counts[source_index]
        if candidate_count < rows_per_source:
            label = "non-transfer " if variant.exclude_gt_transfer else ""
            raise ValueError(
                f"source {source_index} has {candidate_count:,} eligible {label}rows; "
                f"cannot sample {rows_per_source:,}"
            )
        selected_keys.extend(
            (source_index, candidate_ordinal)
            for candidate_ordinal in rng.sample(range(candidate_count), rows_per_source)
        )
    rng.shuffle(selected_keys)
    return {key: output_rank for output_rank, key in enumerate(selected_keys)}


def _sampled_row(row: dict, variant: SampleVariant) -> dict:
    sampled_row = dict(row)
    metadata = dict(row["metadata"])
    original_dataset_name = metadata.get("dataset_name")
    original_row_id = str(row["id"])
    metadata["sampled_from_dataset_name"] = original_dataset_name
    metadata["sampled_from_row_id"] = original_row_id
    metadata["sampling_seed"] = variant.seed
    metadata["dataset_name"] = variant.dataset_name
    sampled_row["metadata"] = metadata

    if original_dataset_name and original_row_id.startswith(f"{original_dataset_name}_"):
        sampled_row["id"] = f"{variant.dataset_name}_{original_row_id[len(original_dataset_name) + 1 :]}"
    else:
        original_id_hash = hashlib.sha256(original_row_id.encode("utf-8")).hexdigest()[:20]
        sampled_row["id"] = f"{variant.dataset_name}_{original_id_hash}"
    return sampled_row


class ShuffledBucketWriter:
    def __init__(self, variant: SampleVariant, temp_dir: Path, *, total_rows: int, bucket_count: int) -> None:
        self.variant = variant
        self.total_rows = total_rows
        self.bucket_count = min(bucket_count, total_rows)
        self.bucket_dir = temp_dir / variant.dataset_name
        self.bucket_dir.mkdir(parents=True)
        self._handles: dict[int, BinaryIO] = {}
        self.rows_by_source: Counter[int] = Counter()
        self.gt_transfer_rows = 0
        self.domain_counts: Counter[tuple[int, str]] = Counter()
        self.source_names: dict[int, str] = {}
        self.row_ids: set[str] = set()

    def _bucket_index(self, output_rank: int) -> int:
        return min(self.bucket_count - 1, output_rank * self.bucket_count // self.total_rows)

    def add(self, output_rank: int, row: dict) -> None:
        metadata = row["metadata"]
        scenario_gt_transfer = row["customer_scenario"].get("outside_policy_scope")
        metadata_gt_transfer = metadata.get("outside_policy_scope")
        if not isinstance(scenario_gt_transfer, bool) or scenario_gt_transfer != metadata_gt_transfer:
            raise ValueError(
                f"row {row.get('id')} has inconsistent outside_policy_scope values: "
                f"scenario={scenario_gt_transfer!r}, metadata={metadata_gt_transfer!r}"
            )
        if self.variant.exclude_gt_transfer and scenario_gt_transfer:
            raise ValueError(f"GT-transfer row {row.get('id')} reached no-transfer output")

        source_index = int(metadata["source_index"])
        self.rows_by_source[source_index] += 1
        self.gt_transfer_rows += int(scenario_gt_transfer)
        self.domain_counts[(source_index, str(row["domain_name"]))] += 1
        self.source_names[source_index] = str(metadata["source_name"])

        sampled_row = _sampled_row(row, self.variant)
        if sampled_row["id"] in self.row_ids:
            raise ValueError(f"duplicate sampled row ID: {sampled_row['id']}")
        self.row_ids.add(sampled_row["id"])
        serialized = json.dumps(sampled_row, ensure_ascii=False).encode("utf-8") + b"\n"
        bucket_index = self._bucket_index(output_rank)
        handle = self._handles.get(bucket_index)
        if handle is None:
            handle = (self.bucket_dir / f"bucket-{bucket_index:04d}.jsonl").open("ab")
            self._handles[bucket_index] = handle
        handle.write(str(output_rank).encode("ascii") + b"\t" + serialized)

    def close_buckets(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def finalize(self) -> tuple[str, int]:
        self.close_buckets()
        self.variant.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = self.variant.output_path.with_name(f".{self.variant.output_path.name}.{os.getpid()}.tmp")
        digest = hashlib.sha256()
        bytes_written = 0
        try:
            with temporary_output.open("wb") as output_file:
                for bucket_index in range(self.bucket_count):
                    bucket_path = self.bucket_dir / f"bucket-{bucket_index:04d}.jsonl"
                    if not bucket_path.exists():
                        continue
                    entries: list[tuple[int, bytes]] = []
                    with bucket_path.open("rb") as bucket_file:
                        for raw_entry in bucket_file:
                            raw_rank, serialized = raw_entry.split(b"\t", 1)
                            entries.append((int(raw_rank), serialized))
                    entries.sort(key=lambda entry: entry[0])
                    for _, serialized in entries:
                        output_file.write(serialized)
                        digest.update(serialized)
                        bytes_written += len(serialized)
            os.replace(temporary_output, self.variant.output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
        return digest.hexdigest(), bytes_written


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, path)


def sample_materialized_datasets(
    input_path: Path,
    variants: list[SampleVariant],
    *,
    rows_per_source: int = DEFAULT_ROWS_PER_SOURCE,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    temp_dir: Path | None = None,
    progress_every: int = 100_000,
) -> list[dict]:
    if rows_per_source <= 0:
        raise ValueError("rows_per_source must be positive")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if not variants:
        raise ValueError("at least one sample variant is required")
    input_path = input_path.resolve()
    output_paths = [variant.output_path.resolve() for variant in variants]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("sample variants must use distinct output paths")
    if input_path in output_paths:
        raise ValueError("sample output path must not overwrite the input dataset")
    dataset_names = [variant.dataset_name for variant in variants]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("sample variants must use distinct dataset names")
    for dataset_name in dataset_names:
        if not dataset_name or Path(dataset_name).name != dataset_name:
            raise ValueError(f"invalid dataset name: {dataset_name!r}")

    input_counts = count_input_rows(input_path, progress_every=progress_every)
    selections = {
        variant.dataset_name: choose_rows(variant, input_counts, rows_per_source=rows_per_source)
        for variant in variants
    }
    total_rows = rows_per_source * len(input_counts.rows_by_source)

    parent_temp_dir = temp_dir or variants[0].output_path.parent
    parent_temp_dir.mkdir(parents=True, exist_ok=True)
    working_dir = Path(tempfile.mkdtemp(prefix=".synthetic-sample-", dir=parent_temp_dir))
    writers = {
        variant.dataset_name: ShuffledBucketWriter(
            variant,
            working_dir,
            total_rows=total_rows,
            bucket_count=bucket_count,
        )
        for variant in variants
    }
    candidate_ordinals = {variant.dataset_name: Counter() for variant in variants}
    rows_seen = 0
    bytes_seen = 0
    try:
        with input_path.open("rb") as input_file:
            for line_number, raw_line in enumerate(input_file, 1):
                bytes_seen += len(raw_line)
                if not raw_line.strip():
                    continue
                rows_seen += 1
                source_index, gt_transfer = extract_sampling_fields(raw_line, line_number=line_number)
                selected_variants: list[tuple[SampleVariant, int]] = []
                for variant in variants:
                    if variant.exclude_gt_transfer and gt_transfer:
                        continue
                    ordinal = candidate_ordinals[variant.dataset_name][source_index]
                    candidate_ordinals[variant.dataset_name][source_index] += 1
                    output_rank = selections[variant.dataset_name].get((source_index, ordinal))
                    if output_rank is not None:
                        selected_variants.append((variant, output_rank))
                if selected_variants:
                    row = json.loads(raw_line)
                    for variant, output_rank in selected_variants:
                        writers[variant.dataset_name].add(output_rank, row)
                _log_progress("sample", rows_seen, bytes_seen, progress_every)

        reports = []
        for variant in variants:
            writer = writers[variant.dataset_name]
            expected_rows_by_source = {source_index: rows_per_source for source_index in input_counts.rows_by_source}
            if dict(writer.rows_by_source) != expected_rows_by_source:
                raise RuntimeError(
                    f"selected row count mismatch for {variant.dataset_name}: "
                    f"expected {expected_rows_by_source}, got {dict(writer.rows_by_source)}"
                )
            output_sha256, output_bytes = writer.finalize()
            domain_counts = [
                {
                    "source_index": source_index,
                    "domain_name": domain_name,
                    "rows": count,
                }
                for (source_index, domain_name), count in sorted(writer.domain_counts.items())
            ]
            report = {
                "input_path": str(input_path),
                "output_path": str(variant.output_path),
                "dataset_name": variant.dataset_name,
                "sampling_method": "uniform_without_replacement_per_source_then_global_shuffle",
                "seed": variant.seed,
                "rows_per_source": rows_per_source,
                "exclude_gt_transfer": variant.exclude_gt_transfer,
                "rows_written": sum(writer.rows_by_source.values()),
                "gt_transfer_rows_written": writer.gt_transfer_rows,
                "output_bytes": output_bytes,
                "output_sha256": output_sha256,
                "input": {
                    "rows_by_source": input_counts.rows_by_source,
                    "gt_transfer_rows_by_source": input_counts.gt_transfer_rows_by_source,
                    "no_transfer_rows_by_source": input_counts.no_transfer_rows_by_source,
                },
                "selected": {
                    "rows_by_source": dict(writer.rows_by_source),
                    "source_names": writer.source_names,
                    "domain_count": len(writer.domain_counts),
                    "domain_counts": domain_counts,
                },
            }
            _write_report(variant.resolved_report_path, report)
            reports.append(report)
        return reports
    finally:
        for writer in writers.values():
            writer.close_buckets()
        shutil.rmtree(working_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--no-transfer-output-path", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--no-transfer-dataset-name", default=None)
    parser.add_argument("--rows-per-source", type=int, default=DEFAULT_ROWS_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [
        SampleVariant(
            output_path=args.output_path,
            dataset_name=args.dataset_name or args.output_path.stem,
            seed=args.seed,
        ),
        SampleVariant(
            output_path=args.no_transfer_output_path,
            dataset_name=args.no_transfer_dataset_name or args.no_transfer_output_path.stem,
            seed=args.seed + 1,
            exclude_gt_transfer=True,
        ),
    ]
    reports = sample_materialized_datasets(
        args.input_path,
        variants,
        rows_per_source=args.rows_per_source,
        bucket_count=args.bucket_count,
        temp_dir=args.temp_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
