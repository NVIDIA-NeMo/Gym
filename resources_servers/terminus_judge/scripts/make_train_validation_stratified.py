#!/usr/bin/env python3
"""Create deduplicated stratified train/validation splits for terminus_judge."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


BUCKETS = [
    "0-200",
    "200-500",
    "500-1000",
    "1000-2000",
    "2000-5000",
    "5000+",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/validation splits with exact-match grouping and stratification."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input samples.jsonl path.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for split files.")
    parser.add_argument(
        "--train_size",
        type=int,
        default=0,
        help="Train size after validation removal (0 means all remaining samples).",
    )
    parser.add_argument(
        "--val_per_bucket",
        type=int,
        default=200,
        help="Number of validation samples drawn per keystroke-length bucket.",
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=50,
        help="Maximum samples to keep per exact-match command group.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_keystrokes(sample: dict[str, Any]) -> list[str]:
    expected_answer = sample.get("expected_answer", "{}")
    if isinstance(expected_answer, str):
        try:
            expected_answer = json.loads(expected_answer)
        except json.JSONDecodeError:
            return []
    if not isinstance(expected_answer, dict):
        return []
    commands = expected_answer.get("commands", [])
    if not isinstance(commands, list):
        return []
    keystrokes: list[str] = []
    for command in commands:
        if isinstance(command, dict) and "keystrokes" in command:
            value = command.get("keystrokes")
            keystrokes.append(value if isinstance(value, str) else str(value))
    return keystrokes


def _group_key(sample: dict[str, Any]) -> str:
    # Deterministic exact-match key using the command content sequence.
    # We keep sort_keys=True so nested structures remain stable if present.
    key_payload = {"keystrokes": _extract_keystrokes(sample)}
    return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))


def _total_keystroke_len(sample: dict[str, Any]) -> int:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("total_keystroke_len")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)

    return sum(len(k) for k in _extract_keystrokes(sample))


def _bucket_name(length: int) -> str:
    if length < 200:
        return "0-200"
    if length < 500:
        return "200-500"
    if length < 1000:
        return "500-1000"
    if length < 2000:
        return "1000-2000"
    if length < 5000:
        return "2000-5000"
    return "5000+"


def _stratified_counts(available: dict[str, int], requested_total: int) -> dict[str, int]:
    total_available = sum(available.values())
    if requested_total <= 0 or total_available == 0:
        return {bucket: 0 for bucket in BUCKETS}
    if requested_total >= total_available:
        return dict(available)

    raw_targets = {bucket: (available[bucket] / total_available) * requested_total for bucket in BUCKETS}
    assigned = {bucket: min(available[bucket], int(raw_targets[bucket])) for bucket in BUCKETS}
    used = sum(assigned.values())

    remainders = sorted(
        BUCKETS,
        key=lambda bucket: (raw_targets[bucket] - int(raw_targets[bucket])),
        reverse=True,
    )

    idx = 0
    while used < requested_total and idx < len(remainders) * 4:
        bucket = remainders[idx % len(remainders)]
        if assigned[bucket] < available[bucket]:
            assigned[bucket] += 1
            used += 1
        idx += 1

    return assigned


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading samples from {args.input}")
    samples = _load_jsonl(args.input)
    print(f"Loaded {len(samples)} samples")

    print("\nGrouping by exact command-content key...")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    print(f"Unique groups: {len(groups)}")

    print(f"\nApplying max_per_group={args.max_per_group}...")
    kept_samples: list[dict[str, Any]] = []
    truncated_groups = 0
    dropped_samples = 0
    for group_samples in groups.values():
        if len(group_samples) > args.max_per_group:
            rng.shuffle(group_samples)
            dropped_samples += len(group_samples) - args.max_per_group
            truncated_groups += 1
            group_samples = group_samples[: args.max_per_group]
        kept_samples.extend(group_samples)
    print(f"Groups truncated: {truncated_groups}")
    print(f"Samples dropped by cap: {dropped_samples}")
    print(f"Samples after cap: {len(kept_samples)}")

    print("\nBucketing by metadata.total_keystroke_len...")
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in kept_samples:
        bucket = _bucket_name(_total_keystroke_len(sample))
        by_bucket[bucket].append(sample)

    for bucket in BUCKETS:
        print(f"  {bucket}: {len(by_bucket[bucket])}")

    print(f"\nSampling validation set (val_per_bucket={args.val_per_bucket})...")
    validation: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        bucket_samples = list(by_bucket[bucket])
        rng.shuffle(bucket_samples)
        n_val = min(args.val_per_bucket, len(bucket_samples))
        validation.extend(bucket_samples[:n_val])
        remaining.extend(bucket_samples[n_val:])
        print(f"  {bucket}: validation={n_val}, remaining={len(bucket_samples) - n_val}")

    remaining_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in remaining:
        remaining_by_bucket[_bucket_name(_total_keystroke_len(sample))].append(sample)

    for bucket in BUCKETS:
        rng.shuffle(remaining_by_bucket[bucket])

    if args.train_size == 0:
        print("\ntrain_size=0 -> using all remaining samples for train")
        train = list(remaining)
    else:
        available_counts = {bucket: len(remaining_by_bucket[bucket]) for bucket in BUCKETS}
        target_counts = _stratified_counts(available_counts, args.train_size)
        print(f"\nSampling train set with stratified target train_size={args.train_size}")
        train = []
        for bucket in BUCKETS:
            n = target_counts[bucket]
            train.extend(remaining_by_bucket[bucket][:n])
            print(f"  {bucket}: selected={n}, available={available_counts[bucket]}")

    rng.shuffle(train)
    rng.shuffle(validation)

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)

    print("\nFinal split summary")
    print("-------------------")
    print(f"train: {len(train)} -> {train_path}")
    print(f"validation: {len(validation)} -> {validation_path}")

    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for sample in train:
        train_counts[_bucket_name(_total_keystroke_len(sample))] += 1
    for sample in validation:
        val_counts[_bucket_name(_total_keystroke_len(sample))] += 1

    print("\nBucket counts (train / validation)")
    for bucket in BUCKETS:
        print(f"  {bucket}: {train_counts[bucket]} / {val_counts[bucket]}")


if __name__ == "__main__":
    main()
