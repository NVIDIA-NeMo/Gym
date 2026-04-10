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
#!/usr/bin/env python3
"""Augmented data generation for structured outputs RL training.

Reads ds1_verified.jsonl and produces Gym-ready training data across 6 problem
categories: direct, translation, multistep_related, multistep_unrelated,
schema_only, error_correction.

Usage:
  python 260409_augmented_sdg.py -i /path/to/ds1_verified.jsonl -o data/augmented_train.jsonl
  python 260409_augmented_sdg.py -i input.jsonl -o output.jsonl --max-total 500 --categories direct,translation
  python 260409_augmented_sdg.py -i input.jsonl -o output.jsonl --samples-per-record 1 --seed 42
"""

import argparse
import json
import random
import sys
import tomllib
from collections import Counter
from pathlib import Path

import xmltodict
import yaml


sys.path.insert(0, str(Path(__file__).parent))
from generators import ALL_GENERATORS


def parse_schema_to_dict(schema_str: str, fmt: str) -> dict:
    if fmt == "json":
        obj = json.loads(schema_str)
    elif fmt == "yaml":
        obj = yaml.safe_load(schema_str)
    elif fmt == "toml":
        obj = tomllib.loads(schema_str)
    elif fmt == "xml":
        raw = xmltodict.parse(schema_str)
        top = raw.get("schema_definition", raw)
        obj = top if isinstance(top, dict) else raw
    elif fmt == "csv":
        return _csv_schema_to_dict(schema_str)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if isinstance(obj, dict) and "schema" in obj:
        return obj["schema"]
    return obj


def _csv_schema_to_dict(schema_str: str) -> dict:
    lines = schema_str.strip().split("\n")
    headers = [h.strip() for h in lines[0].split(",")]
    types = [t.strip() for t in lines[1].split(",")] if len(lines) > 1 else ["string"] * len(headers)
    properties = {}
    for name, typ in zip(headers, types):
        properties[name] = {"type": typ if typ in ("string", "integer", "number", "boolean") else "string"}
    return {
        "type": "array",
        "items": {"type": "object", "properties": properties, "required": headers, "additionalProperties": False},
    }


def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fmt = record.get("target_output_format", "json")
            try:
                record["_json_schema"] = parse_schema_to_dict(record.get("structured_schema", ""), fmt)
            except Exception as e:
                print(f"  SKIP line {i}: schema parse failed: {e}")
                continue
            record["_record_id"] = record.get("metadata", {}).get("record_id", f"line_{i}")
            records.append(record)
    return records


def parse_weights(weights_str: str) -> dict:
    weights = {}
    for pair in weights_str.split(","):
        k, v = pair.strip().split(":")
        weights[k.strip()] = float(v.strip())
    return weights


def main():
    parser = argparse.ArgumentParser(description="Augmented SDG for structured outputs")
    parser.add_argument("-i", "--input", required=True, help="Path to ds1_verified.jsonl")
    parser.add_argument("-o", "--output", required=True, help="Path to write Gym-ready jsonl")
    parser.add_argument(
        "--categories", default=",".join(ALL_GENERATORS.keys()), help="Comma-separated categories to generate"
    )
    parser.add_argument("--target-formats", default="json,yaml,xml,toml,csv", help="Comma-separated output formats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total", type=int, default=5000, help="Hard cap on total output records")
    parser.add_argument("--max-per-category", type=int, default=1000, help="Cap per category")
    parser.add_argument("--samples-per-record", type=int, default=3, help="Augmented samples per source record")
    parser.add_argument(
        "--category-weights", default=None, help="Relative weights e.g. direct:4,translation:2,schema_only:1"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    categories = [c.strip() for c in args.categories.split(",")]
    target_formats = [f.strip() for f in args.target_formats.split(",")]

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    print(f"Loading records from {src}...")
    records = load_records(src)
    print(f"  Loaded {len(records)} records")

    if not records:
        print("No records loaded. Exiting.")
        sys.exit(1)

    all_samples = []
    category_counts = Counter()

    for cat in categories:
        if cat not in ALL_GENERATORS:
            print(f"  WARNING: unknown category '{cat}', skipping")
            continue

        gen_fn = ALL_GENERATORS[cat]
        remaining = args.max_total - len(all_samples)
        if remaining <= 0:
            break

        cap = min(args.max_per_category, remaining)
        print(f"  Generating '{cat}' (max {cap})...")

        samples = gen_fn(
            records=records,
            rng=rng,
            samples_per_record=args.samples_per_record,
            target_formats=target_formats,
            max_samples=cap,
        )
        all_samples.extend(samples)
        category_counts[cat] = len(samples)
        print(f"    -> {len(samples)} samples")

    if args.category_weights and len(all_samples) > args.max_total:
        weights = parse_weights(args.category_weights)
        total_weight = sum(weights.get(c, 1.0) for c in categories)
        budgets = {c: int(args.max_total * weights.get(c, 1.0) / total_weight) for c in categories}

        filtered = []
        by_cat = {}
        for s in all_samples:
            pt = s.get("problem_type", "unknown")
            by_cat.setdefault(pt, []).append(s)
        for cat, samples in by_cat.items():
            budget = budgets.get(cat, args.max_total // len(categories))
            rng.shuffle(samples)
            filtered.extend(samples[:budget])
        all_samples = filtered

    rng.shuffle(all_samples)
    all_samples = all_samples[: args.max_total]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print("  Augmented SDG Complete")
    print(f"{'=' * 60}")
    print(f"  Source records:  {len(records)}")
    print(f"  Total generated: {len(all_samples)}")
    print(f"  Output: {dst}")
    print()

    print("  By category:")
    final_counts = Counter(s.get("problem_type", "?") for s in all_samples)
    for cat in sorted(final_counts):
        print(f"    {cat}: {final_counts[cat]}")

    print("\n  By schema_type:")
    fmt_counts = Counter(s.get("schema_type", "?") for s in all_samples)
    for fmt in sorted(fmt_counts):
        print(f"    {fmt}: {fmt_counts[fmt]}")

    print("\n  By schema_repr:")
    repr_counts = Counter(s.get("schema_repr", "?") for s in all_samples)
    for r in sorted(repr_counts):
        print(f"    {r}: {repr_counts[r]}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
