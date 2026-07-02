#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""paper_to_gym_jsonl.py — convert OSWorld upstream task JSON to gym JSONL

OSWorld benchmark (NeurIPS 2024, xlang-ai/OSWorld) ships 369 tasks under
`evaluation_examples/examples/<domain>/<task_id>.json`, with manifests
`test_all.json` / `test_small.json` / `test_infeasible.json` /
`test_nogdrive.json` mapping each subset to a `{domain: [task_id, ...]}`
dict.

NeMo-Gym's `ng_collect_rollouts` consumes a JSONL where each row matches the
osworld_agent client's expected schema:

    {
      "responses_create_params": {
        "input": [{"role": "user", "content": "<task instruction>"}]
      },
      "verifier_metadata": {
        "task_id": "<uuid>",
        "domain": "<domain>",
        "osworld_task": { ... full task JSON, passed verbatim to
                          DesktopEnv.reset(task_config=...) ... }
      }
    }

This script converts an OSWorld manifest into that JSONL shape.

Usage:
    paper_to_gym_jsonl.py \\
        --osworld-root /path/to/forked-osworld \\
        --manifest test_all.json \\
        --output test_all.jsonl

Or generate ALL four standard manifests at once:
    paper_to_gym_jsonl.py --osworld-root /path/to/forked-osworld --all
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


MANIFESTS = ("test_all", "test_small", "test_infeasible", "test_nogdrive")


def convert(osworld_root: pathlib.Path, manifest_name: str, out_path: pathlib.Path) -> tuple[int, dict[str, int]]:
    """Convert one OSWorld manifest into a gym-shape JSONL file.

    Returns (total_rows_written, {domain: count}).
    """
    manifest_path = osworld_root / "evaluation_examples" / f"{manifest_name}.json"
    examples_dir = osworld_root / "evaluation_examples" / "examples"
    if not manifest_path.is_file():
        sys.exit(f"manifest not found: {manifest_path}")
    if not examples_dir.is_dir():
        sys.exit(f"examples dir not found: {examples_dir}")

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        sys.exit(f"manifest {manifest_path} is not a {{domain: [task_id, ...]}} dict")

    per_domain: dict[str, int] = {}
    rows: list[dict] = []
    for domain, task_ids in manifest.items():
        for task_id in task_ids:
            task_json_path = examples_dir / domain / f"{task_id}.json"
            if not task_json_path.is_file():
                print(f"  WARN: missing task json {task_json_path}", file=sys.stderr)
                continue
            task_json = json.loads(task_json_path.read_text())
            instruction = task_json.get("instruction", "")
            row = {
                "responses_create_params": {
                    "input": [{"role": "user", "content": instruction}],
                },
                "verifier_metadata": {
                    "task_id": task_id,
                    "domain": domain,
                    "osworld_task": task_json,
                },
            }
            rows.append(row)
            per_domain[domain] = per_domain.get(domain, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), per_domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--osworld-root",
        required=True,
        type=pathlib.Path,
        help="Path to a clone of xlang-ai/OSWorld (or a fork). "
        "Must contain `evaluation_examples/{examples/,test_*.json}`.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="A single manifest name (without .json), e.g. test_all. "
        "Use --all to convert all four standard manifests.",
    )
    ap.add_argument(
        "--output", default=None, type=pathlib.Path, help="Output JSONL path (required when --manifest is given)."
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Convert all four standard manifests "
        "(test_all / test_small / test_infeasible / test_nogdrive) "
        "into the data/ dir alongside this script "
        "(i.e. responses_api_agents/osworld_agent/data/).",
    )
    args = ap.parse_args()

    if args.all == bool(args.manifest):
        sys.exit("pick exactly one of --all or --manifest")

    if args.manifest:
        if not args.output:
            sys.exit("--output is required with --manifest")
        total, per_domain = convert(args.osworld_root, args.manifest, args.output)
        print(f"✓ {args.manifest} → {args.output}: {total} rows")
        for d, n in sorted(per_domain.items()):
            print(f"    {d:25s} {n:4d}")
        return

    # --all mode: drop outputs next to data/ relative to this script
    here = pathlib.Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    print(f"Output dir: {data_dir}\n")
    grand_total = 0
    for name in MANIFESTS:
        out = data_dir / f"{name}.jsonl"
        total, per_domain = convert(args.osworld_root, name, out)
        print(f"✓ {name:18s} → {out.name}: {total} rows")
        for d, n in sorted(per_domain.items()):
            print(f"    {d:25s} {n:4d}")
        print()
        grand_total += total
    print(f"Grand total rows written: {grand_total}")


if __name__ == "__main__":
    main()
