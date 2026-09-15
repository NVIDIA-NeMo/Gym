#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""convert_osworld_tasks.py — convert OSWorld upstream task JSON to Gym JSONL

OSWorld benchmark (NeurIPS 2024, xlang-ai/OSWorld) ships 369 tasks under
`evaluation_examples/examples/<domain>/<task_id>.json`, with manifests
`test_all.json` / `test_small.json` / `test_infeasible.json` /
`test_nogdrive.json` mapping each subset to a `{domain: [task_id, ...]}`
dict.

NeMo Gym's `gym eval run` consumes a JSONL where each row matches the
osworld_agent client's expected schema:

    {
      "responses_create_params": {
        "input": [{"role": "user", "content": "<task instruction>"}]
      },
      "agent_ref": {"name": "osworld_simple_agent"},
      "verifier_metadata": {
        "task_id": "<uuid>",
        "domain": "<domain>",
        "osworld_task": { ... full task JSON, passed verbatim to
                          DesktopEnv.reset(task_config=...) ... }
      }
    }

This script converts an OSWorld manifest into that JSONL shape.

Usage:
    convert_osworld_tasks.py \\
        --osworld-root /path/to/forked-osworld \\
        --manifest test_all.json \\
        --output test_all.jsonl

Or generate ALL four standard manifests at once:
    convert_osworld_tasks.py --osworld-root /path/to/forked-osworld --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


MANIFESTS = ("test_all", "test_small", "test_infeasible", "test_nogdrive")
DEFAULT_AGENT_NAME = "osworld_simple_agent"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    osworld_root: pathlib.Path,
    manifest_name: str,
    out_path: pathlib.Path,
    *,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> tuple[int, dict[str, int]]:
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
    if not agent_name.strip():
        raise ValueError("agent_name must not be empty")

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
                "agent_ref": {"name": agent_name},
                "verifier_metadata": {
                    "task_id": task_id,
                    "domain": domain,
                    "osworld_task": task_json,
                },
            }
            rows.append(row)
            per_domain[domain] = per_domain.get(domain, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    source_commit = None
    git_dir = osworld_root / ".git"
    if git_dir.exists():
        result = subprocess.run(
            ["git", "-C", str(osworld_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            source_commit = result.stdout.strip()
    provenance = {
        "schema_version": 1,
        "osworld_root": str(osworld_root.resolve()),
        "osworld_commit": source_commit,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "output": str(out_path.resolve()),
        "output_sha256": _sha256(out_path),
        "rows": len(rows),
        "task_ids_sha256": hashlib.sha256(
            "\n".join(str(row["verifier_metadata"]["task_id"]) for row in rows).encode()
        ).hexdigest(),
    }
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help=f"Gym Responses API agent name written into every row (default: {DEFAULT_AGENT_NAME}).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Convert all four standard manifests "
        "(test_all / test_small / test_infeasible / test_nogdrive) "
        "into the data/ dir alongside this script "
        "(i.e. benchmarks/osworld/data/).",
    )
    args = ap.parse_args()

    if args.all == bool(args.manifest):
        sys.exit("pick exactly one of --all or --manifest")

    if args.manifest:
        if not args.output:
            sys.exit("--output is required with --manifest")
        total, per_domain = convert(
            args.osworld_root,
            args.manifest,
            args.output,
            agent_name=args.agent_name,
        )
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
        total, per_domain = convert(
            args.osworld_root,
            name,
            out,
            agent_name=args.agent_name,
        )
        print(f"✓ {name:18s} → {out.name}: {total} rows")
        for d, n in sorted(per_domain.items()):
            print(f"    {d:25s} {n:4d}")
        print()
        grand_total += total
    print(f"Grand total rows written: {grand_total}")


if __name__ == "__main__":
    main()
