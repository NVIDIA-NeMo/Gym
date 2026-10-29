#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select balanced LongFact prompts using saved three-way verdict counts."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


FIELD_PATTERNS = {
    "task_id": re.compile(r'"task_id"\s*:\s*"([^"]+)"'),
    "subset": re.compile(r'"subset"\s*:\s*"([^"]+)"'),
    "contradicted_claims": re.compile(r'"contradicted_claims"\s*:\s*([0-9]+)'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-subset", type=int, default=100)
    parser.add_argument("rollouts", type=Path, nargs="+")
    return parser.parse_args()


def atomic_write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_verdicts(paths: list[Path]) -> dict[str, tuple[str, int]]:
    verdicts: dict[str, tuple[str, int]] = {}
    missing_three_way = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                info_offset = line.find('"info":')
                if info_offset < 0:
                    raise ValueError(f"Missing info object in {path}")
                info = line[info_offset:]
                values = {name: pattern.search(info) for name, pattern in FIELD_PATTERNS.items()}
                if values["contradicted_claims"] is None:
                    missing_three_way += 1
                    continue
                if values["task_id"] is None or values["subset"] is None:
                    raise ValueError(f"Missing task identity in {path}")
                task_id = values["task_id"].group(1)
                verdict = (
                    values["subset"].group(1),
                    int(values["contradicted_claims"].group(1)),
                )
                previous = verdicts.setdefault(task_id, verdict)
                if previous != verdict:
                    raise ValueError(f"Conflicting verdicts for {task_id}")
    print(f"Loaded {len(verdicts)} three-way verdicts; skipped {missing_three_way} legacy rows")
    return verdicts


def main() -> None:
    args = parse_args()
    if args.per_subset < 1:
        raise ValueError("--per-subset must be positive")

    verdicts = load_verdicts(args.rollouts)
    selected: list[tuple[str, str, int]] = []
    for subset in ("objects", "concepts"):
        candidates = [
            (task_id, count)
            for task_id, (candidate_subset, count) in verdicts.items()
            if candidate_subset == subset and count > 0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if len(candidates) < args.per_subset:
            raise ValueError(f"Only {len(candidates)} positive-contradiction {subset} prompts are available")
        selected.extend((task_id, subset, count) for task_id, count in candidates[: args.per_subset])

    source_rows: dict[str, tuple[dict[str, object], str]] = {}
    with args.source.open() as handle:
        for line in handle:
            row = json.loads(line)
            task_id = str(row["id"])
            if task_id in source_rows:
                raise ValueError(f"Duplicate source id: {task_id}")
            source_rows[task_id] = (row, line if line.endswith("\n") else f"{line}\n")

    output_lines: list[str] = []
    manifest_lines: list[str] = []
    for rank, (task_id, subset, count) in enumerate(selected, start=1):
        try:
            row, source_line = source_rows[task_id]
        except KeyError as exc:
            raise ValueError(f"Selected id absent from source: {task_id}") from exc
        output_lines.append(source_line)
        manifest_lines.append(
            json.dumps(
                {
                    "selection_rank": rank,
                    "id": task_id,
                    "subset": subset,
                    "topic": row.get("topic"),
                    "contradicted_claims": count,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    atomic_write(args.output, output_lines)
    atomic_write(args.manifest, manifest_lines)
    for subset in ("objects", "concepts"):
        counts = [count for _, candidate_subset, count in selected if candidate_subset == subset]
        tiers = {
            "3plus": sum(count >= 3 for count in counts),
            "exact2": sum(count == 2 for count in counts),
            "exact1": sum(count == 1 for count in counts),
        }
        print(f"{subset}: {len(counts)} selected {tiers}")
    print(f"Wrote {len(output_lines)} prompts to {args.output}")
    print(f"Wrote selection metadata to {args.manifest}")


if __name__ == "__main__":
    main()
