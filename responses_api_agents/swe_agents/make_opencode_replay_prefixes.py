#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build replay-ready OpenCode prefixes from completed Gym rollout JSONL."""

import argparse
import json
from pathlib import Path

from responses_api_agents.swe_agents.opencode_replay import build_replay_prefix_row


def _parse_source_lines(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        line_number = int(part)
        if line_number < 1:
            raise argparse.ArgumentTypeError("source line numbers are one-based and must be positive")
        result.add(line_number)
    if not result:
        raise argparse.ArgumentTypeError("provide at least one source line")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="Completed Gym rollout JSONL")
    parser.add_argument("output_jsonl", type=Path, help="Replay-prefix JSONL to create")
    parser.add_argument(
        "--source-lines",
        type=_parse_source_lines,
        help="Comma-separated one-based source lines; otherwise scan every row",
    )
    parser.add_argument(
        "--strategy",
        choices=("first-task-batch", "last-tool-turn"),
        default="first-task-batch",
        help="Where to end each main-agent prefix (default: first completed task batch)",
    )
    parser.add_argument("--limit", type=int, help="Stop after writing this many valid prefixes")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Report and skip rows without a valid causal prefix instead of failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not args.input_jsonl.is_file():
        raise SystemExit(f"input JSONL does not exist: {args.input_jsonl}")
    if args.input_jsonl.resolve() == args.output_jsonl.resolve():
        raise SystemExit("input and output JSONL paths must differ")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    selected = args.source_lines
    found: set[int] = set()
    written = 0
    skipped = 0
    with args.input_jsonl.open() as source, args.output_jsonl.open("w") as destination:
        for line_number, line in enumerate(source, 1):
            if selected is not None and line_number not in selected:
                if line_number > max(selected) and found == selected:
                    break
                continue
            if selected is not None:
                found.add(line_number)
            try:
                row = json.loads(line)
                prefix = build_replay_prefix_row(
                    row,
                    source_line=line_number,
                    strategy=args.strategy,
                    strict=True,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                if not args.skip_invalid:
                    raise SystemExit(f"line {line_number}: {error}") from error
                skipped += 1
                print(f"skipping line {line_number}: {error}")
                continue
            destination.write(json.dumps(prefix, separators=(",", ":")) + "\n")
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    missing = sorted((selected or set()) - found)
    if missing:
        raise SystemExit(f"requested source lines do not exist: {missing}")
    if written == 0:
        args.output_jsonl.unlink(missing_ok=True)
        raise SystemExit("no replay prefixes were written")
    print(f"wrote {written} replay prefix(es) to {args.output_jsonl} ({skipped} skipped)")


if __name__ == "__main__":
    main()
