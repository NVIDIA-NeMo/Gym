#!/usr/bin/env python3
"""Materialize direct-only RDKit Chemistry data for NeMo RL training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

AGENT_REF = {"type": "responses_api_agents", "name": "rdkit_chemistry_direct_agent"}
SOURCE_REL = Path("data/rdkit-chemistry-no-tool/prepared-train1024-test1000")
OUTPUT_REL = Path("cluster/rdkit_no_tool_grpo/data")
SMOKE_LIMITS = {"train": 64, "test": 128}
DEFAULT_MAX_OUTPUT_TOKENS = 32768


def gym_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_max_output_tokens() -> int:
    raw = os.environ.get("MAX_OUTPUT_TOKENS") or os.environ.get("MAX_NEW_TOKENS")
    if raw is None:
        return DEFAULT_MAX_OUTPUT_TOKENS
    value = int(raw)
    if value <= 0:
        raise ValueError("max output tokens must be positive")
    return value


def convert_split(
    root: Path,
    source_dir: Path,
    split: str,
    max_output_tokens: int,
) -> int:
    source = source_dir / f"{split}.jsonl"
    output_dir = root / OUTPUT_REL
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{split}.jsonl"

    smoke_output = output_dir / f"{split}_smoke.jsonl"
    smoke_limit = SMOKE_LIMITS[split]
    count = 0
    with source.open() as src, output.open("w") as dst:
        smoke_rows = []
        for line_number, line in enumerate(src, start=1):
            row = json.loads(line)
            if row.get("method") != "direct":
                raise ValueError(f"{source}:{line_number} method is not direct")
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{source}:{line_number} missing non-empty prompt")

            row = dict(row)
            row["responses_create_params"] = {
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": max_output_tokens,
                "tools": [],
            }
            row["agent_ref"] = dict(AGENT_REF)
            row_str = json.dumps(row, sort_keys=True) + "\n"
            dst.write(row_str)
            if len(smoke_rows) < smoke_limit:
                smoke_rows.append(row_str)
            count += 1

    with smoke_output.open("w") as smoke_dst:
        smoke_dst.writelines(smoke_rows)

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=gym_root())
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Prepared Inferno split directory (defaults to <root>/data/...)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=default_max_output_tokens())
    args = parser.parse_args()
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")

    root = args.root.absolute()
    source_dir = (
        args.source_dir.absolute() if args.source_dir is not None else root / SOURCE_REL
    )
    train_count = convert_split(root, source_dir, "train", args.max_output_tokens)
    test_count = convert_split(root, source_dir, "test", args.max_output_tokens)
    print(
        f"Wrote {train_count} train rows and {test_count} test rows under {root / OUTPUT_REL} "
        f"with max_output_tokens={args.max_output_tokens}; validation uses each test row once"
    )


if __name__ == "__main__":
    main()
