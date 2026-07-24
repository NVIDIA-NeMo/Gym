# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert domain-generation Gym rollouts into policy/tool-generation Gym inputs."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal


POLICY_TOOL_AGENT_REF = {
    "type": "responses_api_agents",
    "name": "conversational_tool_use_policy_tool_generation",
}
GenerationProfile = Literal["general", "proactive"]


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: rollout row must be a JSON object")
            yield line_number, value


def _rollout_candidates(row: dict[str, Any], *, source: Path, line_number: int) -> list[Any]:
    result = row.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
        raise ValueError(f"{source}:{line_number}: rollout row is missing result.candidates")
    return result["candidates"]


def materialize_policy_tool_rows(
    rollouts: Iterable[tuple[int, dict[str, Any]]],
    *,
    source: Path,
    profile: GenerationProfile,
    shuffle_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Preserve candidate objects while applying casefold-only first-wins deduplication."""
    seen_names: set[str] = set()
    rows: list[dict[str, Any]] = []

    for line_number, rollout in rollouts:
        for candidate_index, candidate in enumerate(
            _rollout_candidates(rollout, source=source, line_number=line_number)
        ):
            if not isinstance(candidate, dict):
                raise ValueError(f"{source}:{line_number}: result.candidates[{candidate_index}] must be a JSON object")
            name = candidate.get("name")
            if not isinstance(name, str):
                raise ValueError(f"{source}:{line_number}: result.candidates[{candidate_index}].name must be a string")

            dedup_key = name.casefold()
            if dedup_key in seen_names:
                continue
            seen_names.add(dedup_key)
            rows.append(
                {
                    "responses_create_params": {"input": []},
                    "domain": candidate,
                    "profile": profile,
                    "agent_ref": POLICY_TOOL_AGENT_REF.copy(),
                }
            )

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize policy/tool Gym inputs from domain-generation rollout JSONL."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Domain Gym rollout JSONL")
    parser.add_argument("--output-file", type=Path, required=True, help="Policy/tool Gym input JSONL")
    parser.add_argument(
        "--profile",
        choices=("general", "proactive"),
        required=True,
        help="Policy/tool generation profile stamped on every output row",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle deduplicated rows with this explicit integer seed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = materialize_policy_tool_rows(
        read_jsonl(args.input_file),
        source=args.input_file,
        profile=args.profile,
        shuffle_seed=args.shuffle_seed,
    )
    write_jsonl(args.output_file, rows)
    print(f"Wrote {len(rows)} policy/tool input rows to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
