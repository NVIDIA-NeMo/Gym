#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create an ordered maze-size curriculum for online VisGym RL.

Rows are grouped by ascending maze size. NeMo-RL must use ``data.shuffle=false``
to consume the combined manifest as a curriculum. Separate per-stage manifests
are also emitted for explicit stage scheduling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTION_GRAMMAR = r"^\('(?:move|stop)',\s*(?:[0-3]|'stop')\)$"
DEFAULT_MODEL = "policy_model"
HORIZON_CAP_BY_SIZE = {
    5: 8,
    7: 12,
    9: 25,
    11: 35,
}


def horizon_cap_for_size(size: int) -> int:
    """Return the tuned cap for the default curriculum, with a safe fallback."""
    return HORIZON_CAP_BY_SIZE.get(size, 2 * (size - 1))


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one maze size is required")
    if any(size < 3 or size % 2 == 0 for size in sizes):
        raise argparse.ArgumentTypeError("maze sizes must be odd integers >= 3")
    if sizes != sorted(set(sizes)):
        raise argparse.ArgumentTypeError("maze sizes must be unique and strictly increasing")
    return sizes


def make_row(
    *,
    size: int,
    stage: int,
    curriculum_name: str,
    num_stages: int,
    seed: int,
    stage_index: int,
    samples_per_stage: int,
    model: str,
    temperature: float,
    max_output_tokens: int,
    reward_shaping_scale: float = 0.0,
) -> dict[str, Any]:
    horizon_cap = horizon_cap_for_size(size)
    return {
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "visgym_agent",
        },
        "env_id": "maze_2d/easy",
        "env_kwargs": {"maze_width": size, "maze_height": size},
        "seed": seed,
        "task_id": (f"maze_2d_easy_curriculum_stage{stage}_{size}x{size}_seed{seed}"),
        "act_grammar_regex": ACTION_GRAMMAR,
        "horizon_cap": horizon_cap,
        "task_metadata": {
            # Terminal reward alone is too sparse to learn from: the maze pays
            # 1.0 only for ('stop', 'stop') on the target and 0.0 for every
            # move, so a GRPO group of rollouts ties at zero and the advantage
            # is degenerate. distance_delta pays the per-step change in
            # info["distance"], which the server adds on top of the raw reward.
            **(
                {
                    "reward_shaping": {
                        "type": "distance_delta",
                        "info_key": "distance",
                        "scale": reward_shaping_scale,
                    }
                }
                if reward_shaping_scale
                else {}
            ),
            "suite": "visgym",
            "task_family": "maze_2d",
            "difficulty": "easy",
            "split": "train",
            "maze_size": f"{size}x{size}",
            "curriculum_name": curriculum_name,
            "curriculum_order": "ascending_maze_size",
            "curriculum_stage": stage,
            "curriculum_num_stages": num_stages,
            "curriculum_stage_index": stage_index,
            "curriculum_samples_per_stage": samples_per_stage,
            "manifest_kind": "online_multiturn_rl_curriculum_seed_manifest",
        },
        "responses_create_params": {
            "model": model,
            "input": [],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "tools": [],
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task_idx, row in enumerate(rows):
            output = dict(row)
            output["task_idx"] = task_idx
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


def main() -> int:
    default_output_dir = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("5,7,9,11"))
    parser.add_argument("--samples-per-stage", type=int, default=1280)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reward-shaping-scale",
        type=float,
        default=0.0,
        help=(
            "Per-step distance_delta shaping coefficient. 0 (default) keeps the "
            "environment's terminal-only reward. A positive value pays "
            "scale * (previous_distance - current_distance) on every step, which is "
            "what keeps GRPO advantages from collapsing to zero on a task that only "
            "scores a correct stop on the target."
        ),
    )
    args = parser.parse_args()

    if args.samples_per_stage < 1:
        parser.error("--samples-per-stage must be positive")
    if args.seed_stride < args.samples_per_stage:
        parser.error("--seed-stride must be at least --samples-per-stage to keep stage seed ranges disjoint")
    if args.reward_shaping_scale < 0:
        parser.error("--reward-shaping-scale must be >= 0")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")

    sizes: list[int] = args.sizes
    size_slug = "_".join(f"{size}x{size}" for size in sizes)
    curriculum_name = "maze_size_" + "_".join(str(size) for size in sizes)
    # Suffix carries the shaping coefficient for the same reason the token
    # budget is in the name: two manifests that differ only in reward would
    # otherwise be indistinguishable on disk.
    shaping_suffix = f"_s{args.reward_shaping_scale:g}".replace(".", "p") if args.reward_shaping_scale else ""
    prefix = f"maze_2d_easy_curriculum_{size_slug}"
    stage_files: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    for stage_offset, size in enumerate(sizes):
        stage = stage_offset + 1
        stage_seed_base = args.seed_base + stage_offset * args.seed_stride
        stage_rows = [
            make_row(
                size=size,
                stage=stage,
                curriculum_name=curriculum_name,
                num_stages=len(sizes),
                seed=stage_seed_base + stage_index,
                stage_index=stage_index,
                samples_per_stage=args.samples_per_stage,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                reward_shaping_scale=args.reward_shaping_scale,
            )
            for stage_index in range(args.samples_per_stage)
        ]
        stage_path = args.output_dir / (
            f"{prefix}_stage{stage}_{size}x{size}_{args.samples_per_stage}_t{args.max_output_tokens}{shaping_suffix}.jsonl"
        )
        write_jsonl(stage_path, stage_rows)
        stage_files.append(
            {
                "stage": stage,
                "maze_size": f"{size}x{size}",
                "horizon_cap": horizon_cap_for_size(size),
                "seed_start": stage_seed_base,
                "seed_end": stage_seed_base + args.samples_per_stage - 1,
                "rows": len(stage_rows),
                "path": f"./{stage_path.name}",
            }
        )
        combined_rows.extend(stage_rows)

    combined_path = args.output_dir / (
        f"{prefix}_{args.samples_per_stage}each_t{args.max_output_tokens}{shaping_suffix}.jsonl"
    )
    write_jsonl(combined_path, combined_rows)

    index = {
        "curriculum_name": curriculum_name,
        "curriculum_order": "ascending_maze_size",
        "shuffle_required": False,
        "samples_per_stage": args.samples_per_stage,
        "total_rows": len(combined_rows),
        "combined": {"path": f"./{combined_path.name}", "rows": len(combined_rows)},
        "stages": stage_files,
    }
    # Same suffixes as the manifests it points at. Without them a second
    # variant silently overwrites the first index, and whoever reads it to
    # pick stage files gets the wrong token budget or shaping coefficient.
    index_path = args.output_dir / f"{prefix}_manifest_index_t{args.max_output_tokens}{shaping_suffix}.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
