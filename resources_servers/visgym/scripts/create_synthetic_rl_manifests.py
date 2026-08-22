#!/usr/bin/env python3
"""Create synthetic VisGym seed-sweep manifests for online multiturn RL.

These manifests do not contain offline trajectories.  Each row is a task seed
for the live VisGym resource server; the environment creates the image and
prompt at reset time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "policy_model"
DEFAULT_ENVS = (
    "maze_2d/easy",
    "matchstick_equation/easy",
    "matchstick_rotation/easy",
    "patch_reassembly/easy",
)

ENV_CONFIGS: dict[str, dict[str, Any]] = {
    "maze_2d/easy": {
        "name": "maze_2d_easy",
        "horizon_cap": 25,
        "act_grammar_regex": r"^\('(?:move|stop)',\s*(?:[0-3]|'stop')\)$",
        "hf_env": "toy_maze_2d",
        "difficulty": "easy",
    },
    "matchstick_equation/easy": {
        "name": "matchstick_equation_easy",
        "horizon_cap": 50,
        "act_grammar_regex": r"^\('(?:move|stop)',\s*(?:\[[0-9,\s-]+\]|'stop')\)$",
        "hf_env": "matchstick_equation",
        "difficulty": "easy",
    },
    "matchstick_rotation/easy": {
        "name": "matchstick_rotation_easy",
        "horizon_cap": 10,
        "act_grammar_regex": r"^\('(?:move|stop)',\s*(?:\[[0-9.,\s-]+\]|'stop')\)$",
        "hf_env": "matchstick_rotation",
        "difficulty": "easy",
    },
    "patch_reassembly/easy": {
        "name": "patch_reassembly_easy",
        "horizon_cap": 20,
        "act_grammar_regex": r"^\('(?:place|stop)',\s*(?:\([0-9,\s-]+\)|'stop')\)$",
        "hf_env": "patch_reassembly",
        "difficulty": "easy",
    },
}


def parse_envs(raw: str) -> list[str]:
    value = raw.strip()
    if value in {"default", "runnable"}:
        return list(DEFAULT_ENVS)
    envs = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(envs).difference(ENV_CONFIGS))
    if unknown:
        raise SystemExit(f"Unknown env id(s): {', '.join(unknown)}")
    return envs


def make_seeds(count: int, seed: int, used: set[int]) -> list[int]:
    rng = random.Random(seed)
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = rng.getrandbits(32)
        if candidate in used:
            continue
        used.add(candidate)
        seeds.append(candidate)
    return seeds


def make_row(
    *,
    env_id: str,
    seed: int,
    split: str,
    synthetic_index: int,
    task_idx: int,
    model: str,
    temperature: float,
    max_output_tokens: int,
    generator_seed: int,
) -> dict[str, Any]:
    cfg = ENV_CONFIGS[env_id]
    name = cfg["name"]
    return {
        "agent_ref": {"type": "responses_api_agents", "name": "visgym_agent"},
        "env_id": env_id,
        "env_kwargs": {},
        "seed": seed,
        "task_id": f"synthetic_{name}_{split}_{synthetic_index:06d}_seed{seed}",
        "act_grammar_regex": cfg["act_grammar_regex"],
        "horizon_cap": cfg["horizon_cap"],
        "task_metadata": {
            "suite": "visgym",
            "source": "synthetic:seed_sweep",
            "source_basis": "local VisGym env reset(seed=...)",
            "related_hf_env": cfg["hf_env"],
            "local_env_id": env_id,
            "difficulty": cfg["difficulty"],
            "split": split,
            "synthetic_index": synthetic_index,
            "generator_seed": generator_seed,
            "manifest_kind": "online_multiturn_rl_seed_manifest",
        },
        "responses_create_params": {
            "model": model,
            "input": [],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "tools": [],
        },
        "task_idx": task_idx,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            out = dict(row)
            out["task_idx"] = idx
            handle.write(json.dumps(out, separators=(",", ":"), sort_keys=False) + "\n")


def build_split(
    *,
    envs: list[str],
    split: str,
    samples_per_env: int,
    seed_base: int,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows_by_env: dict[str, list[dict[str, Any]]] = {}
    combined_rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for env_offset, env_id in enumerate(envs):
        env_seed_base = seed_base + env_offset * 100_000 + (0 if split == "train" else 50_000)
        seeds = make_seeds(samples_per_env, env_seed_base, used)
        env_rows = [
            make_row(
                env_id=env_id,
                seed=seed,
                split=split,
                synthetic_index=index,
                task_idx=index,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                generator_seed=env_seed_base,
            )
            for index, seed in enumerate(seeds)
        ]
        rows_by_env[env_id] = env_rows
        combined_rows.extend(env_rows)
    return rows_by_env, combined_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--envs", default="runnable", help="'runnable' or comma-separated env IDs")
    parser.add_argument("--samples-per-env", type=int, default=1280)
    parser.add_argument("--val-samples-per-env", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=20260627)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-prefix", default="visgym_synthetic_runnable")
    args = parser.parse_args()

    if args.samples_per_env < 1:
        raise SystemExit("--samples-per-env must be positive")
    if args.val_samples_per_env < 0:
        raise SystemExit("--val-samples-per-env must be non-negative")

    envs = parse_envs(args.envs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, Any] = {
        "source": "synthetic:seed_sweep",
        "envs": envs,
        "samples_per_env": args.samples_per_env,
        "val_samples_per_env": args.val_samples_per_env,
        "seed_base": args.seed_base,
        "max_output_tokens": args.max_output_tokens,
        "model": args.model,
        "temperature": args.temperature,
        "files": {},
    }

    train_by_env, train_combined = build_split(
        envs=envs,
        split="train",
        samples_per_env=args.samples_per_env,
        seed_base=args.seed_base,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    combined_train_path = (
        args.output_dir / f"{args.output_prefix}_train_{args.samples_per_env}each_t{args.max_output_tokens}.jsonl"
    )
    write_jsonl(combined_train_path, train_combined)
    index["files"]["combined_train"] = {
        "path": str(combined_train_path),
        "rows": len(train_combined),
    }

    for env_id, rows in train_by_env.items():
        env_name = ENV_CONFIGS[env_id]["name"]
        path = args.output_dir / f"{env_name}_synthetic_train_{args.samples_per_env}_t{args.max_output_tokens}.jsonl"
        write_jsonl(path, rows)
        index["files"][f"{env_name}_train"] = {"path": str(path), "rows": len(rows)}

    if args.val_samples_per_env:
        val_by_env, val_combined = build_split(
            envs=envs,
            split="val",
            samples_per_env=args.val_samples_per_env,
            seed_base=args.seed_base,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        combined_val_path = (
            args.output_dir
            / f"{args.output_prefix}_val_{args.val_samples_per_env}each_t{args.max_output_tokens}.jsonl"
        )
        write_jsonl(combined_val_path, val_combined)
        index["files"]["combined_val"] = {
            "path": str(combined_val_path),
            "rows": len(val_combined),
        }
        for env_id, rows in val_by_env.items():
            env_name = ENV_CONFIGS[env_id]["name"]
            path = (
                args.output_dir
                / f"{env_name}_synthetic_val_{args.val_samples_per_env}_t{args.max_output_tokens}.jsonl"
            )
            write_jsonl(path, rows)
            index["files"][f"{env_name}_val"] = {"path": str(path), "rows": len(rows)}

    index_path = args.output_dir / f"{args.output_prefix}_manifest_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
