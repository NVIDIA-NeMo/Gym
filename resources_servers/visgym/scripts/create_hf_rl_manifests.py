#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create NeMo Gym VisGym RL manifests from Hugging Face VisGym trajectories.

The HF dataset stores full trajectories, including image/history payloads.  For
online multiturn RL we only need seed/env metadata; the resource server creates
observations by resetting the live Gymnasium env.  This script therefore streams
selected JSONL rows, extracts lightweight metadata, and writes VisGym task rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "VisGym/visgym_data"
DEFAULT_REVISION = "main"
DEFAULT_MODEL = "policy_model"

LOCAL_ENV_BY_HF_ENV = {
    "colorization": "colorization/easy",
    "counting": "counting/easy",
    "fetch_pick_place": "fetch_pick_and_place/easy",
    "fetch_reach": "fetch_reach/easy",
    "jigsaw": "jigsaw/easy",
    "matchstick_equation": "matchstick_equation/easy",
    "matchstick_rotation": "matchstick_rotation/easy",
    "mental_rotation_2d": "mental_rotation_2d/easy",
    "mental_rotation_3d_cube": "mental_rotation_3d_cube/easy",
    "mental_rotation_3d_objaverse": "mental_rotation_3d_objaverse/easy",
    "patch_reassembly": "patch_reassembly/easy",
    "refdot": "refdot/easy",
    "sliding_block": "sliding_block/easy",
    "toy_maze_2d": "maze_2d/easy",
    "toy_maze_3d": "maze_3d/easy",
    "video_unshuffle": "video_unshuffle/easy",
    "zoom_in_puzzle": "zoom_in_puzzle/easy",
}

# These reset successfully in the current cluster venv without extra image
# assets or optional packages.  Other envs are intentionally available through
# --env, but may need cv2/trimesh/gymnasium_robotics/sample_dir assets first.
DEFAULT_RUNNABLE_HF_ENVS = (
    "toy_maze_2d",
    "matchstick_equation",
    "matchstick_rotation",
    "patch_reassembly",
)

PREFERRED_VARIANT_BY_HF_ENV = {
    "counting": "guess_only",
    "jigsaw": "reorder",
    "matchstick_equation": "sos",
    "video_unshuffle": "reorder",
    "zoom_in_puzzle": "reorder",
}

ACT_GRAMMAR_BY_LOCAL_ENV = {
    "maze_2d/easy": r"^\('(?:move|stop)',\s*(?:[0-3]|'stop')\)$",
    "maze_2d/hard": r"^\('(?:move|stop)',\s*(?:[0-3]|'stop')\)$",
    "maze_3d/easy": r"^\('(?:turn|move|stop)',\s*(?:-?[0-9]+|'stop')\)$",
    "maze_3d/hard": r"^\('(?:turn|move|stop)',\s*(?:-?[0-9]+|'stop')\)$",
    "matchstick_equation/easy": r"^\('(?:move|stop)',\s*(?:\[[0-9,\s-]+\]|'stop')\)$",
    "matchstick_equation/hard": r"^\('(?:move|stop)',\s*(?:\[[0-9,\s-]+\]|'stop')\)$",
    "matchstick_rotation/easy": r"^\('(?:move|stop)',\s*(?:\[[0-9.,\s-]+\]|'stop')\)$",
    "matchstick_rotation/hard": r"^\('(?:move|stop)',\s*(?:\[[0-9.,\s-]+\]|'stop')\)$",
    "patch_reassembly/easy": r"^\('(?:place|stop)',\s*(?:\([0-9,\s-]+\)|'stop')\)$",
    "patch_reassembly/hard": r"^\('(?:place|stop)',\s*(?:\([0-9,\s-]+\)|'stop')\)$",
    "jigsaw/easy": r"^\('(?:swap|reorder|stop)',\s*.+\)$",
    "jigsaw/hard": r"^\('(?:swap|reorder|stop)',\s*.+\)$",
    "sliding_block/easy": r"^\('(?:move|stop)',\s*(?:\([0-9,\s-]+\)|'stop')\)$",
    "sliding_block/hard": r"^\('(?:move|stop)',\s*(?:\([0-9,\s-]+\)|'stop')\)$",
}


def request_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "visgym-manifest-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def hf_resolve_url(repo_id: str, revision: str, rfilename: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_path = urllib.parse.quote(rfilename, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/datasets/{quoted_repo}/resolve/{quoted_revision}/{quoted_path}"


def list_hf_jsonl_files(repo_id: str, revision: str, timeout: int) -> list[str]:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    url = f"https://huggingface.co/api/datasets/{quoted_repo}/revision/{revision}"
    try:
        payload = request_json(url, timeout)
    except Exception:
        payload = request_json(f"https://huggingface.co/api/datasets/{quoted_repo}", timeout)
    siblings = payload.get("siblings") or []
    files = []
    for item in siblings:
        name = item.get("rfilename") if isinstance(item, dict) else None
        if isinstance(name, str) and name.endswith(".jsonl"):
            files.append(name)
    return sorted(files)


def parse_hf_path(path: str) -> dict[str, str] | None:
    parts = path.split("/")
    if len(parts) < 3 or not parts[-1].endswith(".jsonl"):
        return None
    hf_env = parts[0]
    if parts[1] in {"train", "val", "validation", "test"}:
        variant = ""
        split = parts[1]
        batch_name = parts[2]
    elif len(parts) >= 4:
        variant = parts[1]
        split = parts[2]
        batch_name = parts[3]
    else:
        return None
    if split == "validation":
        split = "val"
    return {"hf_env": hf_env, "variant": variant, "split": split, "batch_name": batch_name}


def read_hf_jsonl_row(repo_id: str, revision: str, path: str, timeout: int) -> dict[str, Any]:
    url = hf_resolve_url(repo_id, revision, path)
    req = urllib.request.Request(url, headers={"User-Agent": "visgym-manifest-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        line = response.readline()
    if not line:
        raise ValueError(f"{path} is empty")
    return json.loads(line.decode("utf-8"))


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def summarize_extra_state(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return {
            "type": "str",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    try:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
    except TypeError:
        text = repr(value)
        payload = text.encode("utf-8")
        return {"type": type(value).__name__, "repr_bytes": len(payload)}
    summary: dict[str, Any] = {
        "type": type(value).__name__,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if isinstance(value, dict):
        summary["keys"] = sorted(str(k) for k in value.keys())[:32]
    return summary


def build_task_row(
    source_row: dict[str, Any],
    hf_path: str,
    hf_info: dict[str, str],
    local_env_id: str,
    repo_id: str,
    revision: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    horizon_cap_max: int | None,
    task_idx: int,
) -> dict[str, Any]:
    init_args = source_row.get("init_args") if isinstance(source_row.get("init_args"), dict) else {}
    run_args = source_row.get("run_args") if isinstance(source_row.get("run_args"), dict) else {}
    stats = source_row.get("stats") if isinstance(source_row.get("stats"), dict) else {}
    seed = safe_int(source_row.get("episode_seed"), safe_int(run_args.get("seed")))
    if seed is None:
        raise ValueError(f"{hf_path} has no episode seed")

    hf_max_steps = safe_int(init_args.get("max_steps"))
    horizon_cap = hf_max_steps
    if horizon_cap is not None and horizon_cap_max is not None:
        horizon_cap = min(horizon_cap, horizon_cap_max)

    variant = hf_info["variant"]
    env_variant = f"{hf_info['hf_env']}_{variant}" if variant else hf_info["hf_env"]
    episode = source_row.get("episode")
    task_id = f"hf_{slug(env_variant)}_{hf_info['split']}_episode{episode}_seed{seed}"

    metadata = {
        "suite": "visgym",
        "source": f"hf:{repo_id}",
        "hf_repo": repo_id,
        "hf_revision": revision,
        "hf_path": hf_path,
        "hf_env": hf_info["hf_env"],
        "hf_variant": variant or None,
        "hf_split": hf_info["split"],
        "hf_batch_name": hf_info["batch_name"],
        "hf_episode": episode,
        "hf_episode_seed": seed,
        "hf_hash": source_row.get("hash"),
        "hf_env_repr": init_args.get("env_repr"),
        "hf_max_steps": hf_max_steps,
        "hf_groundtruth_strategy": init_args.get("groundtruth_strategy"),
        "hf_stats": {
            "step": stats.get("step"),
            "reward": stats.get("reward"),
            "terminated": stats.get("terminated"),
            "truncated": stats.get("truncated"),
        },
        "hf_extra_state": summarize_extra_state(source_row.get("extra_state")),
        "local_env_id": local_env_id,
        "manifest_kind": "online_multiturn_rl_seed_manifest",
    }

    row: dict[str, Any] = {
        "agent_ref": {"type": "responses_api_agents", "name": "visgym_agent"},
        "env_id": local_env_id,
        "env_kwargs": {},
        "seed": seed,
        "task_id": task_id,
        "act_grammar_regex": ACT_GRAMMAR_BY_LOCAL_ENV.get(local_env_id),
        "horizon_cap": horizon_cap,
        "task_metadata": metadata,
        "responses_create_params": {
            "model": model,
            "input": [],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "tools": [],
        },
        "task_idx": task_idx,
    }
    return {k: v for k, v in row.items() if v is not None}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            out = dict(row)
            out["task_idx"] = idx
            handle.write(json.dumps(out, separators=(",", ":"), sort_keys=False) + "\n")


def select_files(
    files: list[str],
    selected_envs: set[str],
    selected_splits: set[str],
    include_all_variants: bool,
) -> list[tuple[str, dict[str, str]]]:
    selected = []
    for path in files:
        info = parse_hf_path(path)
        if info is None:
            continue
        hf_env = info["hf_env"]
        if hf_env not in selected_envs:
            continue
        if info["split"] not in selected_splits:
            continue
        if hf_env not in LOCAL_ENV_BY_HF_ENV:
            continue
        preferred = PREFERRED_VARIANT_BY_HF_ENV.get(hf_env)
        if preferred and info["variant"] and info["variant"] != preferred and not include_all_variants:
            continue
        selected.append((path, info))
    return selected


def parse_env_selection(raw: str) -> set[str]:
    value = raw.strip()
    if value == "all":
        return set(LOCAL_ENV_BY_HF_ENV)
    if value in {"default", "runnable"}:
        return set(DEFAULT_RUNNABLE_HF_ENVS)
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = selected.difference(LOCAL_ENV_BY_HF_ENV)
    if unknown:
        raise SystemExit(f"Unknown --env value(s): {', '.join(sorted(unknown))}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--env", default="runnable", help="'runnable', 'all', or comma-separated HF env names")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to include")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", default="visgym_hf_runnable")
    parser.add_argument("--max-train-per-env", type=int, default=64)
    parser.add_argument("--max-val-per-env", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--horizon-cap-max", type=int, default=50)
    parser.add_argument("--successful-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-all-variants", action="store_true")
    parser.add_argument("--no-dedupe-seeds", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between HF row downloads")
    args = parser.parse_args()

    selected_envs = parse_env_selection(args.env)
    selected_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    split_limits = {"train": args.max_train_per_env, "val": args.max_val_per_env}

    files = list_hf_jsonl_files(args.repo_id, args.revision, args.timeout)
    if not files:
        raise SystemExit(f"No JSONL files found for {args.repo_id}@{args.revision}")

    selected_files = select_files(files, selected_envs, selected_splits, args.include_all_variants)
    if not selected_files:
        raise SystemExit("No selected HF JSONL files matched the requested env/split filters")

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_env_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    counts_by_env_split: dict[tuple[str, str], int] = defaultdict(int)
    seen_seed_keys: set[tuple[str, str, int]] = set()
    stats = defaultdict(int)

    for hf_path, hf_info in selected_files:
        hf_env = hf_info["hf_env"]
        split = hf_info["split"]
        local_env_id = LOCAL_ENV_BY_HF_ENV[hf_env]
        count_key = (hf_env, split)
        limit = split_limits.get(split, args.max_train_per_env)
        if counts_by_env_split[count_key] >= limit:
            continue

        try:
            source_row = read_hf_jsonl_row(args.repo_id, args.revision, hf_path, args.timeout)
        except Exception as exc:
            print(f"WARN failed to read {hf_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            stats["read_failed"] += 1
            continue

        source_stats = source_row.get("stats") if isinstance(source_row.get("stats"), dict) else {}
        if args.successful_only:
            reward = source_stats.get("reward")
            terminated = source_stats.get("terminated")
            if not (reward == 1.0 and terminated is True):
                stats["skipped_unsuccessful"] += 1
                continue

        seed = safe_int(source_row.get("episode_seed"))
        if seed is None:
            stats["skipped_no_seed"] += 1
            continue
        seed_key = (local_env_id, split, seed)
        if not args.no_dedupe_seeds and seed_key in seen_seed_keys:
            stats["skipped_duplicate_seed"] += 1
            continue
        seen_seed_keys.add(seed_key)

        task_idx = len(rows_by_split[split])
        task_row = build_task_row(
            source_row=source_row,
            hf_path=hf_path,
            hf_info=hf_info,
            local_env_id=local_env_id,
            repo_id=args.repo_id,
            revision=args.revision,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            horizon_cap_max=args.horizon_cap_max,
            task_idx=task_idx,
        )
        rows_by_split[split].append(task_row)
        rows_by_env_split[(hf_env, split)].append(task_row)
        counts_by_env_split[count_key] += 1
        stats["rows"] += 1
        print(f"added {split} {hf_env} seed={seed} from {hf_path}", file=sys.stderr)
        if args.sleep > 0:
            time.sleep(args.sleep)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_index: dict[str, Any] = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "selected_envs": sorted(selected_envs),
        "selected_splits": sorted(selected_splits),
        "successful_only": args.successful_only,
        "include_all_variants": args.include_all_variants,
        "horizon_cap_max": args.horizon_cap_max,
        "max_output_tokens": args.max_output_tokens,
        "files": {},
        "stats": dict(stats),
    }

    for split, rows in sorted(rows_by_split.items()):
        if not rows:
            continue
        path = args.output_dir / f"{args.output_prefix}_{split}_t{args.max_output_tokens}.jsonl"
        write_jsonl(path, rows)
        manifest_index["files"][f"combined_{split}"] = {"path": str(path), "rows": len(rows)}

    for (hf_env, split), rows in sorted(rows_by_env_split.items()):
        if not rows:
            continue
        local_env_slug = LOCAL_ENV_BY_HF_ENV[hf_env].replace("/", "_")
        path = args.output_dir / f"{local_env_slug}_hf_{split}_t{args.max_output_tokens}.jsonl"
        write_jsonl(path, rows)
        manifest_index["files"][f"{hf_env}_{split}"] = {"path": str(path), "rows": len(rows)}

    index_path = args.output_dir / f"{args.output_prefix}_manifest_index.json"
    index_path.write_text(json.dumps(manifest_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest_index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
