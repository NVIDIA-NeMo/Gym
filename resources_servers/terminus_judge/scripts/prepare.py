#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare terminus_judge datasets from trajectory-format sources.

This script consolidates the previous 3-stage workflow into one entrypoint:
  1) convert: HF trajectory -> per-turn samples
  2) split: exact-match dedupe + stratified train/validation split
  3) smoke: collect rollouts using canonical 5-row example files by default
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from openapi_schema_validator import validate as validate_against_schema_openapi


GYM_ROOT = Path(__file__).resolve().parents[3]
TERMINUS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = TERMINUS_DIR / "data" / "openthoughts_agent_v1_sft"
DEFAULT_EXAMPLE_INPUT = TERMINUS_DIR / "data" / "example.jsonl"
DEFAULT_EXAMPLE_ROLLOUTS = TERMINUS_DIR / "data" / "example_rollouts.jsonl"

if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

from resources_servers.terminus_judge.schemas import TERMINUS_1_SCHEMA, TERMINUS_2_SCHEMA


try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - import guard for runtime env issues
    raise SystemExit(
        "Failed to import `datasets`. Run with an environment that has huggingface-datasets installed."
    ) from exc


HARNESS_MAP = {
    "terminus-1": "terminus_1",
    "terminus-2": "terminus_2",
}

SCHEMA_MAP = {
    "terminus_1": TERMINUS_1_SCHEMA,
    "terminus_2": TERMINUS_2_SCHEMA,
}

AGENT_REF = {
    "type": "responses_api_agents",
    "name": "terminus_judge_simple_agent",
}

BUCKETS = [
    "0-200",
    "200-500",
    "500-1000",
    "1000-2000",
    "2000-5000",
    "5000+",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare terminus_judge datasets and smoke rollouts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert HF trajectories to per-turn samples.")
    convert_parser.add_argument(
        "--hf_dataset",
        type=str,
        default="open-thoughts/OpenThoughts-Agent-v1-SFT",
        help="HF dataset name (used in streaming mode unless --hf_parquet_glob is provided).",
    )
    convert_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    convert_parser.add_argument(
        "--dataset_name",
        type=str,
        default="openthoughts_agent_v1_sft",
        help="Dataset name used in output UUIDs and metadata.",
    )
    convert_parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Output directory where samples.jsonl is written (default: {DEFAULT_DATASET_DIR}).",
    )
    convert_parser.add_argument(
        "--samples_filename",
        type=str,
        default="samples.jsonl",
        help="Output filename for stage-1 samples.",
    )
    convert_parser.add_argument(
        "--hf_parquet_glob",
        type=str,
        default=None,
        help="Optional parquet glob for offline conversion; enables non-streaming load.",
    )
    convert_parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Maximum number of trajectories to scan (0 means all).",
    )
    convert_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold to write on every kept sample.",
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Create exact-match deduplicated, stratified train/validation splits.",
    )
    split_parser.add_argument("--input", type=Path, required=True, help="Input samples.jsonl path.")
    split_parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for split files.")
    split_parser.add_argument(
        "--train_size",
        type=int,
        default=0,
        help="Train size after validation removal (0 means all remaining samples).",
    )
    split_parser.add_argument(
        "--val_per_bucket",
        type=int,
        default=200,
        help="Number of validation samples drawn per keystroke-length bucket.",
    )
    split_parser.add_argument(
        "--max_per_group",
        type=int,
        default=50,
        help="Maximum samples to keep per exact-match command group.",
    )
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run local rollout smoke collection (defaults to canonical 5-row example files).",
    )
    smoke_parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=DEFAULT_EXAMPLE_INPUT,
        help=f"Input JSONL for rollout collection (default: {DEFAULT_EXAMPLE_INPUT}).",
    )
    smoke_parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=DEFAULT_EXAMPLE_ROLLOUTS,
        help=f"Output rollouts JSONL path (default: {DEFAULT_EXAMPLE_ROLLOUTS}).",
    )
    smoke_parser.add_argument(
        "--agent_name",
        type=str,
        default="terminus_judge_simple_agent",
        help="Agent name passed to ng_collect_rollouts.",
    )
    smoke_parser.add_argument(
        "--num_samples_in_parallel",
        type=int,
        default=4,
        help="Parallelism for ng_collect_rollouts.",
    )
    smoke_parser.add_argument(
        "--config_paths",
        type=str,
        default=(
            "resources_servers/terminus_judge/configs/terminus_judge_simple.yaml,"
            "responses_api_models/openai_model/configs/openai_model.yaml"
        ),
        help="Comma-separated config path list used by ng_run in Hydra list syntax.",
    )
    smoke_parser.add_argument(
        "--policy_base_url",
        type=str,
        default="",
        help="Policy base URL override passed to ng_run.",
    )
    smoke_parser.add_argument(
        "--policy_api_key",
        type=str,
        default="",
        help="Policy API key override passed to ng_run.",
    )
    smoke_parser.add_argument(
        "--policy_model_name",
        type=str,
        default="",
        help="Policy model name override passed to ng_run.",
    )
    smoke_parser.add_argument("--ng_run_bin", type=str, default="ng_run", help="ng_run binary name/path.")
    smoke_parser.add_argument(
        "--ng_collect_bin",
        type=str,
        default="ng_collect_rollouts",
        help="ng_collect_rollouts binary name/path.",
    )
    smoke_parser.add_argument("--ng_status_bin", type=str, default="ng_status", help="ng_status binary name/path.")
    smoke_parser.add_argument(
        "--wait_retries",
        type=int,
        default=180,
        help="Number of status polling attempts while waiting for ng_run readiness.",
    )
    smoke_parser.add_argument(
        "--wait_interval_sec",
        type=float,
        default=5.0,
        help="Delay between readiness polling attempts.",
    )
    smoke_parser.add_argument(
        "--status_timeout_sec",
        type=int,
        default=8,
        help="Timeout for each ng_status invocation.",
    )
    smoke_parser.add_argument(
        "--expected_servers",
        type=int,
        default=3,
        help="Expected healthy server count before rollout collection starts.",
    )

    return parser.parse_args()


def _load_rows_for_convert(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.hf_parquet_glob:
        parquet_files = sorted(glob.glob(args.hf_parquet_glob))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files matched: {args.hf_parquet_glob}")
        return load_dataset("parquet", data_files=parquet_files, split=args.split, streaming=False)
    return load_dataset(args.hf_dataset, split=args.split, streaming=True)


def _detect_harness(parsed: dict[str, Any]) -> str | None:
    if "state_analysis" in parsed:
        return "terminus_1"
    if "analysis" in parsed:
        return "terminus_2"
    return None


def _safe_keystroke_stats(parsed: dict[str, Any]) -> tuple[list[int], int, int]:
    commands = parsed.get("commands", [])
    if not isinstance(commands, list):
        commands = []

    keystroke_lens: list[int] = []
    for command in commands:
        if isinstance(command, dict):
            keystrokes = command.get("keystrokes", "")
            if isinstance(keystrokes, str):
                keystroke_lens.append(len(keystrokes))
            else:
                keystroke_lens.append(len(str(keystrokes)))
        else:
            keystroke_lens.append(0)

    return keystroke_lens, sum(keystroke_lens), len(commands)


def _project_conversations_prefix(conversations: list[Any], stop_idx: int) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for message in conversations[:stop_idx]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role is None:
            continue
        projected.append(
            {
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            }
        )
    return copy.deepcopy(projected)


def run_convert(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.samples_filename

    counters = Counter()
    harness_counts = Counter()
    samples: list[dict[str, Any]] = []

    rows = _load_rows_for_convert(args)

    for row_idx, row in enumerate(rows):
        if args.max_rows > 0 and row_idx >= args.max_rows:
            break

        counters["total_traj"] += 1

        mapped_harness = HARNESS_MAP.get(row.get("agent"))
        if not mapped_harness:
            counters["unknown_harness"] += 1
            continue
        harness_counts[f"mapped_{mapped_harness}"] += 1

        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            counters["missing_conversations"] += 1
            continue

        for turn_index, message in enumerate(conversations):
            if not isinstance(message, dict):
                counters["invalid_message"] += 1
                continue
            if message.get("role") != "assistant":
                continue

            counters["total_assist_turns"] += 1

            raw_content = message.get("content")
            if not isinstance(raw_content, str):
                counters["invalid_assistant_content"] += 1
                continue

            stripped = raw_content.split("</think>")[-1].strip()

            try:
                parsed = json.loads(stripped)
            except Exception:
                counters["json_parse_failed"] += 1
                continue

            if not isinstance(parsed, dict):
                counters["json_not_object"] += 1
                continue

            detected_harness = _detect_harness(parsed)
            if detected_harness is None:
                counters["unknown_schema"] += 1
                continue

            if detected_harness != mapped_harness:
                counters["mismatched_harness"] += 1
                continue

            try:
                validate_against_schema_openapi(parsed, SCHEMA_MAP[detected_harness])
            except Exception:
                counters["schema_invalid"] += 1
                continue

            keystroke_lens, total_keystroke_len, num_commands = _safe_keystroke_stats(parsed)
            input_prefix = _project_conversations_prefix(conversations, stop_idx=turn_index)

            metadata: dict[str, Any] = {
                "harness": detected_harness,
                "dataset_name": args.dataset_name,
                "row_idx": row_idx,
                "turn_index": turn_index,
                "run_id": row.get("run_id"),
                "trial_name": row.get("trial_name"),
                "task": row.get("task"),
                "episode": row.get("episode"),
                "num_commands": num_commands,
                "keystroke_lens": keystroke_lens,
                "total_keystroke_len": total_keystroke_len,
                "category": "first_round" if len(input_prefix) == 1 else "others",
            }

            if "task_complete" in parsed:
                metadata["task_complete"] = parsed["task_complete"]
            elif "is_task_complete" in parsed:
                metadata["task_complete"] = parsed["is_task_complete"]

            sample: dict[str, Any] = {
                "uuid": f"{args.dataset_name}_{row_idx}_turn_{turn_index}",
                "responses_create_params": {
                    "input": input_prefix,
                },
                "expected_answer": stripped,
                "metadata": metadata,
                "agent_ref": copy.deepcopy(AGENT_REF),
            }
            if args.threshold is not None:
                sample["threshold"] = args.threshold

            samples.append(sample)
            counters["kept_samples"] += 1
            harness_counts[f"kept_{detected_harness}"] += 1

        if (row_idx + 1) % 1000 == 0:
            print(f"Processed {row_idx + 1} trajectories...")

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    print("\nConversion summary")
    print("==================")
    print(f"total_traj: {counters['total_traj']}")
    print(f"total_assist_turns: {counters['total_assist_turns']}")
    print(f"kept_samples: {counters['kept_samples']}")

    failure_keys = [
        "unknown_harness",
        "missing_conversations",
        "invalid_message",
        "invalid_assistant_content",
        "json_parse_failed",
        "json_not_object",
        "unknown_schema",
        "mismatched_harness",
        "schema_invalid",
    ]
    print("\nFailure counters")
    print("----------------")
    for key in failure_keys:
        print(f"{key}: {counters[key]}")

    print("\nPer-harness counts")
    print("------------------")
    for key in sorted(harness_counts):
        print(f"{key}: {harness_counts[key]}")

    print(f"\nWrote {len(samples)} samples to {output_path}")
    return output_path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_keystrokes(sample: dict[str, Any]) -> list[str]:
    expected_answer = sample.get("expected_answer", "{}")
    if isinstance(expected_answer, str):
        try:
            expected_answer = json.loads(expected_answer)
        except json.JSONDecodeError:
            return []
    if not isinstance(expected_answer, dict):
        return []

    commands = expected_answer.get("commands", [])
    if not isinstance(commands, list):
        return []

    keystrokes: list[str] = []
    for command in commands:
        if isinstance(command, dict) and "keystrokes" in command:
            value = command.get("keystrokes")
            keystrokes.append(value if isinstance(value, str) else str(value))
    return keystrokes


def _group_key(sample: dict[str, Any]) -> str:
    key_payload = {"keystrokes": _extract_keystrokes(sample)}
    return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))


def _total_keystroke_len(sample: dict[str, Any]) -> int:
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("total_keystroke_len")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)

    return sum(len(k) for k in _extract_keystrokes(sample))


def _bucket_name(length: int) -> str:
    if length < 200:
        return "0-200"
    if length < 500:
        return "200-500"
    if length < 1000:
        return "500-1000"
    if length < 2000:
        return "1000-2000"
    if length < 5000:
        return "2000-5000"
    return "5000+"


def _stratified_counts(available: dict[str, int], requested_total: int) -> dict[str, int]:
    total_available = sum(available.values())
    if requested_total <= 0 or total_available == 0:
        return {bucket: 0 for bucket in BUCKETS}
    if requested_total >= total_available:
        return dict(available)

    raw_targets = {bucket: (available[bucket] / total_available) * requested_total for bucket in BUCKETS}
    assigned = {bucket: min(available[bucket], int(raw_targets[bucket])) for bucket in BUCKETS}
    used = sum(assigned.values())

    remainders = sorted(
        BUCKETS,
        key=lambda bucket: (raw_targets[bucket] - int(raw_targets[bucket])),
        reverse=True,
    )

    idx = 0
    while used < requested_total and idx < len(remainders) * 4:
        bucket = remainders[idx % len(remainders)]
        if assigned[bucket] < available[bucket]:
            assigned[bucket] += 1
            used += 1
        idx += 1

    return assigned


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run_split(args: argparse.Namespace) -> tuple[Path, Path]:
    import random

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading samples from {args.input}")
    samples = _load_jsonl(args.input)
    print(f"Loaded {len(samples)} samples")

    print("\nGrouping by exact command-content key...")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    print(f"Unique groups: {len(groups)}")

    print(f"\nApplying max_per_group={args.max_per_group}...")
    kept_samples: list[dict[str, Any]] = []
    truncated_groups = 0
    dropped_samples = 0
    for group_samples in groups.values():
        if len(group_samples) > args.max_per_group:
            rng.shuffle(group_samples)
            dropped_samples += len(group_samples) - args.max_per_group
            truncated_groups += 1
            group_samples = group_samples[: args.max_per_group]
        kept_samples.extend(group_samples)
    print(f"Groups truncated: {truncated_groups}")
    print(f"Samples dropped by cap: {dropped_samples}")
    print(f"Samples after cap: {len(kept_samples)}")

    print("\nBucketing by metadata.total_keystroke_len...")
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in kept_samples:
        bucket = _bucket_name(_total_keystroke_len(sample))
        by_bucket[bucket].append(sample)

    for bucket in BUCKETS:
        print(f"  {bucket}: {len(by_bucket[bucket])}")

    print(f"\nSampling validation set (val_per_bucket={args.val_per_bucket})...")
    validation: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        bucket_samples = list(by_bucket[bucket])
        rng.shuffle(bucket_samples)
        n_val = min(args.val_per_bucket, len(bucket_samples))
        validation.extend(bucket_samples[:n_val])
        remaining.extend(bucket_samples[n_val:])
        print(f"  {bucket}: validation={n_val}, remaining={len(bucket_samples) - n_val}")

    remaining_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in remaining:
        remaining_by_bucket[_bucket_name(_total_keystroke_len(sample))].append(sample)

    for bucket in BUCKETS:
        rng.shuffle(remaining_by_bucket[bucket])

    if args.train_size == 0:
        print("\ntrain_size=0 -> using all remaining samples for train")
        train = list(remaining)
    else:
        available_counts = {bucket: len(remaining_by_bucket[bucket]) for bucket in BUCKETS}
        target_counts = _stratified_counts(available_counts, args.train_size)
        print(f"\nSampling train set with stratified target train_size={args.train_size}")
        train = []
        for bucket in BUCKETS:
            n = target_counts[bucket]
            train.extend(remaining_by_bucket[bucket][:n])
            print(f"  {bucket}: selected={n}, available={available_counts[bucket]}")

    rng.shuffle(train)
    rng.shuffle(validation)

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)

    print("\nFinal split summary")
    print("-------------------")
    print(f"train: {len(train)} -> {train_path}")
    print(f"validation: {len(validation)} -> {validation_path}")

    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for sample in train:
        train_counts[_bucket_name(_total_keystroke_len(sample))] += 1
    for sample in validation:
        val_counts[_bucket_name(_total_keystroke_len(sample))] += 1

    print("\nBucket counts (train / validation)")
    for bucket in BUCKETS:
        print(f"  {bucket}: {train_counts[bucket]} / {val_counts[bucket]}")

    return train_path, validation_path


def _status_is_ready(status_output: str, expected_count: int) -> bool:
    expected_line = f"{expected_count} servers found ({expected_count} healthy, 0 unhealthy)"
    return expected_line in status_output


def _wait_for_server(
    expected_count: int,
    run_log_path: Path,
    ng_status_bin: str,
    wait_retries: int,
    wait_interval_sec: float,
    status_timeout_sec: int,
    proc: subprocess.Popen[Any],
) -> bool:
    for _ in range(wait_retries):
        if run_log_path.exists():
            log_text = run_log_path.read_text(encoding="utf-8", errors="replace")
            if f"All {expected_count} / {expected_count} servers ready!" in log_text:
                return True
            if "finished unexpectedly" in log_text:
                return False

        if proc.poll() is not None:
            return False

        try:
            status = subprocess.run(
                [ng_status_bin],
                capture_output=True,
                text=True,
                timeout=status_timeout_sec,
                check=False,
            )
            if _status_is_ready(status.stdout + "\n" + status.stderr, expected_count):
                return True
        except subprocess.TimeoutExpired:
            pass

        time.sleep(wait_interval_sec)
    return False


def _tail(path: Path, n_lines: int = 120) -> str:
    if not path.exists():
        return f"[missing] {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def run_smoke(args: argparse.Namespace) -> Path:
    input_path = args.input_jsonl
    output_path = args.output_jsonl

    if not input_path.exists():
        raise FileNotFoundError(f"Missing smoke input: {input_path}")

    for tool in (args.ng_run_bin, args.ng_collect_bin, args.ng_status_bin):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required binary not found on PATH: {tool}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_log = output_path.with_name(f"{output_path.stem}_ng_run.log")
    collect_log = output_path.with_name(f"{output_path.stem}_ng_collect.log")

    if output_path.exists():
        output_path.unlink()
    if run_log.exists():
        run_log.unlink()
    if collect_log.exists():
        collect_log.unlink()

    run_cmd = [
        args.ng_run_bin,
        f"+config_paths=[{args.config_paths}]",
        f"+policy_base_url={args.policy_base_url}",
        f"+policy_api_key={args.policy_api_key}",
        f"+policy_model_name={args.policy_model_name}",
    ]
    collect_cmd = [
        args.ng_collect_bin,
        f"+agent_name={args.agent_name}",
        f"+input_jsonl_fpath={input_path}",
        f"+output_jsonl_fpath={output_path}",
        f"+num_samples_in_parallel={args.num_samples_in_parallel}",
    ]

    proc: subprocess.Popen[Any] | None = None
    run_log_file = run_log.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            run_cmd,
            cwd=GYM_ROOT,
            stdout=run_log_file,
            stderr=subprocess.STDOUT,
        )

        ready = _wait_for_server(
            expected_count=args.expected_servers,
            run_log_path=run_log,
            ng_status_bin=args.ng_status_bin,
            wait_retries=args.wait_retries,
            wait_interval_sec=args.wait_interval_sec,
            status_timeout_sec=args.status_timeout_sec,
            proc=proc,
        )
        if not ready:
            try:
                status_dump = subprocess.run(
                    [args.ng_status_bin],
                    capture_output=True,
                    text=True,
                    timeout=args.status_timeout_sec,
                    check=False,
                )
                print(status_dump.stdout)
                print(status_dump.stderr)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(f"ng_run did not become ready. See {run_log}")

        with collect_log.open("w", encoding="utf-8") as collect_log_file:
            subprocess.run(
                collect_cmd,
                cwd=GYM_ROOT,
                stdout=collect_log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except Exception:
        print("---- ng_run log (tail) ----")
        print(_tail(run_log, n_lines=200))
        if collect_log.exists():
            print("---- ng_collect log (tail) ----")
            print(_tail(collect_log, n_lines=200))
        raise
    finally:
        run_log_file.close()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    print(f"Smoke rollouts complete: {output_path}")
    print(f"ng_run log: {run_log}")
    print(f"ng_collect_rollouts log: {collect_log}")
    return output_path


def main() -> None:
    args = parse_args()
    if args.command == "convert":
        run_convert(args)
        return
    if args.command == "split":
        run_split(args)
        return
    if args.command == "smoke":
        run_smoke(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
