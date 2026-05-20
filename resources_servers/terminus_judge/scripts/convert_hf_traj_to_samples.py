#!/usr/bin/env python3
"""Convert HF trajectory datasets to terminus_judge per-turn samples.

This script supports:
1) Streaming load from a Hugging Face dataset name
2) Non-streaming load from local parquet files via glob
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from openapi_schema_validator import validate as validate_against_schema_openapi


GYM_ROOT = Path(__file__).resolve().parents[3]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF trajectories into terminus_judge per-turn samples.")
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="open-thoughts/OpenThoughts-Agent-v1-SFT",
        help="HF dataset name (used in streaming mode unless --hf_parquet_glob is provided).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openthoughts_agent_v1_sft",
        help="Dataset name used in output UUIDs and metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory where samples.jsonl is written.",
    )
    parser.add_argument(
        "--hf_parquet_glob",
        type=str,
        default=None,
        help="Optional parquet glob for offline conversion; enables non-streaming load.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Maximum number of trajectories to scan (0 means all).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold to write on every kept sample.",
    )
    return parser.parse_args()


def _load_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "samples.jsonl"

    counters = Counter()
    harness_counts = Counter()
    samples: list[dict[str, Any]] = []

    rows = _load_rows(args)

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


if __name__ == "__main__":
    main()
