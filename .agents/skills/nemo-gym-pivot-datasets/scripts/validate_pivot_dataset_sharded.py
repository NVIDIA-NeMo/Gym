#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate a pivot JSONL across every core, for files too large to check in one pass.

Same per-row contract as `validate_pivot_dataset.py` -- the structural checks are imported from it
rather than restated, so there is exactly one definition of the row contract. This script only adds
the sharding, and two checks that need the Gym models: that the label self-scores 1.0, and that the
prefix is a well-formed conversation.

"Sharded", not "parallel": in this skill "parallel" means parallel *tool calls*.

Sharding is by BYTE RANGE, not by copying rows out. Each worker seeks to its offset, skips the
partial line at the boundary (its owner is the previous worker), and reads only to its own end. That
is one pass over the file in total. Copy-sharding a 100 GB dataset would rewrite 100 GB before
validating a single row.

Row counts are exact without a second pass: every worker reports the offset it actually started and
stopped at, and the merge asserts those ranges are contiguous, non-overlapping, and span the file.

Typical use, after a converter run:

    python validate_pivot_dataset_sharded.py --path pivot.jsonl --workers 32 \\
        --gym-repo /path/to/Gym --config /path/to/agent_config.yaml --expect-rows 80407
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_pivot_dataset import (  # noqa: E402
    ValidationError,
    get_path,
    parse_any_field_groups,
    print_metrics,
    validate_agent_ref,
    validate_expected_action,
    validate_responses_create_params,
)


# Populated once per worker process. The Gym imports pull in FastAPI and omegaconf, so they are paid
# once per worker rather than once per row.
_WORKER: dict[str, Any] = {}


def plan_byte_ranges(size: int, workers: int) -> list[tuple[int, int]]:
    step = size // workers + 1
    return [(index * step, min((index + 1) * step, size)) for index in range(workers) if index * step < size]


def comparator_from_config(config_path: Path):
    """Build the comparator the target config would build, so the check is not a guess."""
    import yaml

    from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
        ActionComparator,
        ToolCallComparatorConfig,
    )

    config = yaml.safe_load(config_path.read_text())
    found = []
    for block in (config or {}).values():
        servers = block.get("resources_servers") if isinstance(block, dict) else None
        for server in (servers or {}).values():
            knobs = server.get("tool_call_comparator_config")
            if knobs:
                found.append(knobs)
    if not found:
        raise ValidationError(f"no tool_call_comparator_config found in {config_path}")
    if len(found) > 1:
        # Silently taking the first would validate against knobs the dataset is not scored with.
        raise ValidationError(f"{config_path} defines {len(found)} comparator configs; point --config at one server")
    return ActionComparator(config=ToolCallComparatorConfig(**found[0])), found[0]


def action_as_response(action: dict) -> dict:
    """Render a label as the response a model would have produced to earn it."""
    calls: list[dict] = []
    message: list[dict] = []
    if action["type"] == "message":
        message = [
            {
                "id": "msg_replay",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": action["content"], "annotations": []}],
            }
        ]
    else:
        source = [action] if action["type"] == "function_call" else action["calls"]
        calls = [
            {
                "id": f"fc_replay_{index}",
                "type": "function_call",
                "call_id": f"call_replay_{index}",
                "name": call["name"],
                "arguments": call["arguments"],
                "status": "completed",
            }
            for index, call in enumerate(source)
        ]
    return {
        "id": "resp_replay",
        "created_at": 0.0,
        "model": "replay",
        "object": "response",
        "output": calls + message,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }


def check_prefix(input_items: list[dict]) -> Optional[str]:
    """Structural checks on the model-call prefix. Dependency-free."""
    open_calls: Counter = Counter()
    for item in input_items:
        kind = item.get("type")
        if kind == "function_call":
            open_calls[item.get("call_id")] += 1
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not open_calls[call_id]:
                return f"function_call_output {call_id!r} has no preceding function_call"
            open_calls[call_id] -= 1
        elif kind == "reasoning" and not item.get("id"):
            # A reasoning input item without an id is rejected by the Responses request model.
            return "a reasoning input item is missing its id"
    dangling = [call_id for call_id, count in open_calls.items() if count]
    if dangling:
        return f"tool calls left without output: {dangling[:3]}"
    if input_items and input_items[-1].get("type") == "function_call":
        # The prefix should stop at an environment item. Ending on a call is what a boundary placed
        # inside a parallel batch looks like.
        return "prefix ends on a function_call, which is what a split parallel batch looks like"
    return None


def init_worker(options: dict) -> None:
    _WORKER["options"] = options
    _WORKER["comparator"] = None
    _WORKER["request_model"] = None

    gym_repo = options.get("gym_repo")
    if gym_repo:
        sys.path.insert(0, gym_repo)
        from resources_servers.single_step_tool_use_with_argument_comparison.app import (
            SingleStepToolUseArgumentComparisonRunRequest,
        )

        _WORKER["request_model"] = SingleStepToolUseArgumentComparisonRunRequest

    config = options.get("config")
    if config:
        if not gym_repo:
            raise ValidationError("--config needs --gym-repo so the comparator can be imported")
        from nemo_gym.openai_utils import NeMoGymResponse
        from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
            extract_action,
        )

        comparator, _ = comparator_from_config(Path(config))
        _WORKER["comparator"] = comparator
        _WORKER["extract_action"] = extract_action
        _WORKER["response_model"] = NeMoGymResponse


def validate_row(row: dict, line_label: str, options: dict) -> None:
    """Raise ValidationError on the first problem. Cheapest checks first."""
    if not isinstance(row, dict):
        raise ValidationError(f"{line_label}: row must be an object")

    required = ["responses_create_params", "expected_action"]
    if options["require_agent_ref"]:
        required.append("agent_ref")
    for key in required:
        if key not in row:
            raise ValidationError(f"{line_label}: missing required key {key!r}")

    for field in options["require_field"]:
        if get_path(row, field) is None:
            raise ValidationError(f"{line_label}: missing required field {field!r}")
    for group in options["any_field_groups"]:
        if not any(get_path(row, candidate) is not None for candidate in group):
            raise ValidationError(f"{line_label}: none of required alternative fields are present: {', '.join(group)}")

    params = row["responses_create_params"]
    action = row["expected_action"]
    if "agent_ref" in row:
        validate_agent_ref(row["agent_ref"], 0, options["agent_ref"])
    tool_names = validate_responses_create_params(params, 0)
    _, expected_tool_names = validate_expected_action(action, 0)

    if options["check_tool_names"]:
        missing = sorted(set(expected_tool_names) - tool_names)
        if missing:
            raise ValidationError(f"{line_label}: expected_action uses tools absent from tools: {missing}")

    request_model = _WORKER.get("request_model")
    if request_model is not None:
        try:
            request_model(responses_create_params=params, expected_action=action)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"{line_label}: resource server rejected the row: {str(exc)[:300]}") from exc

    comparator = _WORKER.get("comparator")
    if comparator is not None:
        replayed = _WORKER["extract_action"](_WORKER["response_model"](**action_as_response(action)))
        if replayed is None:
            raise ValidationError(f"{line_label}: the label produced no action when replayed")
        from pydantic import TypeAdapter

        from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
            ExpectedAction,
        )

        result = comparator.compare_action(TypeAdapter(ExpectedAction).validate_python(action), replayed)
        if result.reward != 1.0:
            # A label that cannot score 1.0 against itself is unreachable, so the row is dead weight.
            # The usual cause is a word_count_similarity_threshold above the 0.5 ceiling.
            raise ValidationError(f"{line_label}: label self-scores {result.reward} ({result.category})")

    problem = check_prefix(params.get("input") or [])
    if problem:
        raise ValidationError(f"{line_label}: {problem}")


def validate_range(job: tuple[str, int, int]) -> dict:
    path, start, end = job
    options = _WORKER["options"]
    metrics: Counter = Counter()
    problems: list[str] = []

    with open(path, "rb") as handle:
        if start:
            # The line straddling the boundary belongs to the previous range.
            handle.seek(start - 1)
            handle.readline()
        first_offset = handle.tell()
        while handle.tell() < end:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            metrics["rows"] += 1
            try:
                row = json.loads(raw)
                validate_row(row, f"offset {offset}", options)
            except ValidationError as exc:
                metrics["failures"] += 1
                if len(problems) < options["max_problems"]:
                    problems.append(str(exc))
                continue
            except json.JSONDecodeError as exc:
                metrics["failures"] += 1
                if len(problems) < options["max_problems"]:
                    problems.append(f"offset {offset}: invalid JSON: {exc}")
                continue
            metrics["rows_ok"] += 1
            action = row["expected_action"]
            metrics["action_type", action["type"]] += 1
            if action["type"] == "function_call_batch":
                metrics["batch_size", len(action["calls"])] += 1
            if "agent_ref" in row:
                metrics["agent_ref", row["agent_ref"].get("name")] += 1
            params = row["responses_create_params"]
            metrics["input_items_total"] += len(params.get("input") or [])
            metrics["tools_total"] += len(params.get("tools") or [])
        last_offset = handle.tell()

    return {"metrics": metrics, "problems": problems, "first_offset": first_offset, "last_offset": last_offset}


def merge_ranges(results: list[dict], size: int) -> Counter:
    """Sum the shards, having proved they tile the file exactly."""
    ordered = sorted(results, key=lambda result: result["first_offset"])
    if ordered[0]["first_offset"] != 0:
        raise ValidationError(f"first shard starts at {ordered[0]['first_offset']}, not 0")
    if ordered[-1]["last_offset"] != size:
        raise ValidationError(f"last shard ends at {ordered[-1]['last_offset']}, not {size}")
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier["last_offset"] != later["first_offset"]:
            raise ValidationError(
                f"shards do not tile the file: one ends at {earlier['last_offset']}, "
                f"the next starts at {later['first_offset']}"
            )
    totals: Counter = Counter()
    for result in ordered:
        totals.update(result["metrics"])
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, required=True, help="Pivot JSONL path to validate")
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 8
    )
    parser.add_argument("--gym-repo", type=Path, help="Gym repo root; enables the resource-server request model check")
    parser.add_argument("--config", type=Path, help="Agent config; enables the label self-score check")
    parser.add_argument("--agent-ref", help="Expected row-level agent_ref.name")
    parser.add_argument("--expect-rows", type=int, help="Fail if the total row count differs")
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--no-check-tool-names", dest="check_tool_names", action="store_false")
    parser.add_argument("--no-require-agent-ref", dest="require_agent_ref", action="store_false")
    parser.add_argument("--require-field", action="append", default=[])
    parser.add_argument("--require-any-field", action="append", default=[])
    parser.set_defaults(check_tool_names=True, require_agent_ref=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = args.path.stat().st_size
    if size == 0:
        raise SystemExit(f"validation failed: {args.path} is empty")

    options = {
        "gym_repo": str(args.gym_repo) if args.gym_repo else None,
        "config": str(args.config) if args.config else None,
        "agent_ref": args.agent_ref,
        "check_tool_names": args.check_tool_names,
        "require_agent_ref": args.require_agent_ref,
        "require_field": args.require_field,
        "any_field_groups": parse_any_field_groups(args.require_any_field),
        "max_problems": args.max_problems,
    }
    ranges = plan_byte_ranges(size, args.workers)
    jobs = [(str(args.path), start, end) for start, end in ranges]

    problems: list[str] = []
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=init_worker, initargs=(options,)) as pool:
        results = list(pool.map(validate_range, jobs))
    for result in results:
        problems.extend(result["problems"])

    try:
        totals = merge_ranges(results, size)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"workers: {len(jobs)}")
    print_metrics(totals)
    print(f"rows_ok: {totals['rows_ok']}")
    print(f"failures: {totals['failures']}")

    for problem in problems[: args.max_problems]:
        print(f"  {problem}", file=sys.stderr)
    if totals["failures"]:
        print(f"validation failed: {totals['failures']} rows", file=sys.stderr)
        raise SystemExit(1)
    if args.expect_rows is not None and totals["rows"] != args.expect_rows:
        print(f"validation failed: expected {args.expect_rows} rows, found {totals['rows']}", file=sys.stderr)
        raise SystemExit(1)
    print("All rows pass.")


if __name__ == "__main__":
    main()
