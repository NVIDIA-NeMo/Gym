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
"""Check that pivot rows split a trajectory on whole model calls, and that none leaks its own label.

A rollout artifact stores one flat list of output items, so "one model call" has to be
reconstructed before it can become a pivot. Get that wrong and the damage is silent: a parallel tool
call batch gets cut in half, or the reasoning that produced the answer ends up in the prefix that is
supposed to precede it. Both still validate as JSON and still train.

Two modes:

  --source-path only          Source diagnosis. Segments each trajectory two independent ways and
                              reports where they disagree, plus reasoning placement anomalies. Run
                              this BEFORE writing a converter, and run it on two different models
                              through the same harness -- an anomaly that appears for one model and
                              not the other is an artifact of the recording, not model behaviour.

  --source-path --pivot-path  Full audit of a generated pivot file against its source.

Nothing here assumes a particular producer's field names or call-id format. Every source-shape
assumption is a flag, because hardcoding one is exactly the bug this script exists to catch.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_pivot_dataset import get_path, iter_jsonl  # noqa: E402


SKIPPED = "SKIPPED"


# ----------------------------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------------------------


def segment_by_execution_run(items: list[dict]) -> list[dict]:
    """Segment without reading call ids at all.

    An environment executes every call of one response before returning any result, so a batch can
    only end at a tool result or a message. That holds whatever the producer stamps on a call id,
    which is what makes this the portable rule.

    Reasoning is transparent: a reasoning item does not end a model call. Sources exist that emit
    reasoning between the calls of one response, and treating that as a turn break splits a batch.
    """
    turns: list[dict] = []
    current: Optional[dict] = None
    pending: list[dict] = []

    for index, item in enumerate(items):
        kind = item.get("type")
        if kind == "reasoning":
            (current["body"] if current else pending).append(item)
            continue
        if kind == "function_call":
            if current is None:
                current = {"start": index, "lead": list(pending), "body": [], "calls": [], "message": None}
                pending = []
                turns.append(current)
            current["calls"].append(item)
            current["body"].append(item)
        elif kind == "message" and item.get("role") == "assistant":
            current = {"start": index, "lead": list(pending), "body": [item], "calls": [], "message": item}
            pending = []
            turns.append(current)
        else:
            current = None
            pending = []
    return turns


def segment_by_call_id_index(items: list[dict], pattern: re.Pattern) -> Optional[list[dict]]:
    """Segment using a producer-stamped call-id index that restarts at 0 for each model call.

    Returns None when the ids do not carry an index, which is the common case: many producers use
    opaque ids. That is a SKIP, not a pass.
    """
    calls = [item for item in items if item.get("type") == "function_call"]
    if not calls or not all(pattern.match(item.get("call_id") or "") for item in calls):
        return None

    turns: list[dict] = []
    current: Optional[dict] = None
    pending: list[dict] = []
    for index, item in enumerate(items):
        kind = item.get("type")
        if kind == "reasoning":
            (current["body"] if current else pending).append(item)
            continue
        if kind == "function_call":
            call_index = int(pattern.match(item["call_id"]).group("index"))
            if current is not None and current["calls"] and call_index == 0:
                current = None
            if current is None:
                current = {"start": index, "lead": list(pending), "body": [], "calls": [], "message": None}
                pending = []
                turns.append(current)
            current["calls"].append(item)
            current["body"].append(item)
        elif kind == "message" and item.get("role") == "assistant":
            current = {"start": index, "lead": list(pending), "body": [item], "calls": [], "message": item}
            pending = []
            turns.append(current)
        else:
            current = None
            pending = []
    return turns


def turn_signature(turns: list[dict], items: list[dict]) -> list[tuple]:
    positions = {id(item): index for index, item in enumerate(items)}
    return [
        (
            positions.get(id(turn["message"])) if turn["message"] is not None else None,
            tuple(positions[id(call)] for call in turn["calls"]),
        )
        for turn in turns
    ]


def lead_reasoning(turn: dict) -> list[dict]:
    """Reasoning emitted before the call acted."""
    return turn["lead"]


def inline_reasoning(turn: dict) -> list[dict]:
    """Reasoning recorded between the calls of one response."""
    return [item for item in turn["body"] if item.get("type") == "reasoning"]


# ----------------------------------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------------------------------


def signature(item: dict) -> tuple:
    """Identity of a conversation item, ignoring ids and any reordering a repair may have done."""
    kind = item.get("type")
    if kind == "function_call":
        return ("function_call", item.get("call_id"), item.get("name"), item.get("arguments"))
    if kind == "function_call_output":
        return ("function_call_output", item.get("call_id"), item.get("output"))
    if kind == "message":
        content = item.get("content")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        return ("message", item.get("role"), content)
    return (kind,)


def labelled_calls(action: dict) -> list[dict]:
    if action.get("type") == "function_call":
        return [action]
    if action.get("type") == "function_call_batch":
        return action.get("calls") or []
    return []


def message_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                return block.get("text") or ""
    return ""


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------


class Report:
    def __init__(self, max_problems: int) -> None:
        self.metrics: Counter = Counter()
        self.problems: list[str] = []
        self.max_problems = max_problems
        self.skipped: dict[int, str] = {}

    def fail(self, check: int, message: str) -> None:
        self.metrics["problems"] += 1
        self.metrics["check_failed", check] += 1
        if len(self.problems) < self.max_problems:
            self.problems.append(f"[check {check}] {message}")

    def skip(self, check: int, reason: str) -> None:
        self.skipped.setdefault(check, reason)
        self.metrics["check_skipped", check] += 1

    def note(self, key: str, count: int = 1) -> None:
        self.metrics[key] += count


CHECK_NAMES = {
    1: "one pivot per model call",
    2: "the label is that turn's calls, in full and in order",
    3: "the prefix is exactly the trajectory before this turn",
    4: "the prefix starts with this row's own seed",
    5: "no prefix ends on a function_call",
    6: "every source call is labelled exactly once",
    7: "call-id and execution-run segmentations agree",
    8: "no tool result interleaved inside a batch",
    9: "the prefix holds exactly the reasoning of earlier turns",
    10: "reference_output is the model call verbatim",
    11: "turn count agrees with the recorded model-call count",
    12: "at most one reasoning item precedes the action",
    13: "every reasoning input item carries an id",
}


# ----------------------------------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------------------------------


def diagnose_source(items: list[dict], pattern: re.Pattern, report: Report) -> list[dict]:
    turns = segment_by_execution_run(items)
    report.note("turns", len(turns))

    by_index = segment_by_call_id_index(items, pattern)
    if by_index is None:
        report.skip(7, "call ids carry no index (opaque ids)")
    elif turn_signature(by_index, items) != turn_signature(turns, items):
        report.fail(7, "the two segmentations disagree, so the batch boundaries are not trustworthy")

    for turn in turns:
        leads = lead_reasoning(turn)
        inline = inline_reasoning(turn)
        if len(leads) > 1:
            # A call thinks once before acting. More than one lead item means reasoning belonging to
            # a later call was recorded in front of this one.
            report.note("hoisted_turns")
            report.note("hoisted_surplus_items", len(leads) - 1)
        if inline:
            report.note("interleaved_turns")
            report.note("interleaved_items", len(inline))
        if turn["calls"]:
            report.note(f"batch_size_{len(turn['calls'])}")
        else:
            report.note("message_turns")

    # A tool result between the calls of one response would mean the execution-run rule split a real
    # batch. Reported rather than failed here because it describes the source, not the pivot file.
    seen_output = False
    for item in items:
        kind = item.get("type")
        if kind == "function_call_output":
            seen_output = True
        elif kind == "function_call" and seen_output:
            seen_output = False
        elif kind not in ("reasoning", "function_call"):
            seen_output = False

    return turns


def audit_trajectory(source_row: dict, pivots: list[dict], turns: list[dict], args, report: Report) -> None:
    items = get_path(source_row, args.source_output_path) or []
    seed = get_path(source_row, args.source_seed_path) or []
    seed_signature = [signature(item) for item in seed]

    # check 11 -- the harness's own count, if the source records one
    recorded = get_path(source_row, args.model_call_count_field)
    if recorded is None:
        report.skip(11, f"source has no {args.model_call_count_field}")
    else:
        allowed = (
            {len(turns), len(turns) + 1}
            if args.seeded_greeting == "auto"
            else ({len(turns) + 1} if args.seeded_greeting == "yes" else {len(turns)})
        )
        if recorded not in allowed:
            report.fail(11, f"segmented {len(turns)} model calls, source recorded {recorded}")
        else:
            report.note("turn_count_agrees")

    if len(pivots) != len(turns):
        report.fail(1, f"{len(pivots)} pivots for {len(turns)} reconstructed model calls")
        return

    labelled_call_ids: list[str] = []
    cumulative_own_reasoning = 0

    for depth, (pivot, turn) in enumerate(zip(pivots, turns)):
        action = pivot.get("expected_action") or {}
        expected = labelled_calls(action)
        prefix = (pivot.get("responses_create_params") or {}).get("input") or []

        # check 2
        if len(expected) != len(turn["calls"]):
            report.fail(2, f"depth {depth}: label has {len(expected)} calls, the turn made {len(turn['calls'])}")
        else:
            for labelled, source_call in zip(expected, turn["calls"]):
                if labelled.get("name") != source_call.get("name") or labelled.get("arguments") != source_call.get(
                    "arguments"
                ):
                    report.fail(2, f"depth {depth}: a label call does not match the source call")

        # check 4 -- this row's own seed, whatever its length
        if len(prefix) < len(seed) or [signature(item) for item in prefix[: len(seed)]] != seed_signature:
            report.fail(4, f"depth {depth}: the prefix does not start with this row's seed")

        # check 3 -- compared on non-reasoning items, so a reasoning repair does not trip it
        prefix_body = [signature(item) for item in prefix[len(seed) :] if item.get("type") != "reasoning"]
        source_body = [signature(item) for item in items[: turn["start"]] if item.get("type") != "reasoning"]
        if prefix_body != source_body:
            report.fail(
                3,
                f"depth {depth}: prefix holds {len(prefix_body)} non-reasoning items, "
                f"the trajectory has {len(source_body)} before this turn",
            )

        # check 5
        if prefix and prefix[-1].get("type") == "function_call":
            report.fail(5, f"depth {depth}: prefix ends on a function_call, the signature of a split batch")
        report.note("boundary_" + (source_body[-1][0] if source_body else "trajectory_start"))

        # check 13
        for item in prefix:
            if item.get("type") == "reasoning" and not item.get("id"):
                report.fail(13, f"depth {depth}: a reasoning input item has no id")
                break

        # check 9 -- counted from the rows, since a repair may move reasoning between turns
        own_reasoning = get_path(pivot, args.own_reasoning_field)
        if own_reasoning is None:
            report.skip(9, f"pivot rows have no {args.own_reasoning_field}")
        else:
            prefix_reasoning = sum(1 for item in prefix if item.get("type") == "reasoning")
            if prefix_reasoning != cumulative_own_reasoning:
                report.fail(
                    9,
                    f"depth {depth}: prefix holds {prefix_reasoning} reasoning items, "
                    f"the {depth} earlier turns emitted {cumulative_own_reasoning}",
                )
            cumulative_own_reasoning += own_reasoning

        # checks 10 and 12
        reference = get_path(pivot, args.reference_output_field)
        if reference is None:
            report.skip(10, f"pivot rows have no {args.reference_output_field}")
            report.skip(12, f"pivot rows have no {args.reference_output_field}")
        elif not isinstance(reference, list) or not reference:
            report.fail(10, f"depth {depth}: {args.reference_output_field} is empty or not a list")
        else:
            kinds = [item.get("type") for item in reference]
            stray = [kind for kind in kinds if kind not in ("reasoning", "message", "function_call")]
            if stray:
                report.fail(10, f"depth {depth}: reference output holds environment items {stray[:3]}")
            reference_calls = [item for item in reference if item.get("type") == "function_call"]
            if len(reference_calls) != len(expected):
                report.fail(
                    10, f"depth {depth}: reference output has {len(reference_calls)} calls, label has {len(expected)}"
                )
            if action.get("type") == "message":
                texts = [message_text(item) for item in reference if item.get("type") == "message"]
                if not texts or texts[0] != action.get("content"):
                    report.fail(10, f"depth {depth}: reference output message does not match the chat label")
            lead = 0
            for item in reference:
                if item.get("type") != "reasoning":
                    break
                lead += 1
            if lead > 1:
                report.fail(12, f"depth {depth}: {lead} reasoning items precede the action")
            if any(item.get("type") == "reasoning" for item in reference[lead:]):
                report.fail(12, f"depth {depth}: reasoning follows the action inside the model call")

        labelled_call_ids.extend(call.get("call_id") for call in turn["calls"])

    # check 6
    source_call_ids = [item.get("call_id") for item in items if item.get("type") == "function_call"]
    if Counter(labelled_call_ids) != Counter(source_call_ids):
        missing = Counter(source_call_ids) - Counter(labelled_call_ids)
        extra = Counter(labelled_call_ids) - Counter(source_call_ids)
        report.fail(6, f"calls missing {dict(missing)} / duplicated {dict(extra)}")


def iter_pivot_groups(path: Path, field: str) -> Iterator[list[dict]]:
    """Group consecutive pivot rows by their trajectory id, falling back to file order."""
    group: list[dict] = []
    current: Any = object()
    for _, row in iter_jsonl(path):
        key = get_path(row, field)
        if key != current and group:
            yield group
            group = []
        current = key
        group.append(row)
    if group:
        yield group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-path", type=Path, required=True, help="Source rollout JSONL")
    parser.add_argument("--pivot-path", type=Path, help="Pivot JSONL; omit for source diagnosis only")
    parser.add_argument("--source-output-path", default="response.output", help="Dotted path to the output item list")
    parser.add_argument("--source-seed-path", default="responses_create_params.input", help="Dotted path to the seed")
    parser.add_argument("--trajectory-id-field", default="source_info.trajectory_id")
    parser.add_argument("--model-call-count-field", default="num_agent_calls")
    parser.add_argument(
        "--seeded-greeting",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether the recorded model-call count includes a scripted opening turn that lives in the seed",
    )
    parser.add_argument("--own-reasoning-field", default="pivot_info.num_own_reasoning_items")
    parser.add_argument("--reference-output-field", default="reference_output")
    parser.add_argument("--call-id-pattern", default=r"^(?P<name>.+):(?P<index>\d+)$")
    parser.add_argument("--limit", type=int, help="Only read this many source trajectories")
    parser.add_argument("--max-problems", type=int, default=20)
    args = parser.parse_args()

    pattern = re.compile(args.call_id_pattern)
    report = Report(args.max_problems)
    groups = iter_pivot_groups(args.pivot_path, args.trajectory_id_field) if args.pivot_path else None

    for index, (_, source_row) in enumerate(iter_jsonl(args.source_path)):
        if args.limit is not None and index >= args.limit:
            break
        report.note("trajectories")
        items = get_path(source_row, args.source_output_path) or []
        turns = diagnose_source(items, pattern, report)

        if groups is not None:
            pivots = next(groups, None)
            if pivots is None:
                report.fail(1, f"trajectory {index}: pivot file ended early")
                break
            audit_trajectory(source_row, pivots, turns, args, report)

    metrics = report.metrics
    print(f"trajectories: {metrics['trajectories']}")
    print(f"model calls:  {metrics['turns']}")
    if metrics["turn_count_agrees"]:
        print(f"turn count agrees with the recorded model-call count: {metrics['turn_count_agrees']}")
    print("batch sizes:")
    sizes = sorted(
        (int(key.rsplit("_", 1)[1]), count) for key, count in metrics.items() if str(key).startswith("batch_size_")
    )
    for size, count in sizes:
        print(f"  {size} call{'s' if size > 1 else ''}: {count}")
    print(f"  message-only turns: {metrics['message_turns']}")
    print("reasoning placement:")
    print(
        f"  turns with reasoning hoisted from a later call: {metrics['hoisted_turns']} "
        f"({metrics['hoisted_surplus_items']} surplus items)"
    )
    print(
        f"  turns with reasoning interleaved between calls: {metrics['interleaved_turns']} "
        f"({metrics['interleaved_items']} items)"
    )
    if args.pivot_path:
        print("boundaries land on:")
        for key, count in sorted(metrics.items(), key=lambda item: str(item[0])):
            if str(key).startswith("boundary_"):
                print(f"  {str(key)[len('boundary_') :]}: {count}")

    for check, reason in sorted(report.skipped.items()):
        print(f"check {check} ({CHECK_NAMES[check]}): {SKIPPED} -- {reason}")

    if report.problems:
        print(f"\nFAILED: {metrics['problems']} problems", file=sys.stderr)
        for problem in report.problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    if args.pivot_path:
        print("\nEvery pivot splits the trajectory on a whole model call, and none holds its own reasoning.")
    else:
        print("\nSource diagnosis complete. Compare these counts against another model through the same harness.")


if __name__ == "__main__":
    main()
