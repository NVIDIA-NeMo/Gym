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
"""Reference script: Responses API rollout artifacts to pivot rows, including parallel tool calls.

The other reference converters all take a **message list** as input. This one takes a Gym rollout
artifact whose trajectory is a flat list of Responses API output items -- reasoning, tool calls,
tool results, assistant text and user turns interleaved in one sequence. That shape is where the
work is, because "one model call" is not delimited for you and has to be reconstructed.

Unlike the message-list converters, this one is self-contained: standard library only, no external
project imports, so it runs as written.

What it demonstrates, all of which is covered in ``references/rollout-artifact-pitfalls.md``:

* Reconstructing one model call from a flat output list, and cross-checking the count against a
  model-call count the source recorded.
* Labelling the whole call: ``message`` / ``function_call`` / ``function_call_batch``, with tool
  calls taking precedence over assistant text.
* Keeping earlier turns' reasoning in the prefix while withholding the pivot's own, and repairing
  two recording artifacts that break that (``--no-repair-reasoning`` to see the raw behaviour).
* Carrying ``reference_output``: the model call verbatim, next to the reduced label.

Every source-shape assumption is a flag. Point the field paths at your own producer's names.

    python responses_output_to_pivot_dataset_reference.py \\
        --in-path rollouts.jsonl --out-path pivot.jsonl --agent-ref my_pivot_agent
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional


def get_path(row: dict, dotted: str) -> Any:
    current: Any = row
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class Turn:
    """One reconstructed model call: what it thought, then what it did."""

    __slots__ = ("start_index", "lead", "body", "calls", "message")

    def __init__(self, start_index: int) -> None:
        self.start_index = start_index
        # Reasoning emitted before the call acted. Held apart from `body` because some producers
        # record several turns' reasoning in front of the first one; see repair_reasoning.
        self.lead: list[dict] = []
        # The action, plus anything interleaved with it.
        self.body: list[dict] = []
        self.calls: list[dict] = []
        self.message: Optional[dict] = None

    @property
    def items(self) -> list[dict]:
        return [*self.lead, *self.body]

    @property
    def kind(self) -> str:
        if not self.calls:
            return "chat"
        return "single_tool_call" if len(self.calls) == 1 else "parallel_tool_calls"


def segment_turns(output_items: list[dict]) -> list[Turn]:
    """Split a flat output list into the model calls that produced it.

    The rule needs nothing from the call ids: an environment executes every call of one response
    before returning any result, so a tool-call batch can only end at a tool result or a message.
    Producers that stamp an index into the call id give you a second opinion for free, but many use
    opaque ids, so this is the portable rule.

    Reasoning never ends a model call. Some producers emit reasoning between the calls of one
    response, and treating that as a boundary splits a batch that was a single generation.
    """
    turns: list[Turn] = []
    current: Optional[Turn] = None
    pending: list[dict] = []
    pending_start = 0

    for index, item in enumerate(output_items):
        kind = item.get("type")

        if kind == "reasoning":
            (current.body if current is not None else pending).append(item)
            continue

        if kind == "function_call":
            if current is None:
                current = Turn(pending_start if pending or pending_start == index else index)
                current.lead = pending
                pending = []
                turns.append(current)
            current.calls.append(item)
            current.body.append(item)

        elif kind == "message" and item.get("role") == "assistant":
            # An assistant message opens a call; calls that follow with no environment item in
            # between came from that same response, so the turn stays open.
            current = Turn(pending_start if pending or pending_start == index else index)
            current.lead = pending
            current.message = item
            current.body.append(item)
            pending = []
            turns.append(current)

        else:
            # A tool result or a user turn: the environment spoke, so the model call is over.
            current = None
            pending = []
            pending_start = index + 1

    return turns


def repair_reasoning(turns: list[Turn], metrics: Counter) -> None:
    """Put reasoning back with the call that produced it.

    Two recording artifacts show up in real artifacts. Reasoning recorded *between* the calls of one
    response carries no information, because the calls are one generation. And some producers hoist
    several turns' reasoning in front of the first turn's action, which is a forward leak: those
    items land in the prefix of every later pivot, handing it conclusions drawn from tool results it
    has not seen.

    Diagnose before trusting this: run the same source diagnosis over two different models through
    the same harness. An anomaly that appears for one and not the other is an artifact of the
    recording, not model behaviour. This repair is a heuristic, so it reports what it moved.
    """
    for turn in turns:
        inline = [item for item in turn.body if item.get("type") == "reasoning"]
        if inline:
            metrics["interleaved_reasoning_normalized"] += len(inline)
            turn.body = [item for item in turn.body if item.get("type") != "reasoning"]
            turn.lead = [*turn.lead, *inline]

    # A call thinks once before acting, so lead reasoning past the first belongs to a later call.
    queue: list[dict] = []
    for turn in turns:
        if len(turn.lead) > 1:
            metrics["hoisted_turns_repaired"] += 1
            metrics["hoisted_surplus_items"] += len(turn.lead) - 1
            queue.extend(turn.lead[1:])
            turn.lead = turn.lead[:1]
        elif not turn.lead and queue:
            turn.lead = [queue.pop(0)]
            metrics["reasoning_reattributed"] += 1
    # Surplus with nowhere to land is dropped: keeping it preserves the leak it was queued to fix.
    metrics["reasoning_dropped_unattributable"] += len(queue)


def message_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                return block.get("text") or ""
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""


def to_input_item(item: dict, keep_reasoning: bool, fallback_index: int) -> Optional[dict]:
    """Rebuild one trajectory item as a Responses API *input* item.

    Rebuilt rather than copied, so nulls the artifact materialized never reach the pivot row.
    """
    kind = item.get("type")

    if kind == "reasoning":
        if not keep_reasoning:
            return None
        # `id` is required on a reasoning input item; the request model rejects it otherwise.
        return {
            "type": "reasoning",
            "id": item.get("id") or f"rs_{fallback_index}",
            "summary": item.get("summary") or [],
            "status": "completed",
        }

    if kind == "message":
        if item.get("role") == "assistant":
            rebuilt = {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"annotations": [], "type": "output_text", "text": message_text(item)}],
            }
            if item.get("id"):
                rebuilt["id"] = item["id"]
            return rebuilt
        return {"type": "message", "role": item.get("role") or "user", "content": message_text(item)}

    if kind == "function_call":
        rebuilt = {
            "type": "function_call",
            "call_id": item["call_id"],
            "name": item["name"],
            "arguments": item["arguments"],
            "status": "completed",
        }
        if item.get("id"):
            rebuilt["id"] = item["id"]
        return rebuilt

    if kind == "function_call_output":
        output = item.get("output")
        return {
            "type": "function_call_output",
            "call_id": item["call_id"],
            "output": output if isinstance(output, str) else json.dumps(output),
        }

    return None


def build_stream(output_items: list[dict], turns: list[Turn]) -> list[dict]:
    """Re-emit the trajectory with each turn's reasoning in front of the call that owns it.

    The repair moves reasoning between turns, so the raw order no longer says who thought what. Each
    lead item is lifted from its original position and re-inserted before its turn's first action,
    which also leaves every turn occupying one contiguous span.
    """
    lifted = {id(item) for turn in turns for item in turn.lead}
    opens = {id(turn.body[0]): turn for turn in turns if turn.body}

    stream: list[dict] = []
    for item in output_items:
        if id(item) in lifted:
            continue
        turn = opens.get(id(item))
        if turn is not None:
            turn.start_index = len(stream)
            stream.extend(turn.lead)
        stream.append(item)
    return stream


def build_messages(seed: list[dict], stream: list[dict], keep_reasoning: bool) -> tuple[list[dict], list[int]]:
    """Convert the whole trajectory once; record where each stream index lands.

    `boundary_at[i]` is the prefix length for a pivot starting at stream index `i`, so the prefix is
    a slice rather than a re-conversion.
    """
    messages = deepcopy(seed)
    boundary_at = [len(messages)]
    for index, item in enumerate(stream):
        converted = to_input_item(item, keep_reasoning, index)
        if converted is not None:
            messages.append(converted)
        boundary_at.append(len(messages))
    return messages, boundary_at


def expected_action(turn: Turn) -> Optional[dict]:
    """Label the whole model call.

    Tool calls take precedence over assistant text, mirroring the response-side normalizer the
    verifier runs. One call is a `function_call`; two or more are one `function_call_batch`. A
    one-element batch is never emitted -- the response side cannot produce that shape.
    """
    if not turn.calls:
        return {"type": "message", "content": message_text(turn.message) if turn.message else ""}

    calls = []
    for call in turn.calls:
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            return None
        try:
            json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return None
        calls.append({"type": "function_call", "name": call["name"], "arguments": arguments})

    if len(calls) == 1:
        return calls[0]
    return {"type": "function_call_batch", "calls": calls}


def convert(args: argparse.Namespace) -> Counter:
    start_time = time.time()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent_ref = {"type": "responses_api_agents", "name": args.agent_ref}
    metrics: Counter = Counter()

    with open(args.in_path) as source, out_path.open("w") as target:
        for trajectory_id, line in enumerate(source):
            if not line.strip():
                continue
            if args.limit_trajectories and trajectory_id >= args.limit_trajectories:
                break
            metrics["trajectories"] += 1
            row = json.loads(line)

            seed = get_path(row, args.seed_path) or []
            output_items = get_path(row, args.output_path) or []
            tools = get_path(row, args.tools_path) or []

            turns = segment_turns(output_items)
            if args.repair_reasoning:
                repair_reasoning(turns, metrics)
            stream = build_stream(output_items, turns)
            messages, boundary_at = build_messages(seed, stream, args.keep_reasoning)

            recorded = get_path(row, args.model_call_count_field)
            if recorded is not None and recorded not in (len(turns), len(turns) + 1):
                # Not fatal, but it means the segmentation and the producer disagree about what a
                # model call is. Investigate before trusting the output.
                metrics["turn_count_disagrees"] += 1

            for depth, turn in enumerate(turns):
                action = expected_action(turn)
                if action is None:
                    metrics["skipped_malformed_arguments"] += 1
                    continue
                if turn.kind == "chat" and args.drop_chat_pivots:
                    metrics["skipped_chat_pivots"] += 1
                    continue

                pivot_index = boundary_at[turn.start_index]
                pivot = {
                    "responses_create_params": {
                        "input": messages[:pivot_index],
                        "tools": tools,
                        "tool_choice": "auto",
                        # Must be true: a batch label asks the policy for several calls at once.
                        "parallel_tool_calls": True,
                    },
                    "expected_action": action,
                    # The model call verbatim, beside the reduced label: the narration a
                    # narrate-and-call label drops, and the reasoning the prefix withholds.
                    "reference_output": [to_input_item(item, True, index) for index, item in enumerate(turn.items)],
                    "pivot_info": {
                        "kind": turn.kind,
                        "num_expected_tool_calls": len(turn.calls),
                        "expected_tool_names": [call["name"] for call in turn.calls],
                        "narrates_and_calls": bool(turn.message) and bool(turn.calls),
                        "pivot_depth": depth,
                        "trajectory_num_turns": len(turns),
                        "num_input_items": pivot_index,
                        "num_own_reasoning_items": sum(1 for item in turn.items if item.get("type") == "reasoning"),
                    },
                    "source_info": {"trajectory_id": trajectory_id},
                    "agent_ref": agent_ref,
                }
                for field in args.carry_field:
                    pivot["source_info"][field.split(".")[-1]] = get_path(row, field)

                target.write(json.dumps(pivot) + "\n")
                metrics["pivot_rows_written"] += 1
                metrics["kind", turn.kind] += 1
                if turn.calls:
                    metrics["batch_size", len(turn.calls)] += 1
                if turn.message and turn.calls:
                    metrics["narrates_and_calls"] += 1

    metrics["elapsed_seconds"] = int(time.time() - start_time)
    return metrics


def report(metrics: Counter) -> str:
    written = metrics["pivot_rows_written"] or 1
    lines = [
        f"trajectories:        {metrics['trajectories']}",
        f"pivot rows written:  {metrics['pivot_rows_written']}",
        "",
        "pivot kinds:",
        f"  parallel_tool_calls: {metrics['kind', 'parallel_tool_calls']} "
        f"({metrics['kind', 'parallel_tool_calls'] / written:.1%})",
        f"  single_tool_call:    {metrics['kind', 'single_tool_call']} "
        f"({metrics['kind', 'single_tool_call'] / written:.1%})",
        f"  chat:                {metrics['kind', 'chat']} ({metrics['kind', 'chat'] / written:.1%})",
        "",
        "batch sizes:",
    ]
    sizes = sorted(
        (key[1], count) for key, count in metrics.items() if isinstance(key, tuple) and key[0] == "batch_size"
    )
    lines += [f"  {size} call{'s' if size > 1 else ''}: {count}" for size, count in sizes]
    lines += [
        "",
        f"turns that narrated and called:      {metrics['narrates_and_calls']}",
        f"interleaved reasoning normalized:    {metrics['interleaved_reasoning_normalized']}",
        f"hoisted turns repaired:              {metrics['hoisted_turns_repaired']} "
        f"({metrics['hoisted_surplus_items']} surplus items, "
        f"{metrics['reasoning_reattributed']} reattributed, "
        f"{metrics['reasoning_dropped_unattributable']} dropped)",
        f"trajectories disagreeing with the recorded model-call count: {metrics['turn_count_disagrees']}",
        f"skipped, malformed arguments:        {metrics['skipped_malformed_arguments']}",
        f"skipped, chat pivots:                {metrics['skipped_chat_pivots']}",
        f"elapsed seconds:                     {metrics['elapsed_seconds']}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-path", required=True, help="Rollout artifacts, one JSON object per line")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--agent-ref", required=True, help="Must match the agent block in the generated config")
    parser.add_argument("--output-path", default="response.output", help="Dotted path to the flat output item list")
    parser.add_argument("--seed-path", default="responses_create_params.input", help="Dotted path to the seed prefix")
    parser.add_argument("--tools-path", default="responses_create_params.tools")
    parser.add_argument("--model-call-count-field", default="num_agent_calls")
    parser.add_argument(
        "--carry-field",
        action="append",
        default=[],
        help="Dotted source field to copy into source_info for provenance. May be repeated.",
    )
    parser.add_argument(
        "--drop-reasoning",
        dest="keep_reasoning",
        action="store_false",
        help="Strip earlier turns' reasoning from the prefix. Kept by default so a pivot sees the "
        "context the source agent had.",
    )
    parser.add_argument(
        "--no-repair-reasoning",
        dest="repair_reasoning",
        action="store_false",
        help="Leave recorded reasoning placement alone. Off by default the converter moves "
        "misplaced reasoning back to the call that produced it; see the repair_reasoning docstring.",
    )
    parser.add_argument("--drop-chat-pivots", action="store_true", help="Emit only tool-call pivots")
    parser.add_argument("--limit-trajectories", type=int, default=0)
    parser.set_defaults(keep_reasoning=True, repair_reasoning=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = convert(args)
    summary = report(metrics)
    print(summary)
    Path(args.out_path).with_suffix(".info.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
