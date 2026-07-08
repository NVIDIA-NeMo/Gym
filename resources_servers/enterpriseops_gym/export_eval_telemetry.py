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
"""Export rollouts to the eval team's per-turn telemetry schema (one JSONL row per turn).

Input rollouts must be collected with ``responses_api_agents/turn_logging_agent`` (rows
carry per-turn ``turns`` records with token usage, cached-token counts, timestamps, and
tool-call names) against the ``enterpriseops_gym`` resources server (rows carry
``tool_latencies_ms``).

Field notes:
- ``stacked_input`` equals ``input_length``: this harness's ReAct loop grows context
  monotonically (no compaction/summarization), so each turn's prompt IS the full stack.
- ``tokensBefore``/``tokensAfter``/``content before/after compaction`` are emitted as
  null for the same reason — no compaction event ever occurs. Flagged for the eval team.
- ``per-tool latency`` is emitted as this turn's total tool-execution milliseconds (int),
  with per-call detail in the supplementary ``tool_latency_detail`` field. Latencies are
  measured by the resources server's MCP proxy; model-side malformed tool calls never
  reach it and align to null.
- ``output_length`` includes reasoning tokens (API completion-token accounting), and
  ``answer`` includes any reasoning-item summary text present in the turn's output slice.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def render_question(responses_create_params: Dict[str, Any]) -> str:
    """Render the task prompt (system + user messages) as readable text."""
    parts = []
    for message in responses_create_params.get("input") or []:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def extract_turn_answer(turn: Dict[str, Any], output_items: List[Dict[str, Any]]) -> str:
    """Assistant text for the turn, prefixed with any reasoning summaries in its slice."""
    start, end = turn.get("output_start_index"), turn.get("output_end_index")
    reasoning_texts: List[str] = []
    if start is not None and end is not None:
        for item in output_items[start:end]:
            if item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    text = summary.get("text") if isinstance(summary, dict) else None
                    if text:
                        reasoning_texts.append(text)
    answer = turn.get("assistant_text") or ""
    if reasoning_texts:
        answer = "\n".join(reasoning_texts) + ("\n\n" + answer if answer else "")
    return answer


def align_tool_latencies(
    turns: List[Dict[str, Any]], tool_latencies: List[Dict[str, Any]]
) -> List[List[Optional[Dict[str, Any]]]]:
    """Align session-ordered tool latencies to turns by walking both in execution order.

    A tool call whose arguments failed to parse never reaches the resources server, so it
    has no latency record — those align to None rather than shifting later turns.
    """
    aligned: List[List[Optional[Dict[str, Any]]]] = []
    pointer = 0
    for turn in turns:
        turn_latencies: List[Optional[Dict[str, Any]]] = []
        for tool_name in turn.get("tool_call_names") or []:
            if pointer < len(tool_latencies) and tool_latencies[pointer].get("tool") == tool_name:
                turn_latencies.append(tool_latencies[pointer])
                pointer += 1
            else:
                turn_latencies.append(None)
        aligned.append(turn_latencies)
    return aligned


def export_rows(rollout_rows: List[Dict[str, Any]], total_passes: Optional[int] = None) -> List[Dict[str, Any]]:
    # Passes per task (for trial_name) default to the observed repeat count.
    passes_by_task: Dict[Any, int] = {}
    for row in rollout_rows:
        task_index = row.get("_ng_task_index")
        passes_by_task[task_index] = passes_by_task.get(task_index, 0) + 1

    telemetry_rows: List[Dict[str, Any]] = []
    for row in rollout_rows:
        turns = row.get("turns")
        if turns is None:
            raise ValueError(
                "Rollout row has no per-turn 'turns' records. Collect rollouts with "
                "responses_api_agents/turn_logging_agent (see enterpriseops_gym_turnlog.yaml)."
            )

        metadata = row.get("verifier_metadata") or {}
        task_id = metadata.get("task_id", f"task_{row.get('_ng_task_index')}")
        rollout_index = row.get("_ng_rollout_index", 0)
        num_passes = total_passes or passes_by_task.get(row.get("_ng_task_index"), 1)
        trial_name = f"{task_id}.{rollout_index + 1}-of-{num_passes}.default"

        question = render_question(row.get("responses_create_params") or {})
        output_items = (row.get("response") or {}).get("output") or []
        is_resolved = row.get("reward", 0.0) == 1.0
        latency_per_turn = align_tool_latencies(turns, row.get("tool_latencies_ms") or [])

        for i, turn in enumerate(turns):
            is_final_turn = i == len(turns) - 1
            input_length = turn.get("input_tokens", 0)
            cached = turn.get("cached_input_tokens", 0)
            turn_latencies = latency_per_turn[i]
            known_latencies = [entry["latency_ms"] for entry in turn_latencies if entry]

            telemetry_rows.append(
                {
                    "task_id": task_id,
                    "trial_name": trial_name,
                    "turn": turn.get("turn", i),
                    "num_turns": len(turns),
                    "input_length": input_length,
                    "output_length": turn.get("output_tokens", 0),
                    "stacked_input": input_length,  # no compaction: prompt == full stack
                    "is_resolved": is_resolved,
                    "task_complete": bool(
                        is_final_turn and not turn.get("tool_call_names") and turn.get("assistant_text")
                    ),
                    "num_steps": turn.get("num_tool_calls", 0),
                    "timestamp": turn.get("timestamp"),
                    "question": question,
                    "answer": extract_turn_answer(turn, output_items),
                    "cached input length": cached,
                    "new input length": input_length - cached,
                    "tool_call names": ",".join(turn.get("tool_call_names") or []),
                    "per-tool latency": int(sum(known_latencies)) if known_latencies else None,
                    "tokensBefore": None,  # no compaction in this harness
                    "tokensAfter": None,
                    "content before compaction": None,
                    "content after compaction": None,
                    # Supplementary (beyond the required schema)
                    "tool_latency_detail": turn_latencies,
                    "model": turn.get("model"),
                    "turn_duration_ms": turn.get("duration_ms"),
                    "domain": metadata.get("domain"),
                    "mode": metadata.get("mode"),
                    "reward": row.get("reward"),
                }
            )
    return telemetry_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Rollouts JSONL from the turn_logging_agent")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-passes", type=int, default=None, help="Override {total_passes} in trial_name")
    args = parser.parse_args()

    rollout_rows = [json.loads(line) for line in open(args.input)]
    telemetry_rows = export_rows(rollout_rows, total_passes=args.total_passes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in telemetry_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(telemetry_rows)} turn rows ({len(rollout_rows)} trials) to {args.output}")


if __name__ == "__main__":
    main()
