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

"""Prepare prefix pass@K rows for AutomationBench (function-calling wire, resources-server tools).

A pool entry names one captured trajectory -- `run` (the source model's run directory), `task` (its task index in
`benchmarks/automationbench/data/automationbench-600.jsonl`) and `rep` -- and its decisive turn `T_first`: the first
turn that changes any assertion's verdict. The pool (`data/automationbench_pilot100.json`, 100 tasks stratified over
domain x T_first/N tercile x source model) was located by replaying every turn of every trajectory against the
rubric; the trajectories themselves live outside the repo.

Turn t of a trajectory is its t-th run of consecutive function calls. A turn carries the assistant text that opened
it, its calls, and their captured outputs; turns whose calls never ran (the max-turns cut) are dropped. Call ids are
renamed `call_<turn>_<i>` in calls and outputs alike: source ids give away the source model by their format, and some
embed long opaque signatures that a base model copies when it continues the transcript. Ids only pair a call with its
output, so the replay is unchanged.

Each row is the task's AutomationBench row (task_id, info) with:
  * responses_create_params: the task's prompt and tools, and no output budget (the model server's applies);
  * determinism: the record the trajectory started from, so its replay mints the same ids and timestamps;
  * prefix_pass_k: target_turn, max_forwards, prefix (turns 1..T-1), target (T), suffix (T+1..N, for the gold
    check), and the trajectory's source.

    PREFIX_PASS_K_AUTOMATIONBENCH_TRACES=/path/to/traces python -m benchmarks.prefix_pass_k.prepare_automationbench

`..._TRACES` is a comma-separated list of roots, each holding `<run>/rollouts.jsonl` (Gym rollouts of
`benchmarks/automationbench` run with `deterministic: true`, which export `_determinism_start`); earlier roots win,
so list a re-collection before the run it replaces. `PREFIX_PASS_K_AUTOMATIONBENCH_POOL`,
`PREFIX_PASS_K_AUTOMATIONBENCH_TASKS` and `PREFIX_PASS_K_OUTPUT` override the pool, the task rows and the output.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "prefix_pass_k_automationbench_benchmark.jsonl"
DEFAULT_POOL_FPATH = DATA_DIR / "automationbench_pilot100.json"
DEFAULT_TASKS_FPATH = BENCHMARK_DIR.parent / "automationbench" / "data" / "automationbench-600.jsonl"


def message_text(item: Dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content or [] if isinstance(part, dict))


def turns_of(output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group a rollout's output items into turns: {turn, content, tool_calls, observations}."""
    turns: List[Dict[str, Any]] = []
    text: List[str] = []
    prev = None
    outputs = {it.get("call_id"): it.get("output") for it in output if it.get("type") == "function_call_output"}
    for it in output:
        kind = it.get("type")
        if kind == "function_call":
            if prev != "function_call":
                turns.append({"turn": len(turns) + 1, "content": "".join(text) or None, "tool_calls": []})
                text = []
            turns[-1]["tool_calls"].append({"id": it["call_id"], "name": it["name"], "arguments": it["arguments"]})
        elif kind == "message" and it.get("role", "assistant") == "assistant":
            text.append(message_text(it))
        elif kind == "function_call_output":
            text = []  # text after a tool output belongs to the next turn
        prev = kind
    for turn in turns:
        turn["observations"] = [
            {"tool_call_id": c["id"], "content": outputs[c["id"]]} for c in turn["tool_calls"] if c["id"] in outputs
        ]
    return [t for t in turns if t["observations"]]


def neutral_ids(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rename every call id to call_<turn>_<i>, keeping each output paired with its call."""
    for turn in turns:
        renamed: Dict[str, str] = {}
        for i, call in enumerate(turn["tool_calls"]):
            new_id = f"call_{turn['turn']}_{i}"
            if call["id"] in renamed:
                raise ValueError(f"turn {turn['turn']}: duplicate call id {call['id'][:40]}")
            renamed[call["id"]] = new_id
            call["id"] = new_id
        for obs in turn["observations"]:
            obs["tool_call_id"] = renamed[obs["tool_call_id"]]
    return turns


def call_labels(turn: Dict[str, Any]) -> List[str]:
    """Each parseable call's label (`METHOD url` for api_fetch, else the tool name), to match the pool's gold_calls."""
    labels = []
    for call in turn["tool_calls"]:
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            continue
        if not isinstance(args, dict):
            continue
        if call["name"] == "api_fetch":
            labels.append(f"{str(args.get('method', 'GET')).upper()} {args.get('url', '')}")
        else:
            labels.append(call["name"])
    return labels


def as_step(turn: Dict[str, Any]) -> Dict[str, Any]:
    """A turn in the agent's step schema: chat-completions tool calls and their captured outputs, in call order."""
    captured = {o["tool_call_id"]: o["content"] for o in turn["observations"]}
    missing = [c["id"] for c in turn["tool_calls"] if c["id"] not in captured]
    if missing:
        raise ValueError(f"turn {turn['turn']}: calls {missing} have no captured output")
    return {
        "turn": turn["turn"],
        "content": turn["content"],
        "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in turn["tool_calls"]
        ],
        "captured_observations": [captured[c["id"]] for c in turn["tool_calls"]],
    }


def find_rollouts(pool: List[Dict[str, Any]], roots: List[Path]) -> Dict[tuple, Dict[str, Any]]:
    """The rollout behind each pool entry, from the first root that has it."""
    need: Dict[str, set] = {}
    for entry in pool:
        need.setdefault(entry["run"], set()).add((entry["task"], entry["rep"]))
    found: Dict[tuple, Dict[str, Any]] = {}
    for run, keys in need.items():
        for root in roots:
            path = root / run / "rollouts.jsonl"
            if not path.exists():
                continue
            with path.open() as f:
                for line in f:
                    rollout = json.loads(line)
                    key = (rollout["_ng_task_index"], rollout["_ng_rollout_index"])
                    if key in keys and (run, *key) not in found:
                        found[(run, *key)] = rollout
    return found


def build_row(entry: Dict[str, Any], task: Dict[str, Any], rollout: Dict[str, Any]) -> Dict[str, Any]:
    if task["info"]["task_name"] != entry["task_name"]:
        raise ValueError(f"task {entry['task']} is {task['info']['task_name']}, the pool says {entry['task_name']}")
    turns = neutral_ids(turns_of(rollout["response"]["output"]))
    t = entry["T_first"]
    if call_labels(turns[t - 1]) != entry["gold_calls"]:
        raise ValueError(f"{entry['run']} task {entry['task']} rep {entry['rep']}: turn {t} is not the pool's T_first")
    record = (rollout["response"].get("exported_state") or {}).get("_determinism_start")
    if not record:
        raise ValueError(f"{entry['run']} task {entry['task']} rep {entry['rep']}: the rollout exports no determinism")
    return {
        "task_idx": task["task_idx"],
        "task_id": task["task_id"],
        "example_id": task["example_id"],
        "info": copy.deepcopy(task["info"]),
        "responses_create_params": {
            "input": copy.deepcopy(task["responses_create_params"]["input"]),
            "tools": copy.deepcopy(task["responses_create_params"]["tools"]),
        },
        "determinism": record,
        "prefix_pass_k": {
            "target_turn": t,
            "n_prefix_turns": t - 1,
            "max_forwards": entry["max_forwards"],
            "prefix": [as_step(turn) for turn in turns[: t - 1]],
            "target": as_step(turns[t - 1]),
            "suffix": [as_step(turn) for turn in turns[t:]],
            "source": {"run": entry["run"], "task": entry["task"], "rep": entry["rep"], "N": entry["N"]},
        },
    }


def build(pool_fpath: Path, tasks_fpath: Path, roots: List[Path], output: Path) -> Path:
    pool = json.loads(pool_fpath.read_text())
    with tasks_fpath.open() as f:
        tasks = {row["task_idx"]: row for row in map(json.loads, f)}
    found = find_rollouts(pool, roots)
    missing = [(e["run"], e["task"], e["rep"]) for e in pool if (e["run"], e["task"], e["rep"]) not in found]
    if missing:
        raise SystemExit(f"{len(missing)} pool trajectories not found under {roots}, e.g. {missing[:3]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for entry in pool:
            row = build_row(entry, tasks[entry["task"]], found[(entry["run"], entry["task"], entry["rep"])])
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(pool)} rows -> {output}")
    return output


def prepare(output: Optional[Path] = None) -> Path:
    """Entry point for `gym eval prepare`; paths come from the environment (module docstring)."""
    traces = os.environ.get("PREFIX_PASS_K_AUTOMATIONBENCH_TRACES")
    if not traces:
        raise SystemExit(
            "Set PREFIX_PASS_K_AUTOMATIONBENCH_TRACES to the trajectory roots (see the module docstring)."
        )
    return build(
        Path(os.environ.get("PREFIX_PASS_K_AUTOMATIONBENCH_POOL", DEFAULT_POOL_FPATH)),
        Path(os.environ.get("PREFIX_PASS_K_AUTOMATIONBENCH_TASKS", DEFAULT_TASKS_FPATH)),
        [Path(root) for root in traces.split(",")],
        Path(output or os.environ.get("PREFIX_PASS_K_OUTPUT", OUTPUT_FPATH)),
    )


if __name__ == "__main__":
    prepare()
